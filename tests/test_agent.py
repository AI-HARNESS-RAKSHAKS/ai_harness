"""Tests for Agent.run and the optimisation layer (LLM is mocked)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harness.agent import Agent, TokenUsage
from harness.config import Config
from harness.llm import Completion


def _make_config(**overrides) -> Config:
    base = dict(
        api_key="test",
        base_url="https://example/v1",
        model="test-model",
        max_steps=5,
        max_output_tokens=512,
        history_window=10,
        max_tool_output_chars=4000,
        max_tool_output_lines=200,
        externalize_threshold=8000,
        token_budget=200000,
        loop_window=20,
        loop_identical_threshold=5,
        loop_tool_freq_threshold=50,
        parallel_tools=True,
        subagent_model=None,
        subagent_max_steps=15,
        subagent_default_max_turns=10,
    )
    base.update(overrides)
    return Config(**base)


def _msg(content="", tool_calls=None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_call(name: str, args: dict, call_id: str | None = None) -> dict:
    return {
        "id": call_id or f"call_{name}_{id(args)}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _completion(content="", tool_calls=None, usage=None) -> Completion:
    return Completion(
        message=_msg(content, tool_calls),
        usage=usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


@pytest.mark.asyncio
async def test_agent_runs_until_done():
    client = SimpleNamespace(
        chat=AsyncMock(side_effect=[
            _completion(content="", tool_calls=[_tool_call("done", {"answer": "all done"})]),
        ]),
        aclose=AsyncMock(),
    )
    agent = Agent(_make_config(), client)
    events = [ev async for ev in agent.run("hi")]
    types = [e.type for e in events]
    assert "user" in types
    assert "step" in types
    assert "done" in types
    assert agent.usage.input_tokens == 10
    assert agent.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_agent_runs_tool_loop():
    client = SimpleNamespace(
        chat=AsyncMock(side_effect=[
            _completion(tool_calls=[_tool_call("bash", {"command": "echo hi"})]),
            _completion(tool_calls=[_tool_call("done", {"answer": "ok"})]),
        ]),
        aclose=AsyncMock(),
    )
    agent = Agent(_make_config(), client)
    events = [ev async for ev in agent.run("test")]
    tool_call_evs = [e for e in events if e.type == "tool_call"]
    tool_result_evs = [e for e in events if e.type == "tool_result"]
    assert len(tool_call_evs) == 2
    assert len(tool_result_evs) == 2


@pytest.mark.asyncio
async def test_agent_stops_at_max_steps():
    client = SimpleNamespace(
        chat=AsyncMock(side_effect=[
            _completion(tool_calls=[_tool_call("bash", {"command": "true"})])
        ] * 10),
        aclose=AsyncMock(),
    )
    agent = Agent(_make_config(max_steps=3), client)
    events = [ev async for ev in agent.run("loop")]
    assert events[-1].type == "error"
    assert "max steps" in events[-1].payload["message"]


@pytest.mark.asyncio
async def test_agent_records_token_usage_with_cached():
    client = SimpleNamespace(
        chat=AsyncMock(return_value=_completion(usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 80},
        })),
        aclose=AsyncMock(),
    )
    agent = Agent(_make_config(), client)
    async for _ in agent.run("hi"):
        pass
    assert agent.usage.input_tokens == 100
    assert agent.usage.output_tokens == 20
    assert agent.usage.cached_input_tokens == 80
    assert agent.usage.calls == 1


@pytest.mark.asyncio
async def test_history_compaction_keeps_original_task_and_recent():
    client = SimpleNamespace(
        chat=AsyncMock(side_effect=[_completion(content="ok")] * 5),
        aclose=AsyncMock(),
    )
    agent = Agent(_make_config(history_window=6, max_steps=10), client)
    agent.messages = [{"role": "user", "content": "original task"}]
    for i in range(20):
        agent.messages.append({"role": "assistant", "content": f"step {i} response"})
        agent.messages.append({"role": "user", "content": f"step {i} follow-up"})

    async for _ in agent.run("original task"):
        pass
    assert agent.messages[0] == {"role": "user", "content": "original task"}
    assert len(agent.messages) <= 7


@pytest.mark.asyncio
async def test_loop_detection_terminates_run():
    client = SimpleNamespace(
        chat=AsyncMock(side_effect=[
            _completion(tool_calls=[_tool_call("bash", {"command": "ls"})])
        ] * 20),
        aclose=AsyncMock(),
    )
    agent = Agent(_make_config(loop_identical_threshold=3, max_steps=20), client)
    events = [ev async for ev in agent.run("loop")]
    assert events[-1].type == "error"
    assert "loop" in events[-1].payload["message"].lower()


@pytest.mark.asyncio
async def test_token_budget_terminates_run():
    client = SimpleNamespace(
        chat=AsyncMock(return_value=_completion(
            tool_calls=[_tool_call("bash", {"command": "true"})],
            usage={"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550},
        )),
        aclose=AsyncMock(),
    )
    agent = Agent(_make_config(token_budget=600, max_steps=10), client)
    events = [ev async for ev in agent.run("hi")]
    types = [e.type for e in events]
    assert "error" in types
    budget_errors = [e for e in events if e.type == "error" and "budget" in e.payload["message"]]
    assert budget_errors


@pytest.mark.asyncio
async def test_parallel_tool_execution_runs_all():
    async def fake_chat(messages, system):
        if any(m.get("role") == "tool" for m in messages):
            return _completion(tool_calls=[_tool_call("done", {"answer": "ok"})])
        return _completion(tool_calls=[
            _tool_call("bash", {"command": "echo 1"}, call_id="a"),
            _tool_call("bash", {"command": "echo 2"}, call_id="b"),
            _tool_call("bash", {"command": "echo 3"}, call_id="c"),
        ])

    client = SimpleNamespace(chat=AsyncMock(side_effect=fake_chat), aclose=AsyncMock())
    agent = Agent(_make_config(), client)
    events = [ev async for ev in agent.run("hi")]
    tool_call_evs = [e for e in events if e.type == "tool_call" and e.payload["name"] == "bash"]
    tool_result_evs = [e for e in events if e.type == "tool_result" and e.payload["name"] == "bash"]
    assert len(tool_call_evs) == 3
    assert len(tool_result_evs) == 3


@pytest.mark.asyncio
async def test_subagent_runs_isolated_and_returns_answer():
    parent_client = SimpleNamespace(
        chat=AsyncMock(side_effect=[
            _completion(tool_calls=[_tool_call("task", {
                "prompt": "find the file", "subagent_type": "general",
            })]),
            _completion(tool_calls=[_tool_call("done", {"answer": "parent done"})]),
        ]),
        aclose=AsyncMock(),
    )
    parent = Agent(_make_config(), parent_client)

    child_client = SimpleNamespace(
        chat=AsyncMock(return_value=_completion(
            tool_calls=[_tool_call("done", {"answer": "found it"})],
        )),
        aclose=AsyncMock(),
    )
    from harness import agent as agent_mod
    orig_llm_client = agent_mod.LLMClient
    agent_mod.LLMClient = lambda cfg: child_client
    try:
        events = [ev async for ev in parent.run("delegate")]
    finally:
        agent_mod.LLMClient = orig_llm_client

    task_results = [e for e in events
                    if e.type == "tool_result" and e.payload["name"] == "task"]
    assert len(task_results) == 1
    assert "found it" in task_results[0].payload["output"]


def test_token_usage_summary():
    u = TokenUsage()
    u.add({"prompt_tokens": 50, "completion_tokens": 10})
    u.add({"prompt_tokens": 30, "completion_tokens": 5,
           "prompt_tokens_details": {"cached_tokens": 20}})
    assert u.input_tokens == 80
    assert u.output_tokens == 15
    assert u.cached_input_tokens == 20
    assert u.total() == 95
    s = u.summary()
    assert "in:80" in s
    assert "out:15" in s
    assert "cached 20" in s


def test_agent_reset_clears_state():
    client = SimpleNamespace(chat=AsyncMock(), aclose=AsyncMock())
    agent = Agent(_make_config(), client)
    agent.messages.append({"role": "user", "content": "x"})
    agent.usage.add({"prompt_tokens": 100, "completion_tokens": 50})
    agent.budget.add(150)
    agent.reset()
    assert agent.messages == []
    assert agent.usage.total() == 0
    assert agent.budget.used == 0


@pytest.mark.asyncio
async def test_dedup_caches_repeated_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_WORKDIR", str(tmp_path))
    from harness import tools
    tools.WORKDIR = tmp_path.resolve()
    (tmp_path / "x.py").write_text("print('hi')")

    real_read = tools.read_file
    calls = []

    async def counting_read(path, **kwargs):
        calls.append(path)
        return await real_read(path, **kwargs)

    tools._TOOL_FUNCS["read_file"] = counting_read

    chat_n = {"n": 0}

    async def fake_chat(messages, system):
        chat_n["n"] += 1
        if chat_n["n"] >= 3:
            return _completion(tool_calls=[_tool_call("done", {"answer": "ok"})])
        return _completion(tool_calls=[_tool_call("read_file", {"path": "x.py"})])

    try:
        client = SimpleNamespace(chat=AsyncMock(side_effect=fake_chat), aclose=AsyncMock())
        agent = Agent(_make_config(), client)
        async for _ in agent.run("hi"):
            pass
        assert calls == ["x.py"], f"read_file invoked {calls}, expected only ['x.py']"
    finally:
        tools._TOOL_FUNCS["read_file"] = real_read
