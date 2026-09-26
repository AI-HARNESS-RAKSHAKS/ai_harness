"""Tests for the optimization layer."""

from __future__ import annotations

import asyncio
import time

import pytest

from harness.optimize import (
    CallDedup,
    LoopDetector,
    TokenBudget,
    ToolOutputStore,
    elide_superseded_reads,
    elide_superseded_writes,
    gather_tool_results,
    optimize_messages_for_llm,
    strip_intermediate_assistant_content,
    tombstone_old_tool_results,
    truncate_head,
    truncate_tail,
)


# ---------- TokenBudget ----------

def test_budget_starts_empty():
    b = TokenBudget(limit=1000)
    assert b.used == 0
    assert not b.exceeded
    assert b.remaining == 1000


def test_budget_add_and_exceed():
    b = TokenBudget(limit=100)
    b.add(60)
    assert not b.exceeded
    b.add(50)
    assert b.exceeded
    assert b.remaining == 0


def test_budget_fraction():
    b = TokenBudget(limit=100)
    b.add(25)
    assert b.fraction == 0.25


def test_budget_reset():
    b = TokenBudget(limit=100)
    b.add(50)
    b.reset()
    assert b.used == 0


# ---------- LoopDetector ----------

def test_loop_detector_passes_distinct_calls():
    ld = LoopDetector(window=5, identical_threshold=3, tool_freq_threshold=100)
    for i in range(4):
        stop, reason = ld.check("read_file", {"path": f"f{i}.py"})
        assert stop is False, f"unexpected stop: {reason}"


def test_loop_detector_identical_threshold():
    ld = LoopDetector(window=10, identical_threshold=3, tool_freq_threshold=100)
    ld.check("bash", {"command": "ls"})
    assert ld.check("bash", {"command": "ls"})[0] is False
    stop, reason = ld.check("bash", {"command": "ls"})
    assert stop is True
    assert "repeated" in reason


def test_loop_detector_frequency_threshold():
    ld = LoopDetector(window=100, identical_threshold=100, tool_freq_threshold=5)
    for i in range(4):
        ld.check("bash", {"command": f"echo {i}"})
    stop, reason = ld.check("bash", {"command": "echo last"})
    assert stop is True
    assert "called" in reason


def test_loop_detector_window_eviction():
    ld = LoopDetector(window=3, identical_threshold=3, tool_freq_threshold=100)
    for _ in range(3):
        ld.check("x", {"a": 1})
    stop, _ = ld.check("x", {"a": 1})
    assert stop


# ---------- ToolOutputStore ----------

def test_externalize_under_threshold():
    s = ToolOutputStore.default(threshold=1000)
    short = "hello world"
    out, path = s.maybe_externalize(short, "read_file")
    assert out == short
    assert path is None


def test_externalize_over_threshold(tmp_path):
    s = ToolOutputStore(base_dir=tmp_path, threshold=100)
    big = "x" * 500
    out, path = s.maybe_externalize(big, "read_file")
    assert path is not None
    assert tmp_path.joinpath(path.split("/")[-1]).exists()
    assert "externalized" in out
    assert "500 chars" in out


def test_externalize_creates_dir(tmp_path):
    target = tmp_path / "subdir" / "outputs"
    s = ToolOutputStore(base_dir=target, threshold=10)
    s.maybe_externalize("y" * 100, "bash")
    assert target.is_dir()


# ---------- truncate_head / truncate_tail ----------

def test_truncate_head_short_passthrough():
    assert truncate_head("hello\nworld\n", 1000, 100) == "hello\nworld\n"


def test_truncate_head_truncates_lines():
    big = "\n".join(f"line {i}" for i in range(500))
    out = truncate_head(big, 100_000, 10)
    assert "line 0" in out
    assert "line 9" in out
    assert "line 10" not in out
    assert "500 lines total" in out


def test_truncate_tail_short_passthrough():
    assert truncate_tail("hello\nworld\n", 1000, 100) == "hello\nworld\n"


def test_truncate_tail_keeps_errors_at_end():
    big = "\n".join(f"ok {i}" for i in range(100)) + "\nERROR: boom"
    out = truncate_tail(big, 100_000, 10)
    assert "ERROR: boom" in out
    assert "ok 0" not in out


def test_truncate_tail_truncates_chars():
    big = "x" * 5000
    out = truncate_tail(big, 200, 1000)
    assert len(out) < 500


# ---------- elide_superseded_writes ----------

def test_elide_marks_superseded_writes():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "write_file",
             "arguments": '{"path": "a.py", "content": "' + ("x" * 500) + '"}'}}
        ]},
        {"role": "tool", "tool_call_id": "1", "name": "write_file", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "2", "type": "function", "function": {"name": "write_file",
             "arguments": '{"path": "a.py", "content": "short"}'}}
        ]},
        {"role": "tool", "tool_call_id": "2", "name": "write_file", "content": "ok"},
    ]
    rewritten = elide_superseded_writes(messages)
    first_args = rewritten[1]["tool_calls"][0]["function"]["arguments"]
    assert "elided" in first_args
    second_args = rewritten[3]["tool_calls"][0]["function"]["arguments"]
    assert '"short"' in second_args


def test_elide_preserves_non_superseded():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "write_file",
             "arguments": '{"path": "a.py", "content": "' + ("x" * 500) + '"}'}}
        ]},
        {"role": "tool", "tool_call_id": "1", "name": "write_file", "content": "ok"},
    ]
    rewritten = elide_superseded_writes(messages)
    args = rewritten[1]["tool_calls"][0]["function"]["arguments"]
    assert "elided" not in args
    assert "x" * 100 in args


def test_elide_no_writes():
    messages = [{"role": "user", "content": "hi"}]
    assert elide_superseded_writes(messages) is messages


# ---------- CallDedup ----------

def test_dedup_hit_and_miss():
    d = CallDedup(ttl_seconds=60)
    assert d.get("read_file", {"path": "x.py"}) is None
    d.put("read_file", {"path": "x.py"}, "content here")
    assert d.get("read_file", {"path": "x.py"}) == "content here"
    assert d.get("read_file", {"path": "y.py"}) is None


def test_dedup_ttl_expiry():
    d = CallDedup(ttl_seconds=0.05)
    d.put("read_file", {"path": "x.py"}, "content")
    assert d.get("read_file", {"path": "x.py"}) == "content"
    time.sleep(0.1)
    assert d.get("read_file", {"path": "x.py"}) is None


def test_dedup_lru_eviction():
    d = CallDedup(ttl_seconds=60, max_entries=2)
    d.put("read_file", {"path": "a"}, "A")
    d.put("read_file", {"path": "b"}, "B")
    d.put("read_file", {"path": "c"}, "C")
    assert d.get("read_file", {"path": "a"}) is None
    assert d.get("read_file", {"path": "b"}) == "B"
    assert d.get("read_file", {"path": "c"}) == "C"


# ---------- gather_tool_results ----------

@pytest.mark.asyncio
async def test_gather_returns_in_order():
    async def make(x, delay):
        await asyncio.sleep(delay)
        return x

    results = await gather_tool_results([
        make("a", 0.02),
        make("b", 0.01),
        make("c", 0.005),
    ])
    assert results == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_gather_converts_exceptions():
    async def ok():
        return "good"
    async def bad():
        raise ValueError("nope")

    results = await gather_tool_results([ok(), bad()])
    assert results[0] == "good"
    assert "ValueError" in results[1]
    assert "nope" in results[1]


# ---------- elide_superseded_reads ----------

def test_elide_superseded_reads_marks_stale_read():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read_file",
             "arguments": '{"path": "x.py"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file",
         "content": "line 1\n" + ("x" * 500)},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_2", "type": "function", "function": {"name": "edit_file",
             "arguments": '{"path": "x.py", "old_string": "line 1", "new_string": "line 1 modified"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_2", "name": "edit_file", "content": "edited x.py"},
    ]
    rewritten = elide_superseded_reads(messages)
    first_tool_res = rewritten[2]["content"]
    assert "stale read elided" in first_tool_res
    assert "x.py" in first_tool_res
    assert "chars saved" in first_tool_res
    # Unchanged edit_file tool result
    assert rewritten[4]["content"] == "edited x.py"


def test_elide_superseded_reads_preserves_latest_read():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read_file",
             "arguments": '{"path": "x.py"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file",
         "content": "line 1\n" + ("x" * 500)},
    ]
    rewritten = elide_superseded_reads(messages)
    assert rewritten[2]["content"] == messages[2]["content"]


# ---------- tombstone_old_tool_results ----------

def test_tombstone_old_tool_results_prunes_older_outputs():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "bash"}}]},
        {"role": "tool", "tool_call_id": "1", "name": "bash", "content": "A" * 600},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "function": {"name": "bash"}}]},
        {"role": "tool", "tool_call_id": "2", "name": "bash", "content": "B" * 600},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "3", "function": {"name": "bash"}}]},
        {"role": "tool", "tool_call_id": "3", "name": "bash", "content": "C" * 600},
    ]
    # keep_recent_tools=2 means tool 1 gets tombstoned, tool 2 and 3 stay
    rewritten = tombstone_old_tool_results(messages, keep_recent_tools=2, threshold_chars=100)
    assert "elided" in rewritten[2]["content"]
    assert "bash" in rewritten[2]["content"]
    assert rewritten[4]["content"] == "B" * 600
    assert rewritten[6]["content"] == "C" * 600


# ---------- strip_intermediate_assistant_content ----------

def test_strip_intermediate_assistant_content():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "I am thinking out loud about how to fix this bug for 100 characters...",
         "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "res"},
        {"role": "assistant", "content": "Now I will do the next thing...",
         "tool_calls": [{"id": "2"}]},
    ]
    # keep_recent_turns=1: older assistant thought is stripped to ""
    rewritten = strip_intermediate_assistant_content(messages, keep_recent_turns=1)
    assert rewritten[1]["content"] == ""
    assert rewritten[3]["content"] == "Now I will do the next thing..."


# ---------- optimize_messages_for_llm pipeline ----------

def test_optimize_messages_pipeline_applies_all():
    messages = [
        {"role": "user", "content": "task"},
        # Step 1: read file (big)
        {"role": "assistant", "content": "Let me read the file first to check it out...",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "f.js"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "const foo = 1;\n" + ("x" * 600)},
        # Step 2: write file (big)
        {"role": "assistant", "content": "Let me write the initial file now...",
         "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "write_file", "arguments": '{"path": "f.js", "content": "' + ("z" * 600) + '"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "write_file", "content": "wrote bytes"},
        # Step 3: edit file (supersedes step 1 read and step 2 write)
        {"role": "assistant", "content": "Now editing it...",
         "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "edit_file", "arguments": '{"path": "f.js", "old_string": "a", "new_string": "b"}'}}]},
        {"role": "tool", "tool_call_id": "c3", "name": "edit_file", "content": "edited f.js"},
    ]
    optimized = optimize_messages_for_llm(messages, keep_recent_tools=1)
    # Step 1 assistant chatter stripped
    assert optimized[1]["content"] == ""
    # Step 1 read was superseded by write/edit
    assert "stale read elided" in optimized[2]["content"]
    # Step 2 write was superseded by edit
    assert "elided" in optimized[3]["tool_calls"][0]["function"]["arguments"]
