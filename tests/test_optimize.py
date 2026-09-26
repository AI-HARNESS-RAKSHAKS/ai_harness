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
    elide_superseded_writes,
    gather_tool_results,
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
