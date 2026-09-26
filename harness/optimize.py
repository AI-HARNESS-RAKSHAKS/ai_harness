"""Runtime optimisation layer.

Implements techniques adapted from DeerFlow and Pi to cut token usage
and tool latency:

- ``TokenBudget`` - per-run cap that forces the agent to stop calling tools.
- ``LoopDetector`` - two-layer detection (identical-call hash + per-tool
  frequency) like DeerFlow's ``loop_detection_middleware``.
- ``ToolOutputStore`` - externalises oversized tool outputs to disk with a
  short synopsis, mirroring DeerFlow's ``tool_output_budget_middleware``.
- ``truncate_head`` / ``truncate_tail`` - context-aware truncation (Pi keeps
  the head of ``read`` and the tail of ``bash``).
- ``elide_superseded_writes`` - replaces ``write_file`` payloads that have
  been followed by a later read/edit/write of the same path (DeerFlow's
  ``elide_superseded_writes``).
- ``call_dedup`` - short-window memoisation of identical tool calls (Pi
  does not have this; cheap to add).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable


_EXTERNALIZE_DIR = Path(os.environ.get("AI_EXTERNALIZE_DIR", ".harness_outputs"))


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

@dataclass
class TokenBudget:
    """Per-run cumulative token counter with a hard cap."""

    limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exceeded(self) -> bool:
        return self.used >= self.limit

    @property
    def fraction(self) -> float:
        return self.used / self.limit if self.limit else 1.0

    def add(self, n: int) -> None:
        self.used += max(0, n)

    def reset(self) -> None:
        self.used = 0


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

@dataclass
class LoopDetector:
    """Detects runaway loops using two signals:

    1. Identical ``(name, args)`` calls within a sliding window.
    2. Total frequency of a single tool across the run.

    When triggered, the caller should stop the agent loop and surface the
    reason to the user.
    """

    window: int = 20
    identical_threshold: int = 5
    tool_freq_threshold: int = 50

    _recent: deque = field(default_factory=deque)
    _tool_freq: dict = field(default_factory=dict)

    def _signature(self, name: str, args: dict) -> str:
        try:
            s = json.dumps(args, sort_keys=True, default=str)
        except TypeError:
            s = repr(args)
        return f"{name}:{hashlib.sha1(s.encode()).hexdigest()[:10]}"

    def check(self, name: str, args: dict) -> tuple[bool, str]:
        sig = self._signature(name, args)
        self._recent.append(sig)
        if len(self._recent) > self.window:
            self._recent.popleft()

        identical = sum(1 for s in self._recent if s == sig)
        if identical >= self.identical_threshold:
            return True, f"identical call repeated {identical}x in window"

        self._tool_freq[name] = self._tool_freq.get(name, 0) + 1
        if self._tool_freq[name] >= self.tool_freq_threshold:
            return True, f"tool '{name}' called {self._tool_freq[name]}x total"

        return False, ""

    def reset(self) -> None:
        self._recent.clear()
        self._tool_freq.clear()


# ---------------------------------------------------------------------------
# Tool output externalisation
# ---------------------------------------------------------------------------

@dataclass
class ToolOutputStore:
    """Persists large tool outputs to disk and returns a short synopsis.

    Anything over ``threshold`` chars is written to ``base_dir/<name>-<hash>.txt``
    and the model sees only a header plus a preview. The model can fetch the
    full content later via ``read_file`` if it really needs it.
    """

    base_dir: Path
    threshold: int

    @classmethod
    def default(cls, threshold: int = 8000) -> "ToolOutputStore":
        return cls(base_dir=_EXTERNALIZE_DIR, threshold=threshold)

    def maybe_externalize(self, content: str, tool_name: str) -> tuple[str, str | None]:
        if not content or len(content) <= self.threshold:
            return content, None
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha1(content.encode(errors="replace")).hexdigest()[:12]
            path = self.base_dir / f"{tool_name}-{digest}.txt"
            path.write_text(content, encoding="utf-8")
        except Exception:
            return content[: self.threshold] + f"\n... [truncated, {len(content)} chars]", None

        chars = len(content)
        lines = content.count("\n") + 1
        head = content[:200].replace("\n", " ")
        synopsis = (
            f"[externalized: {chars} chars / {lines} lines -> {path}]\n"
            f"head: {head!r}\n"
            f"(use read_file to fetch full content if needed)"
        )
        return synopsis, str(path)


# ---------------------------------------------------------------------------
# Truncation strategies
# ---------------------------------------------------------------------------

def truncate_head(text: str, max_chars: int, max_lines: int) -> str:
    """Keep the FIRST N lines/chars. Use for ``read_file`` / ``list_files``.

    Short outputs pass through byte-for-byte (including trailing newlines).
    """
    if not text:
        return text
    total_lines = text.count("\n") + 1
    if len(text) <= max_chars and total_lines <= max_lines:
        return text

    truncated_lines = False
    truncated_chars = False
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated_lines = True
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars]
        truncated_chars = True

    if truncated_lines or truncated_chars:
        out += f"\n... [truncated, {len(text)} chars / {total_lines} lines total]"
    return out


def truncate_tail(text: str, max_chars: int, max_lines: int) -> str:
    """Keep the LAST N lines/chars. Use for ``bash`` (errors at end).

    Short outputs pass through unchanged.
    """
    if not text:
        return text
    total_lines = text.count("\n") + 1
    if len(text) <= max_chars and total_lines <= max_lines:
        return text

    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = "..." + out[-max_chars:]
        else:
            out = f"... [truncated, earlier {total_lines - max_lines} lines elided]\n" + out
    elif len(text) > max_chars:
        out = "..." + text[-max_chars:]
    else:
        out = text

    return out + f"\n[total: {len(text)} chars / {total_lines} lines]"


# ---------------------------------------------------------------------------
# Superseded write elision
# ---------------------------------------------------------------------------

def elide_superseded_writes(messages: list[dict]) -> list[dict]:
    """Replace ``write_file`` payloads that have been superseded by a later
    tool call referencing the same path with a tiny placeholder.

    Mirrors DeerFlow's ``elide_superseded_writes`` - only mutates the
    model-bound copy; the in-memory state stays intact so re-execution
    (e.g. on a retry) still has the original payload.
    """
    writes: list[tuple[int, int, str]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for j, tc in enumerate(msg["tool_calls"]):
            fn = tc.get("function") or {}
            if fn.get("name") != "write_file":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
                writes.append((i, j, args.get("path", "")))
            except json.JSONDecodeError:
                pass

    if not writes:
        return messages

    superseded: set[tuple[int, int]] = set()
    for idx, (i, j, path) in enumerate(writes):
        if not path:
            continue
        is_superseded = False
        for later_msg in messages[i + 1:]:
            if later_msg.get("role") != "assistant" or not later_msg.get("tool_calls"):
                continue
            for later_tc in later_msg["tool_calls"]:
                l_fn = later_tc.get("function") or {}
                if l_fn.get("name") in ("write_file", "edit_file"):
                    try:
                        l_args = json.loads(l_fn.get("arguments") or "{}")
                        if l_args.get("path") == path:
                            is_superseded = True
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass
            if is_superseded:
                break
        if is_superseded:
            superseded.add((i, j))

    if not superseded:
        return messages

    new_messages: list[dict] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            new_messages.append(msg)
            continue
        new_tcs = []
        for j, tc in enumerate(msg["tool_calls"]):
            if (i, j) in superseded:
                fn = tc.get("function") or {}
                if fn.get("name") == "write_file":
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    content = args.get("content", "")
                    if isinstance(content, str) and len(content) > 100:
                        args["content"] = (
                            f"[elided: {len(content)} chars; "
                            "superseded by a later write/edit/read of the same path]"
                        )
                        new_tc = {**tc, "function": {**fn, "arguments": json.dumps(args)}}
                        new_tcs.append(new_tc)
                        continue
            new_tcs.append(tc)
        new_messages.append({**msg, "tool_calls": new_tcs})

    return new_messages


# ---------------------------------------------------------------------------
# Superseded read elision
# ---------------------------------------------------------------------------

def elide_superseded_reads(messages: list[dict]) -> list[dict]:
    """Replace read_file tool outputs that have been superseded by a later
    read or edit/write of the same path with a compact placeholder.

    Stale reads waste hundreds/thousands of tokens and confuse the model
    with outdated code.
    """
    read_calls: dict[str, tuple[int, str]] = {}
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            fn = tc.get("function") or {}
            if fn.get("name") != "read_file":
                continue
            cid = tc.get("id")
            if not cid:
                continue
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                path = args.get("path")
                if path:
                    read_calls[cid] = (i, str(path))
            except (json.JSONDecodeError, TypeError):
                pass

    if not read_calls:
        return messages

    superseded_ids: set[str] = set()
    for cid, (call_idx, path) in read_calls.items():
        is_superseded = False
        for later_msg in messages[call_idx + 1:]:
            if later_msg.get("role") != "assistant" or not later_msg.get("tool_calls"):
                continue
            for later_tc in later_msg["tool_calls"]:
                l_fn = later_tc.get("function") or {}
                if l_fn.get("name") in ("read_file", "edit_file", "write_file"):
                    l_args_raw = l_fn.get("arguments") or "{}"
                    try:
                        l_args = json.loads(l_args_raw) if isinstance(l_args_raw, str) else l_args_raw
                        if l_args.get("path") == path:
                            is_superseded = True
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass
            if is_superseded:
                break
        if is_superseded:
            superseded_ids.add(cid)

    if not superseded_ids:
        return messages

    new_messages: list[dict] = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") in superseded_ids:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 100:
                cid = msg.get("tool_call_id", "")
                path = read_calls.get(cid, (0, "file"))[1]
                new_msg = {
                    **msg,
                    "content": f"[stale read elided: '{path}' was modified or re-read later ({len(content)} chars saved)]",
                }
                new_messages.append(new_msg)
                continue
        new_messages.append(msg)

    return new_messages


# ---------------------------------------------------------------------------
# Old tool result tombstoning (observation pruning)
# ---------------------------------------------------------------------------

def tombstone_old_tool_results(
    messages: list[dict],
    keep_recent_tools: int = 2,
    threshold_chars: int = 250,
) -> list[dict]:
    """Compacts tool outputs older than `keep_recent_tools` tool executions.

    The model has already observed older tool outputs. Keeping full 4000-char
    bash outputs or file listings from early steps causes prompt tokens to grow
    quadratically with every step.
    Older tool outputs exceeding `threshold_chars` are replaced with a concise
    tombstone summary.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent_tools:
        return messages

    old_indices = set(tool_indices[:-keep_recent_tools])

    new_messages: list[dict] = []
    for i, msg in enumerate(messages):
        if i in old_indices:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > threshold_chars:
                tool_name = msg.get("name", "tool")
                lines = content.count("\n") + 1
                first_line = content.splitlines()[0][:60] if content else ""
                new_messages.append({
                    **msg,
                    "content": (
                        f"[earlier {tool_name} output elided ({len(content)} chars / {lines} lines); "
                        f"head: {first_line!r}]"
                    ),
                })
                continue
        new_messages.append(msg)

    return new_messages


# ---------------------------------------------------------------------------
# Intermediate thought commentary stripping
# ---------------------------------------------------------------------------

def strip_intermediate_assistant_content(
    messages: list[dict],
    keep_recent_turns: int = 1,
) -> list[dict]:
    """Strip verbose intermediate thoughts/commentary from past assistant messages
    that contain tool calls.

    When an assistant outputs both text and tool_calls, the text was ephemeral
    scratchpad thinking. Keeping 10 steps of verbose self-talk wastes tokens.
    """
    assistant_indices = [
        i for i, m in enumerate(messages)
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    if len(assistant_indices) <= keep_recent_turns:
        return messages

    old_assistant_indices = set(assistant_indices[:-keep_recent_turns])
    new_messages: list[dict] = []
    for i, msg in enumerate(messages):
        if i in old_assistant_indices:
            content = msg.get("content")
            if content and isinstance(content, str):
                new_messages.append({**msg, "content": ""})
                continue
        new_messages.append(msg)
    return new_messages


# ---------------------------------------------------------------------------
# Pipeline message optimizer
# ---------------------------------------------------------------------------

def optimize_messages_for_llm(
    messages: list[dict],
    keep_recent_tools: int = 2,
    threshold_chars: int = 250,
) -> list[dict]:
    """Pipeline that applies all context-slimming optimizations before sending
    to the LLM API:
    1. elide_superseded_writes (DeerFlow-style write payload elision)
    2. elide_superseded_reads (evicts stale file read contents)
    3. tombstone_old_tool_results (prunes older observations, cutting O(N^2) growth)
    4. strip_intermediate_assistant_content (drops ephemeral scratchpad thoughts)
    """
    msgs = elide_superseded_writes(messages)
    msgs = elide_superseded_reads(msgs)
    msgs = tombstone_old_tool_results(msgs, keep_recent_tools=keep_recent_tools, threshold_chars=threshold_chars)
    msgs = strip_intermediate_assistant_content(msgs)
    return msgs


# ---------------------------------------------------------------------------
# Short-window tool result dedup
# ---------------------------------------------------------------------------

@dataclass
class CallDedup:
    """Memoise identical tool calls within a short window.

    Avoids the model paying the cost of, e.g., reading the same file
    multiple times across a long-running investigation.
    """

    ttl_seconds: float = 120.0
    max_entries: int = 256

    _cache: OrderedDict = field(default_factory=OrderedDict)

    def _key(self, name: str, args: dict) -> str:
        try:
            s = json.dumps(args, sort_keys=True, default=str)
        except TypeError:
            s = repr(args)
        return f"{name}:{hashlib.sha1(s.encode()).hexdigest()}"

    def get(self, name: str, args: dict) -> str | None:
        k = self._key(name, args)
        entry = self._cache.get(k)
        if entry is None:
            return None
        result, ts = entry
        if time.monotonic() - ts > self.ttl_seconds:
            self._cache.pop(k, None)
            return None
        self._cache.move_to_end(k)
        return result

    def put(self, name: str, args: dict, result: str) -> None:
        k = self._key(name, args)
        self._cache[k] = (result, time.monotonic())
        self._cache.move_to_end(k)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Parallel tool execution helper
# ---------------------------------------------------------------------------

async def gather_tool_results(
    coros: list[Awaitable[Any]],
) -> list[Any]:
    """Run coroutines in parallel; convert exceptions into error strings.

    Like Pi's ``executeToolCallsParallel`` but tolerant of failures.
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    out: list[Any] = []
    for r in results:
        if isinstance(r, BaseException):
            out.append(f"ERROR: {type(r).__name__}: {r}")
        else:
            out.append(r)
    return out
