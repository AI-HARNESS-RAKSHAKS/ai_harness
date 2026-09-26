"""Tool implementations and their JSON schemas for the LLM.

Designed for low token usage: schemas are terse, large outputs are
truncated, and outputs over ``externalize_threshold`` chars are spilled
to disk with a short synopsis (see ``optimize.ToolOutputStore``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .optimize import ToolOutputStore, truncate_head, truncate_tail

if TYPE_CHECKING:
    from .agent import Agent


WORKDIR = Path(os.environ.get("AI_WORKDIR") or os.getcwd()).resolve()

DEFAULT_MAX_OUTPUT_CHARS = 4000
DEFAULT_MAX_OUTPUT_LINES = 200

_LIMITS: dict[str, int] = {
    "max_chars": DEFAULT_MAX_OUTPUT_CHARS,
    "max_lines": DEFAULT_MAX_OUTPUT_LINES,
}

_OUTPUT_STORE: ToolOutputStore = ToolOutputStore.default()


def configure_limits(max_chars: int | None = None, max_lines: int | None = None) -> None:
    if max_chars is not None:
        _LIMITS["max_chars"] = max(64, int(max_chars))
    if max_lines is not None:
        _LIMITS["max_lines"] = max(2, int(max_lines))


def get_limits() -> tuple[int, int]:
    return _LIMITS["max_chars"], _LIMITS["max_lines"]


def configure_output_store(store: ToolOutputStore) -> None:
    global _OUTPUT_STORE
    _OUTPUT_STORE = store


def get_output_store() -> ToolOutputStore:
    return _OUTPUT_STORE


def _safe_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = WORKDIR / p
    return p.resolve()


def _postprocess(content: str, tool_name: str, *, tail: bool = False) -> str:
    """Apply size cap + externalisation in one place."""
    max_chars, max_lines = get_limits()
    if tail:
        truncated = truncate_tail(content, max_chars, max_lines)
    else:
        truncated = truncate_head(content, max_chars, max_lines)
    if len(truncated) > _OUTPUT_STORE.threshold:
        externalized, _path = _OUTPUT_STORE.maybe_externalize(truncated, tool_name)
        if externalized != truncated:
            return externalized
    return truncated


async def read_file(path: str) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if p.is_dir():
        return f"ERROR: {path} is a directory"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"ERROR: failed to read {path}: {exc}"
    return _postprocess(text, "read_file", tail=False)


async def write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"ERROR: failed to write {path}: {exc}"


async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"ERROR: failed to read {path}: {exc}"

    if replace_all:
        if old_string not in text:
            return f"ERROR: old_string not found in {path}"
        new_text = text.replace(old_string, new_string)
        count = text.count(old_string)
    else:
        occurrences = text.count(old_string)
        if occurrences == 0:
            return f"ERROR: old_string not found in {path}"
        if occurrences > 1:
            return (
                f"ERROR: old_string occurs {occurrences} times in {path}; "
                "provide more context or set replace_all=true"
            )
        new_text = text.replace(old_string, new_string, 1)
        count = 1

    try:
        p.write_text(new_text, encoding="utf-8")
    except Exception as exc:
        return f"ERROR: failed to write {path}: {exc}"
    return f"edited {path} ({count} replacement{'s' if count != 1 else ''})"


async def list_files(path: str = ".") -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: path not found: {path}"
    if not p.is_dir():
        return f"ERROR: {path} is not a directory"

    entries: list[str] = []
    for entry in sorted(p.iterdir()):
        if entry.name.startswith(".") and entry.name not in (".",):
            continue
        if entry.is_dir():
            entries.append(f"d {entry.name}/")
        elif entry.is_file():
            entries.append(f"f {entry.name}")
    return _postprocess("\n".join(entries) if entries else "(empty directory)", "list_files", tail=False)


async def bash(command: str, timeout: int = 30) -> str:
    """Run a shell command. Output is tail-truncated and externalised if huge."""
    timeout = max(1, min(int(timeout), 120))
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout}s: {command}"
    except Exception as exc:
        return f"ERROR: failed to execute command: {exc}"

    out = proc.stdout or ""
    err = proc.stderr or ""
    rc = proc.returncode
    parts = [f"$ {command}", f"[exit {rc}]"]
    if out:
        parts.append(out.rstrip("\n"))
    if err:
        parts.append("[stderr]\n" + err.rstrip("\n"))
    if not out and not err:
        parts.append("(no output)")
    return _postprocess("\n".join(parts), "bash", tail=True)


DONE_MARKER = "__DONE__:"

TASK_TOOL_NAME = "task"


async def execute_tool(name: str, arguments: dict, agent: "Agent | None" = None) -> str:
    """Dispatch a tool call. ``agent`` is required for the ``task`` sub-agent tool."""
    if name == "done":
        return DONE_MARKER + str(arguments.get("answer", ""))

    if name == TASK_TOOL_NAME:
        if agent is None:
            return "ERROR: task tool requires an active agent context"
        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return "ERROR: task requires a non-empty prompt"
        subagent_type = str(arguments.get("subagent_type", "general"))
        max_turns = int(arguments.get("max_turns", 0)) or None
        return await agent.run_subagent(prompt, subagent_type=subagent_type, max_turns=max_turns)

    func = _TOOL_FUNCS.get(name)
    if func is None:
        return f"ERROR: unknown tool: {name}"
    try:
        return await func(**arguments)
    except TypeError as exc:
        return f"ERROR: invalid arguments for {name}: {exc}"
    except Exception as exc:
        return f"ERROR: {name} failed: {exc}"


_TOOL_FUNCS = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_files": list_files,
    "bash": bash,
}

# Terse schemas. The system prompt and these schemas form the static prefix
# that providers cache automatically.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace old_string with new_string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files/dirs at a path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command, return stdout/stderr/exit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "default": 30},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": (
                "Spawn an isolated sub-agent for a subtask. The sub-agent gets "
                "only the prompt you provide (no parent history), bounded turns, "
                "and an optionally cheaper model. Returns the sub-agent's final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "subagent_type": {"type": "string", "default": "general"},
                    "max_turns": {"type": "integer"},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Finish and return the final answer.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
]
