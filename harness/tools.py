"""Tool implementations and their JSON schemas for the LLM.

Designed for low token usage: schemas are terse, large outputs are
truncated, and outputs over ``externalize_threshold`` chars are spilled
to disk with a short synopsis (see ``optimize.ToolOutputStore``).
"""

from __future__ import annotations

import os
import re
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
    _LIMITS["max_chars"] = max(64, int(max_chars)) if max_chars is not None else DEFAULT_MAX_OUTPUT_CHARS
    _LIMITS["max_lines"] = max(2, int(max_lines)) if max_lines is not None else DEFAULT_MAX_OUTPUT_LINES


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


def extract_outline(text: str, path: str) -> str:
    """Extract code skeleton (classes, functions, methods, top-level constants) with line numbers."""
    lines = text.splitlines()
    total = len(lines)
    matches = []
    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#", "/*", "*")):
            continue
        is_symbol = False
        if stripped.startswith(("class ", "def ", "async def ", "function ", "async function ")):
            is_symbol = True
        elif any(stripped.startswith(k) for k in ("const ", "let ", "var ")) and ("=" in stripped):
            is_symbol = True
        elif re.match(r"^[a-zA-Z_$][\w$]*\s*\([^)]*\)\s*\{", stripped):
            is_symbol = True

        if is_symbol:
            snippet = stripped[:75]
            matches.append(f"{idx:4d} | {snippet}")

    if not matches:
        return f"[no outline symbols found in {path} ({total} lines total)]"

    header = f"[outline of {path}: {len(matches)} symbols across {total} lines]\n"
    return header + "\n".join(matches)


async def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    outline: bool = False,
) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if p.is_dir():
        return f"ERROR: {path} is a directory"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"ERROR: failed to read {path}: {exc}"

    if outline:
        return _postprocess(extract_outline(text, path), "read_file", tail=False)

    if start_line is not None or end_line is not None:
        try:
            s = max(1, int(start_line)) if start_line is not None else 1
        except (ValueError, TypeError):
            s = 1
        try:
            e = int(end_line) if end_line is not None else None
        except (ValueError, TypeError):
            e = None

        lines = text.splitlines()
        total = len(lines)
        if s > total:
            return f"(empty: start_line {s} exceeds {total} lines in {path})"
        actual_e = min(total, e) if e is not None else total
        selected = lines[s - 1 : actual_e]
        numbered = [f"{s + idx:4d} | {line}" for idx, line in enumerate(selected)]
        header = f"[lines {s}-{s + len(selected) - 1} of {total} in {path}]\n"
        return _postprocess(header + "\n".join(numbered), "read_file", tail=False)

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


def prune_stack_trace(stderr: str) -> str:
    """Strips internal runtime frames (node:internal, site-packages) from crash dumps."""
    if not stderr:
        return stderr
    lines = stderr.splitlines()
    cleaned = []
    omitted = 0
    for line in lines:
        if "node:internal/" in line or "(internal/" in line:
            omitted += 1
            continue
        cleaned.append(line)
    if omitted > 0:
        cleaned.append(f"  ... [{omitted} internal runtime frames omitted]")
    return "\n".join(cleaned)


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
        cleaned_err = prune_stack_trace(err.rstrip("\n"))
        parts.append("[stderr]\n" + cleaned_err)
    if not out and not err:
        parts.append("(no output)")
    return _postprocess("\n".join(parts), "bash", tail=True)


async def search_code(query: str, path: str = ".", max_matches: int = 20) -> str:
    """Fast regex or literal code search. Returns file:line: match snippets without reading whole files."""
    if not query:
        return "ERROR: search query cannot be empty"
    target = _safe_path(path)
    if not target.exists():
        return f"ERROR: path not found: {path}"

    try:
        regex = re.compile(query, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(query), re.IGNORECASE)

    matches: list[str] = []
    ignore_dirs = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".harness_outputs"}

    def _search_file(f_path: Path) -> bool:
        try:
            rel = f_path.relative_to(WORKDIR)
        except ValueError:
            rel = f_path
        try:
            text = f_path.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{rel}:{line_no}: {line.strip()[:80]}")
                    if len(matches) >= max_matches:
                        return True
        except Exception:
            pass
        return False

    if target.is_file():
        _search_file(target)
    else:
        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
            for file in sorted(files):
                if file.startswith("."):
                    continue
                if _search_file(Path(root) / file):
                    break
            if len(matches) >= max_matches:
                break

    if not matches:
        return f"no matches found for {query!r} in {path}"

    header = f"[found {len(matches)} match{'es' if len(matches) != 1 else ''} for {query!r}]\n"
    return _postprocess(header + "\n".join(matches), "search_code", tail=False)


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
    "search_code": search_code,
}

# Terse schemas. The system prompt and these schemas form the static prefix
# that providers cache automatically.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents, a line slice, or symbol outline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "outline": {"type": "boolean"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Fast regex search across files. Returns file:line matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query"],
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
            "description": "Spawn an isolated sub-agent for a subtask. Returns final answer.",
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
