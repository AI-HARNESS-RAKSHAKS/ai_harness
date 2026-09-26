"""Tool implementation tests (no LLM required)."""

from __future__ import annotations

import pytest

from harness import tools


@pytest.fixture
def workdir(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_WORKDIR", str(tmp_path))
    tools.WORKDIR = tmp_path.resolve()
    tools.configure_limits()
    return tmp_path


@pytest.fixture
def tight_limits():
    tools.configure_limits(max_chars=200, max_lines=10)
    yield
    tools.configure_limits()


@pytest.mark.asyncio
async def test_write_then_read(workdir):
    res = await tools.write_file("hello.txt", "world")
    assert "wrote" in res
    assert (workdir / "hello.txt").read_text() == "world"
    assert await tools.read_file("hello.txt") == "world"


@pytest.mark.asyncio
async def test_read_missing(workdir):
    res = await tools.read_file("missing.txt")
    assert "ERROR" in res
    assert "not found" in res


@pytest.mark.asyncio
async def test_edit_unique(workdir):
    (workdir / "f.py").write_text("a = 1\nb = 2\n")
    res = await tools.edit_file("f.py", "a = 1", "a = 10")
    assert "edited" in res
    assert (workdir / "f.py").read_text() == "a = 10\nb = 2\n"


@pytest.mark.asyncio
async def test_edit_not_unique(workdir):
    (workdir / "f.py").write_text("a = 1\na = 1\n")
    res = await tools.edit_file("f.py", "a = 1", "a = 2")
    assert "ERROR" in res
    assert "occurs" in res


@pytest.mark.asyncio
async def test_edit_replace_all(workdir):
    (workdir / "f.py").write_text("a = 1\na = 1\n")
    res = await tools.edit_file("f.py", "a = 1", "a = 2", replace_all=True)
    assert "edited" in res
    assert (workdir / "f.py").read_text() == "a = 2\na = 2\n"


@pytest.mark.asyncio
async def test_edit_missing_file(workdir):
    res = await tools.edit_file("missing.py", "x", "y")
    assert "ERROR" in res
    assert "not found" in res


@pytest.mark.asyncio
async def test_list_files(workdir):
    (workdir / "sub").mkdir()
    (workdir / "sub" / "x.txt").write_text("x")
    (workdir / "y.txt").write_text("y")
    listing = await tools.list_files(".")
    assert "y.txt" in listing
    assert "sub/" in listing
    assert "x.txt" not in listing

    sub_listing = await tools.list_files("sub")
    assert "x.txt" in sub_listing


@pytest.mark.asyncio
async def test_bash_success(workdir):
    (workdir / "marker.txt").write_text("hi")
    res = await tools.bash("cat marker.txt")
    assert "[exit 0]" in res
    assert "hi" in res


@pytest.mark.asyncio
async def test_bash_failure(workdir):
    res = await tools.bash("exit 7")
    assert "[exit 7]" in res


@pytest.mark.asyncio
async def test_bash_timeout(workdir):
    res = await tools.bash("sleep 5", timeout=1)
    assert "timed out" in res


@pytest.mark.asyncio
async def test_execute_tool_done_marker():
    res = await tools.execute_tool("done", {"answer": "all good"})
    assert res.startswith(tools.DONE_MARKER)
    assert res == tools.DONE_MARKER + "all good"


@pytest.mark.asyncio
async def test_execute_tool_unknown(workdir):
    res = await tools.execute_tool("nope", {})
    assert "ERROR" in res
    assert "unknown tool" in res


@pytest.mark.asyncio
async def test_execute_tool_bad_args(workdir):
    res = await tools.execute_tool("read_file", {})
    assert "ERROR" in res


# ---- truncation tests ----------------------------------------------------

@pytest.mark.asyncio
async def test_read_file_truncates_by_lines(workdir, tight_limits):
    big = "\n".join(f"line {i}" for i in range(500))
    (workdir / "big.txt").write_text(big)
    res = await tools.read_file("big.txt")
    assert "truncated" in res
    assert "line 0" in res
    assert "line 9" in res
    assert "line 10" not in res
    assert "500 lines total" in res


@pytest.mark.asyncio
async def test_read_file_truncates_by_chars(workdir, tight_limits):
    (workdir / "wide.txt").write_text("x" * 5000)
    res = await tools.read_file("wide.txt")
    assert "truncated" in res
    assert len(res) < 500


@pytest.mark.asyncio
async def test_bash_truncates(workdir, tight_limits):
    (workdir / "noise.txt").write_text("\n".join(f"row {i}" for i in range(500)))
    res = await tools.bash("cat noise.txt")
    assert "truncated" in res


@pytest.mark.asyncio
async def test_truncate_short_passes_through(workdir, tight_limits):
    (workdir / "small.txt").write_text("hello\nworld\n")
    res = await tools.read_file("small.txt")
    assert res == "hello\nworld\n"
    assert "truncated" not in res


@pytest.mark.asyncio
async def test_configure_limits_enforces_minimums():
    tools.configure_limits(max_chars=1, max_lines=1)
    assert tools.get_limits()[0] >= 64
    assert tools.get_limits()[1] >= 2
    tools.configure_limits()


@pytest.mark.asyncio
async def test_truncate_empty(workdir):
    res = await tools.read_file("missing.txt")
    assert res.startswith("ERROR:")


@pytest.mark.asyncio
async def test_task_tool_requires_agent():
    res = await tools.execute_tool("task", {"prompt": "do something"})
    assert "ERROR" in res
    assert "agent" in res.lower()


@pytest.mark.asyncio
async def test_task_tool_requires_prompt(workdir):
    class _StubAgent:
        async def run_subagent(self, *a, **kw):
            return "would have run"
    res = await tools.execute_tool("task", {}, agent=_StubAgent())
    assert "ERROR" in res
    assert "prompt" in res.lower()
