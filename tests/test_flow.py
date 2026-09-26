"""Tests for the FlowPanel renderer (pure function, no Textual needed)."""

from __future__ import annotations

from harness.flow import FlowState, _render_flow


def _state(**kwargs) -> FlowState:
    s = FlowState(model="gemini-3.8-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai", budget_limit=200000, max_steps=20)
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def test_renders_user_input():
    s = _state(user_input="list files in this directory")
    out = _render_flow(s)
    assert "USER" in out
    assert "list files in this directory" in out
    assert "chars in" in out
    assert "DATA FLOW PIPELINE" in out


def test_renders_step_counter():
    s = _state(step=3, max_steps=20)
    out = _render_flow(s)
    assert "step 3/20" in out


def test_renders_tokens_in_and_out():
    s = _state(step_in=1200, step_out=85)
    out = _render_flow(s)
    assert "in: 1,200 tok" in out
    assert "out: 85 tok" in out


def test_renders_pending_tools():
    s = _state(pending_tools=["read_file", "list_files"])
    out = _render_flow(s)
    assert "TOOLS" in out
    assert "read_file" in out
    assert "list_files" in out


def test_renders_finished_tools():
    s = _state(finished_tools=["bash"])
    out = _render_flow(s)
    assert "bash" in out


def test_renders_final_answer():
    s = _state(final_answer="Done — 3 files found.")
    out = _render_flow(s)
    assert "USER" in out
    assert "Done" in out


def test_renders_error():
    s = _state(error="rate limited")
    out = _render_flow(s)
    assert "rate limited" in out
    assert "✗" in out


def test_renders_budget_bar():
    s = _state(budget_used=50000, budget_limit=200000)
    out = _render_flow(s)
    assert "50,000" in out
    assert "200,000" in out
    assert "25%" in out
    assert "budget:" in out


def test_renders_budget_bar_color_coded_red_when_high():
    s = _state(budget_used=180000, budget_limit=200000)
    out = _render_flow(s)
    assert "90%" in out
    assert "red" in out


def test_renders_externalized_count():
    s = _state(externalized=3)
    out = _render_flow(s)
    assert "ext:3" in out


def test_renders_streaming_text():
    s = _state(streaming_text="reading the file now")
    out = _render_flow(s)
    assert "streaming" in out
    assert "reading the file now" in out


def test_long_input_truncated():
    s = _state(user_input="x" * 200)
    out = _render_flow(s)
    # Should be truncated with ellipsis
    assert "…" in out


def test_no_user_input_still_renders():
    s = _state()
    out = _render_flow(s)
    assert "DATA FLOW PIPELINE" in out
    assert "USER" in out or "step 0" in out  # either header or step


def test_reset_clears_state():
    s = _state(user_input="old", step=5, total_in=1000, externalized=2)
    s.reset()
    assert s.user_input == ""
    assert s.step == 0
    assert s.total_in == 0
    assert s.externalized == 0
