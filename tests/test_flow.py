"""Tests for the extended FlowPanel renderer (pure function)."""

from __future__ import annotations

from harness.flow import FlowState, _render_flow


def _state(**kwargs) -> FlowState:
    s = FlowState(
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        budget_limit=200000,
        max_steps=20,
        context_window=1_048_576,
    )
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


# --- existing tests still work ---

def test_renders_user_input():
    s = _state(user_input="list files in this directory")
    out = _render_flow(s)
    assert "USER" in out
    assert "list files in this directory" in out
    assert "chars in" in out
    assert "DATA FLOW" in out


def test_renders_step_counter():
    s = _state(step=3, max_steps=20)
    out = _render_flow(s)
    assert "step 3/20" in out


def test_renders_tokens_in_and_out():
    s = _state(step_in=1200, step_out=85)
    out = _render_flow(s)
    assert "1,200" in out
    assert "85" in out


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
    s = _state(final_answer="Done - 3 files found.")
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
    assert "50.0k" in out
    assert "200.0k" in out
    assert "25.0%" in out


def test_renders_budget_bar_color_coded_red_when_high():
    s = _state(budget_used=180000, budget_limit=200000)
    out = _render_flow(s)
    assert "90.0%" in out
    assert "red" in out


def test_renders_streaming_text():
    s = _state(streaming_text="reading the file now")
    out = _render_flow(s)
    assert "▍" in out
    assert "reading the file now" in out


def test_long_input_truncated():
    s = _state(user_input="x" * 200)
    out = _render_flow(s)
    assert "…" in out


def test_no_user_input_still_renders():
    s = _state()
    out = _render_flow(s)
    assert "DATA FLOW" in out


def test_reset_clears_state():
    s = _state(user_input="old", step=5, total_in=1000, externalized=2)
    s.reset()
    assert s.user_input == ""
    assert s.step == 0
    assert s.total_in == 0
    assert s.externalized == 0


# --- new detailed metrics ---

def test_renders_cost():
    s = _state(cost_total=0.0042, cost_in=0.0017, cost_out=0.0025)
    out = _render_flow(s)
    assert "$" in out
    assert "$0.0042" in out
    assert "$0.0017" in out
    assert "$0.0025" in out


def test_renders_context_usage():
    s = _state(context_used=5000, context_window=1_048_576)
    out = _render_flow(s)
    assert "ctx" in out
    assert "0.5%" in out or "0.4%" in out  # ~0.48%


def test_renders_context_usage_high():
    s = _state(context_used=900_000, context_window=1_000_000)
    out = _render_flow(s)
    assert "90.0%" in out
    assert "red" in out


def test_renders_speed_metrics():
    s = _state(avg_tokens_per_second=87.5, last_step_duration=1.2, elapsed=4.2)
    out = _render_flow(s)
    assert "spd" in out
    assert "88 tok/s" in out
    assert "1.2s" in out
    assert "4.2s" in out


def test_renders_cached_tokens_with_color():
    s = _state(total_in=1000, total_cached=500)
    out = _render_flow(s)
    assert "%↻" in out  # compact cache indicator
    assert "50%" in out


def test_renders_dedup_hits():
    s = _state(dedup_hits=3)
    out = _render_flow(s)
    assert "dedup:3" in out


def test_renders_subagent_calls():
    s = _state(subagent_calls=2)
    out = _render_flow(s)
    assert "sub:2" in out


def test_renders_externalized_count():
    s = _state(externalized=1)
    out = _render_flow(s)
    assert "ext:1" in out


def test_renders_extras_none_when_empty():
    s = _state()
    out = _render_flow(s)
    assert "—" in out


def test_step_cached_displayed():
    s = _state(step_in=1000, step_out=50, step_cached=400)
    out = _render_flow(s)
    assert "/c400" in out


def test_detail_section_present():
    s = _state()
    out = _render_flow(s)
    assert "── DETAIL ──" in out


def test_cost_zero_displayed():
    s = _state(cost_total=0.0, cost_in=0.0, cost_out=0.0)
    out = _render_flow(s)
    assert "$0.0000" in out
