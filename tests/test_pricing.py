"""Tests for pricing model lookup and cost estimation."""

from __future__ import annotations

import pytest

from harness import pricing
from harness.pricing import (
    CONTEXT_WINDOWS,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_PRICING,
    PRICING,
    estimate_cost,
    get_context_window,
    get_pricing,
)


def test_known_model_pricing():
    p = get_pricing("gpt-4o-mini")
    assert p.input == 0.15
    assert p.output == 0.60
    assert p.cached_input == 0.075


def test_unknown_model_falls_back_to_default():
    p = get_pricing("totally-unknown-model-xyz")
    assert p == DEFAULT_PRICING


def test_known_context_window():
    assert get_context_window("gpt-4.1-nano") == 1_047_576
    assert get_context_window("gemini-2.5-flash") == 1_048_576
    assert get_context_window("claude-3-5-haiku") == 200_000


def test_unknown_context_window():
    assert get_context_window("mystery-model") == DEFAULT_CONTEXT_WINDOW


def test_estimate_cost_basic():
    # 1M input @ $0.15 + 1M output @ $0.60 = $0.75
    in_c, out_c, total = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert in_c == pytest.approx(0.15)
    assert out_c == pytest.approx(0.60)
    assert total == pytest.approx(0.75)


def test_estimate_cost_with_cached():
    # 100k input total, 50k cached, 50k output
    in_c, out_c, total = estimate_cost(
        "gpt-4o-mini",
        input_tokens=100_000,
        output_tokens=50_000,
        cached_input_tokens=50_000,
    )
    # 50k uncached @ $0.15/M = $0.0075
    # 50k cached @ $0.075/M = $0.00375
    # 50k output @ $0.60/M = $0.03
    assert in_c == pytest.approx(0.0075 + 0.00375)
    assert out_c == pytest.approx(0.03)
    assert total == pytest.approx(0.01125 + 0.03)


def test_estimate_cost_cached_clamps_to_input():
    # If cached > input, clamp to input (no negative)
    in_c, out_c, total = estimate_cost("gpt-4o-mini", 100, 50, cached_input_tokens=200)
    assert in_c >= 0
    assert total >= 0


def test_env_override_pricing(monkeypatch):
    monkeypatch.setenv("AI_INPUT_PRICE", "0.5")
    monkeypatch.setenv("AI_OUTPUT_PRICE", "2.0")
    monkeypatch.setenv("AI_CACHED_PRICE", "0.1")
    p = get_pricing("gpt-4o-mini")  # any model, override applies
    assert p.input == 0.5
    assert p.output == 2.0
    assert p.cached_input == 0.1


def test_env_override_context(monkeypatch):
    monkeypatch.setenv("AI_CONTEXT_WINDOW", "999000")
    assert get_context_window("gpt-4o-mini") == 999000


def test_pricing_table_has_major_models():
    expected = {"gpt-4o-mini", "gpt-4.1-nano", "gemini-2.5-flash", "grok-2-latest"}
    assert expected <= set(PRICING.keys())


def test_context_table_has_major_models():
    expected = {"gpt-4o-mini", "gemini-2.5-flash", "grok-2-latest"}
    assert expected <= set(CONTEXT_WINDOWS.keys())


def test_format_dollars():
    # Just exercise the import / format helper
    from harness.flow import _fmt_dollars
    assert _fmt_dollars(0.000123).startswith("$")
    assert _fmt_dollars(1.234) == "$1.23"
