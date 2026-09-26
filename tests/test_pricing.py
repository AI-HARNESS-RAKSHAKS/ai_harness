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


# --- DeepSeek (the judges will use this) ---

def test_deepseek_chat_pricing():
    p = get_pricing("deepseek-chat")
    assert p.input > 0
    assert p.output > p.input
    assert p.cached_input > 0
    assert p.cached_input < p.input


def test_deepseek_context_windows():
    assert get_context_window("deepseek-chat") == 64_000
    assert get_context_window("deepseek-reasoner") == 128_000
    assert get_context_window("deepseek-flash") == 64_000


def test_deepseek_flash_is_cheapest():
    flash = get_pricing("deepseek-flash")
    chat = get_pricing("deepseek-chat")
    assert flash.input < chat.input
    assert flash.output < chat.output


# --- Qwen via DashScope (judges will also use this) ---

def test_qwen_pricing():
    for m in ("qwen-turbo", "qwen-plus", "qwen-max", "qwen-coder-plus"):
        p = get_pricing(m)
        assert p.input > 0
        assert p.output > 0


def test_qwen_context_windows():
    # qwen-long has 10M context
    assert get_context_window("qwen-long") == 10_000_000
    assert get_context_window("qwen-turbo") == 1_000_000
    assert get_context_window("qwen-plus") == 128_000


def test_qwen_turbo_is_cheapest():
    turbo = get_pricing("qwen-turbo")
    max_ = get_pricing("qwen-max")
    assert turbo.input < max_.input
    assert turbo.output < max_.output


# --- Base URLs ---

def test_base_urls():
    from harness.pricing import BASE_URLS
    assert "deepseek" in BASE_URLS
    assert "qwen" in BASE_URLS
    assert BASE_URLS["deepseek"].startswith("https://")
    assert BASE_URLS["qwen"].startswith("https://")


# --- Cost estimate for judge models ---

def test_cost_deepseek_chat_typical_run():
    # 10k input, 1k output, 5k cached - typical 5-step coding task
    in_c, out_c, total = estimate_cost("deepseek-chat", 10_000, 1_000, 5_000)
    # 5k uncached @ $0.27/M = $0.00135
    # 5k cached @ $0.07/M = $0.00035
    # 1k output @ $1.10/M = $0.0011
    assert total < 0.01  # Less than a cent per task


def test_cost_qwen_turbo_typical_run():
    in_c, out_c, total = estimate_cost("qwen-turbo", 10_000, 1_000, 5_000)
    assert total < 0.005  # Even cheaper
