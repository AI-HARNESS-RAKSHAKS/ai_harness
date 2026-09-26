"""Pricing and context-window data for common LLM models.

Used by the FlowPanel to estimate cost and show context usage.
All prices are USD per million tokens. Cached input is the discounted
rate (most providers charge 50% for cached input).

Override at runtime via env vars (optional):
- ``AI_INPUT_PRICE``  - $ / 1M input tokens
- ``AI_OUTPUT_PRICE`` - $ / 1M output tokens
- ``AI_CACHED_PRICE`` - $ / 1M cached input tokens
- ``AI_CONTEXT_WINDOW`` - model context window in tokens
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input: float          # $ / 1M input tokens
    output: float         # $ / 1M output tokens
    cached_input: float   # $ / 1M cached input tokens (defaults to 50% of input)


PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-5-nano":           ModelPricing(0.05, 0.40, 0.005),
    "gpt-5-mini":           ModelPricing(0.25, 2.00, 0.025),
    "gpt-4.1-nano":         ModelPricing(0.10, 0.40, 0.025),
    "gpt-4.1-mini":         ModelPricing(0.40, 1.60, 0.10),
    "gpt-4o-mini":          ModelPricing(0.15, 0.60, 0.075),
    "gpt-4o":               ModelPricing(2.50, 10.00, 1.25),
    "o4-mini":              ModelPricing(1.10, 4.40, 0.275),
    # Google Gemini
    "gemini-2.5-flash":     ModelPricing(0.075, 0.30, 0.01875),
    "gemini-2.5-pro":       ModelPricing(1.25, 10.00, 0.31),
    "gemini-2.0-flash":     ModelPricing(0.10, 0.40, 0.025),
    "gemini-3.8-flash":     ModelPricing(0.075, 0.30, 0.01875),  # custom proxy
    # xAI Grok
    "grok-2":               ModelPricing(2.00, 10.00, 1.00),
    "grok-2-latest":        ModelPricing(2.00, 10.00, 1.00),
    "grok-3-mini":          ModelPricing(0.30, 0.50, 0.15),
    "grok-3":               ModelPricing(3.00, 15.00, 0.75),
    # Groq
    "llama-3.1-8b-instant": ModelPricing(0.05, 0.08, 0.05),
    "llama-3.3-70b-versatile": ModelPricing(0.59, 0.79, 0.59),
    # Anthropic (via gateway)
    "claude-3-5-haiku":     ModelPricing(0.80, 4.00, 0.08),
    "claude-3-5-sonnet":    ModelPricing(3.00, 15.00, 0.30),
    "claude-haiku-4-5":     ModelPricing(1.00, 5.00, 0.10),
}

DEFAULT_PRICING = ModelPricing(0.50, 1.50, 0.25)

# Context window per model (tokens).
CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5-nano":           400_000,
    "gpt-5-mini":           400_000,
    "gpt-4.1-nano":         1_047_576,
    "gpt-4.1-mini":         1_047_576,
    "gpt-4o-mini":          128_000,
    "gpt-4o":               128_000,
    "o4-mini":              200_000,
    "gemini-2.5-flash":     1_048_576,
    "gemini-2.5-pro":       2_097_152,
    "gemini-2.0-flash":     1_048_576,
    "gemini-3.8-flash":     1_048_576,
    "grok-2":               131_072,
    "grok-2-latest":        131_072,
    "grok-3-mini":          131_072,
    "grok-3":               131_072,
    "llama-3.1-8b-instant": 131_072,
    "llama-3.3-70b-versatile": 131_072,
    "claude-3-5-haiku":     200_000,
    "claude-3-5-sonnet":    200_000,
    "claude-haiku-4-5":     200_000,
}

DEFAULT_CONTEXT_WINDOW = 128_000


def get_pricing(model: str) -> ModelPricing:
    """Look up pricing; check AI_* env overrides first."""
    if os.environ.get("AI_INPUT_PRICE") or os.environ.get("AI_OUTPUT_PRICE"):
        try:
            return ModelPricing(
                input=float(os.environ.get("AI_INPUT_PRICE") or DEFAULT_PRICING.input),
                output=float(os.environ.get("AI_OUTPUT_PRICE") or DEFAULT_PRICING.output),
                cached_input=float(os.environ.get("AI_CACHED_PRICE") or DEFAULT_PRICING.cached_input),
            )
        except ValueError:
            pass
    return PRICING.get(model, DEFAULT_PRICING)


def get_context_window(model: str) -> int:
    if os.environ.get("AI_CONTEXT_WINDOW"):
        try:
            return int(os.environ["AI_CONTEXT_WINDOW"])
        except ValueError:
            pass
    return CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> tuple[float, float, float]:
    """Return (input_cost, output_cost, total_cost) in USD."""
    p = get_pricing(model)
    cached_effective = min(cached_input_tokens, input_tokens)
    non_cached = max(0, input_tokens - cached_effective)
    in_cost = (non_cached / 1_000_000) * p.input
    cached_cost = (cached_effective / 1_000_000) * p.cached_input
    out_cost = (output_tokens / 1_000_000) * p.output
    return in_cost + cached_cost, out_cost, in_cost + cached_cost + out_cost


def estimate_cost_str(model: str, input_tokens: int, output_tokens: int, cached: int = 0) -> str:
    """Format a one-line cost summary."""
    in_c, out_c, total = estimate_cost(model, input_tokens, output_tokens, cached)
    return f"${total:.4f} (in ${in_c:.4f} + out ${out_c:.4f})"
