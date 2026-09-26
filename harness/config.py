"""Environment-driven configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    api_key: str
    base_url: str
    model: str
    max_steps: int
    max_output_tokens: int
    history_window: int
    max_tool_output_chars: int
    max_tool_output_lines: int
    # Optimization knobs (new)
    externalize_threshold: int
    token_budget: int
    loop_window: int
    loop_identical_threshold: int
    loop_tool_freq_threshold: int
    parallel_tools: bool
    subagent_model: str | None
    subagent_max_steps: int
    subagent_default_max_turns: int
    tombstone_recent_tools: int = 2


def load_config() -> Config:
    """Build a Config from environment variables.

    `AI_API_KEY` is mandatory; everything else falls back to defaults tuned
    for low token usage and cost.
    """
    api_key = os.environ.get("AI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "AI_API_KEY environment variable is not set. "
            "Export it before running: export AI_API_KEY=<your-key>"
        )

    return Config(
        api_key=api_key,
        base_url=os.environ.get("AI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        model=os.environ.get("AI_MODEL", "gpt-4o-mini"),
        max_steps=max(1, int(os.environ.get("AI_MAX_STEPS", "20"))),
        max_output_tokens=max(64, int(os.environ.get("AI_MAX_OUTPUT_TOKENS", "2048"))),
        history_window=max(8, int(os.environ.get("AI_HISTORY_WINDOW", "40"))),
        max_tool_output_chars=max(256, int(os.environ.get("AI_MAX_TOOL_CHARS", "4000"))),
        max_tool_output_lines=max(20, int(os.environ.get("AI_MAX_TOOL_LINES", "200"))),
        externalize_threshold=max(1000, int(os.environ.get("AI_EXTERNALIZE_THRESHOLD", "8000"))),
        token_budget=max(1000, int(os.environ.get("AI_TOKEN_BUDGET", "200000"))),
        loop_window=max(5, int(os.environ.get("AI_LOOP_WINDOW", "20"))),
        loop_identical_threshold=max(2, int(os.environ.get("AI_LOOP_IDENTICAL", "5"))),
        loop_tool_freq_threshold=max(5, int(os.environ.get("AI_LOOP_FREQ", "50"))),
        parallel_tools=os.environ.get("AI_PARALLEL_TOOLS", "true").lower() not in ("0", "false", "no"),
        subagent_model=os.environ.get("AI_SUBAGENT_MODEL") or None,
        subagent_max_steps=max(1, int(os.environ.get("AI_SUBAGENT_MAX_STEPS", "15"))),
        subagent_default_max_turns=max(1, int(os.environ.get("AI_SUBAGENT_DEFAULT_MAX_TURNS", "10"))),
        tombstone_recent_tools=max(1, int(os.environ.get("AI_TOMBSTONE_RECENT_TOOLS", "2"))),
    )
