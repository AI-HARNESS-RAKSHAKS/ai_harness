"""Smoke tests: imports, config loading, tool schemas."""

from __future__ import annotations

import pytest


def test_modules_import():
    from harness import agent, app, config, llm, tools, optimize, flow  # noqa: F401
    assert hasattr(agent, "Agent")
    assert hasattr(agent, "TokenUsage")
    assert hasattr(app, "HarnessApp")
    assert hasattr(config, "load_config")
    assert hasattr(llm, "LLMClient")
    assert hasattr(tools, "TOOL_SCHEMAS")
    assert hasattr(optimize, "TokenBudget")
    assert hasattr(flow, "FlowPanel")
    assert hasattr(flow, "FlowState")


def test_config_requires_key(monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    from harness.config import load_config

    with pytest.raises(RuntimeError, match="AI_API_KEY"):
        load_config()


def test_config_loads_defaults(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "test-key")
    for var in (
        "AI_BASE_URL", "AI_MODEL", "AI_MAX_STEPS",
        "AI_MAX_OUTPUT_TOKENS", "AI_HISTORY_WINDOW",
        "AI_MAX_TOOL_CHARS", "AI_MAX_TOOL_LINES",
        "AI_EXTERNALIZE_THRESHOLD", "AI_TOKEN_BUDGET",
        "AI_LOOP_WINDOW", "AI_LOOP_IDENTICAL", "AI_LOOP_FREQ",
        "AI_PARALLEL_TOOLS", "AI_SUBAGENT_MODEL",
        "AI_SUBAGENT_MAX_STEPS", "AI_SUBAGENT_DEFAULT_MAX_TURNS",
    ):
        monkeypatch.delenv(var, raising=False)

    from harness.config import load_config

    cfg = load_config()
    assert cfg.api_key == "test-key"
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.max_steps == 15
    assert cfg.max_output_tokens == 1024
    assert cfg.history_window == 30
    assert cfg.max_tool_output_chars == 2500
    assert cfg.max_tool_output_lines == 120
    assert cfg.externalize_threshold == 3000
    assert cfg.token_budget == 80000
    assert cfg.loop_window == 15
    assert cfg.loop_identical_threshold == 4
    assert cfg.loop_tool_freq_threshold == 30
    assert cfg.parallel_tools is True
    assert cfg.subagent_model is None
    assert cfg.subagent_max_steps == 10
    assert cfg.subagent_default_max_turns == 8


def test_config_overrides(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "k")
    monkeypatch.setenv("AI_BASE_URL", "https://example.com/v1/")
    monkeypatch.setenv("AI_MODEL", "gpt-4.1-nano")
    monkeypatch.setenv("AI_MAX_STEPS", "5")
    monkeypatch.setenv("AI_MAX_OUTPUT_TOKENS", "512")
    monkeypatch.setenv("AI_HISTORY_WINDOW", "10")
    monkeypatch.setenv("AI_MAX_TOOL_CHARS", "1000")
    monkeypatch.setenv("AI_MAX_TOOL_LINES", "50")
    monkeypatch.setenv("AI_EXTERNALIZE_THRESHOLD", "2000")
    monkeypatch.setenv("AI_TOKEN_BUDGET", "50000")
    monkeypatch.setenv("AI_LOOP_WINDOW", "8")
    monkeypatch.setenv("AI_LOOP_IDENTICAL", "3")
    monkeypatch.setenv("AI_LOOP_FREQ", "20")
    monkeypatch.setenv("AI_PARALLEL_TOOLS", "false")
    monkeypatch.setenv("AI_SUBAGENT_MODEL", "gpt-4.1-nano")
    from harness.config import load_config

    cfg = load_config()
    assert cfg.base_url == "https://example.com/v1"
    assert cfg.model == "gpt-4.1-nano"
    assert cfg.max_steps == 5
    assert cfg.max_output_tokens == 512
    assert cfg.history_window == 10
    assert cfg.max_tool_output_chars == 1000
    assert cfg.max_tool_output_lines == 50
    assert cfg.externalize_threshold == 2000
    assert cfg.token_budget == 50000
    assert cfg.loop_window == 8
    assert cfg.loop_identical_threshold == 3
    assert cfg.loop_tool_freq_threshold == 20
    assert cfg.parallel_tools is False
    assert cfg.subagent_model == "gpt-4.1-nano"


def test_tool_schemas_present():
    from harness.tools import TOOL_SCHEMAS

    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert {"read_file", "write_file", "edit_file", "list_files", "bash", "task", "done"} <= names
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert params.get("additionalProperties") is False


def test_tool_schemas_are_lean():
    import json
    from harness.tools import TOOL_SCHEMAS

    serialized = json.dumps(TOOL_SCHEMAS)
    assert len(serialized) < 4000, f"tool schemas are {len(serialized)} bytes - too large"
