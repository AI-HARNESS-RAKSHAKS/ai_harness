"""Tests for LLMClient retry, streaming, and error handling."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from harness.config import Config
from harness.llm import LLMClient, RETRYABLE_STATUS, StreamEvent


def _cfg() -> Config:
    return Config(
        api_key="test", base_url="https://example/v1", model="m",
        max_steps=5, max_output_tokens=512, history_window=10,
        max_tool_output_chars=4000, max_tool_output_lines=200,
        externalize_threshold=8000, token_budget=200000,
        loop_window=20, loop_identical_threshold=5, loop_tool_freq_threshold=50,
        parallel_tools=True, subagent_model=None, subagent_max_steps=15,
        subagent_default_max_turns=10,
    )


def _resp(status: int, body: dict | None = None, headers: dict | None = None) -> MagicMock:
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.reason_phrase = "OK" if status == 200 else "ERR"
    r.headers = headers or {}
    r.json.return_value = body or {"choices": [{"message": {"content": "ok"}}]}
    r.raise_for_status = MagicMock()
    if status >= 400:
        r.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status}", request=MagicMock(), response=r
        )
    return r


@pytest.mark.asyncio
async def test_retryable_status_set_includes_429_and_503():
    assert 429 in RETRYABLE_STATUS
    assert 503 in RETRYABLE_STATUS
    assert 502 in RETRYABLE_STATUS
    assert 504 in RETRYABLE_STATUS


@pytest.mark.asyncio
async def test_retry_succeeds_after_503():
    client = LLMClient(_cfg())
    success = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    responses = [
        _resp(503, {"error": "unavailable"}),
        _resp(503, {"error": "unavailable"}),
        _resp(200, success),
    ]
    client._client = AsyncMock()
    client._client.post = AsyncMock(side_effect=responses)

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await client.chat([{"role": "user", "content": "x"}])
    assert result.message["content"] == "hi"
    assert client._client.post.call_count == 3


@pytest.mark.asyncio
async def test_retry_gives_up_after_max_attempts():
    client = LLMClient(_cfg())
    responses = [_resp(503)] * 5
    client._client = AsyncMock()
    client._client.post = AsyncMock(side_effect=responses)

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat([{"role": "user", "content": "x"}])
    # Should have tried 5 times before giving up
    assert client._client.post.call_count == 5


@pytest.mark.asyncio
async def test_no_retry_on_400():
    client = LLMClient(_cfg())
    responses = [_resp(400, {"error": "bad request"})]
    client._client = AsyncMock()
    client._client.post = AsyncMock(side_effect=responses)

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat([{"role": "user", "content": "x"}])
    assert client._client.post.call_count == 1


@pytest.mark.asyncio
async def test_retry_on_connection_error():
    client = LLMClient(_cfg())
    success = {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    client._client = AsyncMock()
    client._client.post = AsyncMock(side_effect=[
        httpx.ConnectError("connection refused"),
        httpx.ConnectError("connection refused"),
        _resp(200, success),
    ])

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await client.chat([{"role": "user", "content": "x"}])
    assert result.message["content"] == "ok"
    assert client._client.post.call_count == 3


@pytest.mark.asyncio
async def test_retry_respects_retry_after_header():
    client = LLMClient(_cfg())
    success = {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    responses = [
        _resp(429, headers={"retry-after": "2"}),
        _resp(200, success),
    ]
    client._client = AsyncMock()
    client._client.post = AsyncMock(side_effect=responses)

    with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
        await client.chat([{"role": "user", "content": "x"}])
    # Should sleep at least 2 seconds (the Retry-After value)
    assert any(call.args[0] >= 2 for call in mock_sleep.call_args_list)


# ---- streaming ----------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_parses_content_deltas():
    """Multiple SSE chunks should produce multiple content events."""
    chunks = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n',
        'data: [DONE]\n\n',
    ]
    body = "".join(chunks).encode()

    async def aiter_lines_impl():
        for line in body.decode().split("\n"):
            yield line.rstrip()

    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.aiter_lines = aiter_lines_impl
    resp.aclose = AsyncMock()

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=None)

    client = LLMClient(_cfg())
    client._client = MagicMock()
    client._client.stream = MagicMock(return_value=ctx)

    events = []
    async for ev in client.chat_stream([{"role": "user", "content": "x"}]):
        events.append(ev)

    contents = [ev.text for ev in events if ev.type == "content"]
    assert contents == ["Hello", " world", "!"]
    done = [ev for ev in events if ev.type == "done"]
    assert len(done) == 1
    assert done[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_stream_assembles_tool_calls():
    """Tool calls arrive incrementally and should be assembled."""
    chunks = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":""}}]}}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":"}}]}}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"x.py\\"}"}}]}}]}\n\n',
        'data: [DONE]\n\n',
    ]
    body = "".join(chunks).encode()

    async def aiter_lines_impl():
        for line in body.decode().split("\n"):
            yield line.rstrip()

    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.aiter_lines = aiter_lines_impl
    resp.aclose = AsyncMock()

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=None)

    client = LLMClient(_cfg())
    client._client = MagicMock()
    client._client.stream = MagicMock(return_value=ctx)

    events = []
    async for ev in client.chat_stream([{"role": "user", "content": "x"}]):
        events.append(ev)

    tcs = [ev for ev in events if ev.type == "tool_call"]
    assert len(tcs) == 1
    assert tcs[0].name == "read"
    assert tcs[0].arguments == {"path": "x.py"}
    assert tcs[0].id == "c1"
