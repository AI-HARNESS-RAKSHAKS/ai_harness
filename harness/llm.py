"""OpenAI-compatible chat completions client.

Features:
- Retry with exponential backoff on transient errors (429, 503, timeouts).
- Streaming responses (SSE parsing).
- Usage tracking (input / output / cached tokens).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

import httpx

from .config import Config
from .tools import TOOL_SCHEMAS


# Status codes worth retrying. 429 = rate-limited, 503 = unavailable,
# 502/504 = gateway issues.
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class Completion:
    """One non-streaming chat-completion result."""
    message: dict
    usage: dict  # raw usage dict from the API


@dataclass
class StreamEvent:
    """One parsed SSE event from a streaming chat-completion.

    - type='content': text delta arrived (text=...)
    - type='tool_call': a tool call is now complete (id, name, arguments)
    - type='done': stream finished (finish_reason)
    - type='error': an error occurred
    """
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    arguments: dict = field(default_factory=dict)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    error: str = ""


class LLMClient:
    """Thin wrapper around an OpenAI-compatible /chat/completions endpoint.

    Only text and tool calls are supported; no image, audio, or video input.
    Static prefix (system prompt + tool schemas) is identical every call so
    providers with prompt caching can cache it automatically.
    """

    def __init__(self, config: Config):
        self.config = config
        # Tighter timeouts so hung providers fail fast (the agent retries).
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=5.0, read=30.0),
            # Keep connections alive between requests (HTTP keep-alive).
            http2=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- non-streaming --------------------------------------------------------

    async def chat(self, messages: list[dict], system: str | None = None) -> Completion:
        """Send a chat-completion request. Retries transient errors with backoff."""
        payload_messages = list(messages)
        if system:
            payload_messages = [{"role": "system", "content": system}, *payload_messages]

        payload = {
            "model": self.config.model,
            "messages": payload_messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "max_tokens": self.config.max_output_tokens,
        }

        data = await self._post_with_retry("/chat/completions", payload)
        if not data.get("choices"):
            raise RuntimeError(f"LLM returned no choices: {data}")
        return Completion(
            message=data["choices"][0]["message"],
            usage=data.get("usage") or {},
        )

    async def _post_with_retry(self, path: str, payload: dict) -> dict:
        """POST with exponential backoff for retryable errors."""
        max_attempts = 5
        base_backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                resp = await self._client.post(path, json=payload)
                if resp.status_code in RETRYABLE_STATUS and attempt < max_attempts - 1:
                    last_exc = httpx.HTTPStatusError(
                        f"{resp.status_code} {resp.reason_phrase}",
                        request=resp.request,
                        response=resp,
                    )
                    await self._sleep_backoff(attempt, base_backoff, resp)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    await self._sleep_backoff(attempt, base_backoff, None)
                    continue
                raise

        # Exhausted retries
        if last_exc:
            raise last_exc
        raise RuntimeError("LLM call failed after retries with no captured exception")

    @staticmethod
    async def _sleep_backoff(attempt: int, base: float, resp: httpx.Response | None) -> None:
        """Sleep with exponential backoff; honour Retry-After if present."""
        delay = base * (2 ** attempt)
        delay = min(delay, 30.0)  # cap at 30s
        if resp is not None:
            ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            if ra:
                try:
                    delay = max(delay, float(ra))
                except ValueError:
                    pass
        await asyncio.sleep(delay)

    # ---- streaming ------------------------------------------------------------

    async def chat_stream(
        self,
        messages: list[dict],
        system: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion. Yields StreamEvents.

        Text deltas arrive as 'content' events; completed tool calls arrive as
        'tool_call' events; 'done' marks end-of-stream with usage.
        """
        payload_messages = list(messages)
        if system:
            payload_messages = [{"role": "system", "content": system}, *payload_messages]

        payload = {
            "model": self.config.model,
            "messages": payload_messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "max_tokens": self.config.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Per-request timeout for SSE stalls (long thinking pauses).
        max_attempts = 3
        base_backoff = 1.0

        for attempt in range(max_attempts):
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload
                ) as resp:
                    if resp.status_code in RETRYABLE_STATUS and attempt < max_attempts - 1:
                        await resp.aclose()
                        await self._sleep_backoff(attempt, base_backoff, resp)
                        continue
                    resp.raise_for_status()
                    async for ev in self._parse_sse(resp):
                        yield ev
                    return
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                if attempt < max_attempts - 1:
                    await self._sleep_backoff(attempt, base_backoff, None)
                    continue
                yield StreamEvent(type="error", error=f"stream failed: {exc}")
                return

    async def _parse_sse(self, resp: httpx.Response) -> AsyncIterator[StreamEvent]:
        """Parse SSE chunks from an OpenAI-compatible streaming response."""
        tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments_str}
        finish_reason = ""
        usage: dict = {}

        def _flush_tool_calls():
            for idx in sorted(tool_calls):
                tc = tool_calls[idx]
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                yield StreamEvent(
                    type="tool_call",
                    id=tc["id"],
                    name=tc["name"],
                    arguments=args,
                )

        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith(":"):
                continue  # SSE comment
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data:
                continue
            if data == "[DONE]":
                # Flush any accumulated tool calls BEFORE the done event.
                for ev in _flush_tool_calls():
                    yield ev
                yield StreamEvent(type="done", finish_reason=finish_reason, usage=usage)
                return
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            # Some providers send a final usage-only chunk with empty choices.
            chunk_usage = obj.get("usage")
            if chunk_usage:
                usage = chunk_usage

            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            fr = choice.get("finish_reason") or ""
            if fr:
                finish_reason = fr

            content = delta.get("content")
            if content:
                yield StreamEvent(type="content", text=content)

            for tc in (delta.get("tool_calls") or []):
                idx = tc.get("index", 0)
                entry = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    entry["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    entry["name"] = fn["name"]
                if fn.get("arguments"):
                    entry["arguments"] += fn["arguments"]

        # Stream ended without [DONE]
        for ev in _flush_tool_calls():
            yield ev
        yield StreamEvent(type="done", finish_reason=finish_reason, usage=usage)
