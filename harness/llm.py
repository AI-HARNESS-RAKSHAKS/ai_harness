"""OpenAI-compatible chat completions client."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import Config
from .tools import TOOL_SCHEMAS


@dataclass
class Completion:
    """One chat-completion result."""
    message: dict
    usage: dict  # raw usage dict from the API


class LLMClient:
    """Thin wrapper around an OpenAI-compatible /chat/completions endpoint.

    Only text and tool calls are supported; no image, audio, or video input.
    Static prefix (system prompt + tool schemas) is identical every call so
    providers with prompt caching can cache it automatically.
    """

    def __init__(self, config: Config):
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(180.0, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(self, messages: list[dict], system: str | None = None) -> Completion:
        """Send a chat-completion request and return the assistant message + usage."""
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

        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("choices"):
            raise RuntimeError(f"LLM returned no choices: {data}")

        return Completion(
            message=data["choices"][0]["message"],
            usage=data.get("usage") or {},
        )
