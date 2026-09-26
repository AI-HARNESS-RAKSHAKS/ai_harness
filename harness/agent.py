"""Agent loop: drives the LLM and dispatches tool calls.

Optimised for low token/cost:
- Parallel tool execution (Pi-style).
- Tool output externalisation for oversized results.
- Superseded write elision before each LLM call.
- Token-budget hard-stop (strips tool_calls at the cap).
- Two-layer loop detection.
- Sub-agent spawning via the ``task`` tool with isolated context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator

from .config import Config
from .llm import LLMClient
from .optimize import (
    CallDedup,
    LoopDetector,
    TokenBudget,
    ToolOutputStore,
    elide_superseded_writes,
    gather_tool_results,
)
from .tools import DONE_MARKER, execute_tool


SYSTEM_PROMPT = """Coding agent. Use tools to act on the working directory.

Tools:
- read_file(path)            read a file (output truncated; huge outputs externalised)
- write_file(path, content)  write a file (overwrites)
- edit_file(path, old, new, replace_all=false)  in-place edit
- list_files(path=".")       list a directory
- bash(command, timeout=30)  run a shell command (errors are tail-kept)
- task(prompt, subagent_type="general", max_turns=N)
                               spawn an isolated sub-agent for a subtask
- done(answer)               finish with a short summary

Rules: explore first (list_files / read_file). Prefer edit_file over write_file. Run small bash commands. Avoid destructive ops. Be concise - no narration. Use task() to delegate bounded subtasks instead of doing everything yourself. Call done when complete.

Tool outputs: large outputs are externalised to a file with a synopsis. Use read_file on the returned path if you need the full content."""


SUBAGENT_SYSTEM_PROMPT = """You are a focused sub-agent spawned by a parent agent.

You have access to the same tools as the parent (read_file, write_file, edit_file, list_files, bash, done) but NOT the `task` tool - do not spawn further sub-agents.

Rules:
- Solve the specific subtask given to you. Do not explore unrelated code.
- Be efficient: minimum tool calls, minimum narration.
- Call done(answer) with a concise result when finished.
- If you cannot finish within your turn budget, call done with a partial answer explaining what blocked you."""


@dataclass
class AgentEvent:
    type: str
    payload: dict = field(default_factory=dict)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    calls: int = 0

    def add(self, usage: dict) -> None:
        self.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.output_tokens += int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        self.cached_input_tokens += int(details.get("cached_tokens") or 0)
        self.calls += 1

    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        cached = f" (cached {self.cached_input_tokens})" if self.cached_input_tokens else ""
        return (
            f"calls:{self.calls}  in:{self.input_tokens}{cached}  "
            f"out:{self.output_tokens}  total:{self.total()}"
        )


class Agent:
    def __init__(self, config: Config, client: LLMClient):
        self.config = config
        self.client = client
        self.messages: list[dict] = []
        self.usage = TokenUsage()
        self.budget = TokenBudget(limit=config.token_budget)
        self.loop_detector = LoopDetector(
            window=config.loop_window,
            identical_threshold=config.loop_identical_threshold,
            tool_freq_threshold=config.loop_tool_freq_threshold,
        )
        self.output_store = ToolOutputStore.default(threshold=config.externalize_threshold)
        self.dedup = CallDedup()
        self.externalized_count = 0

    def reset(self) -> None:
        self.messages = []
        self.usage = TokenUsage()
        self.budget.reset()
        self.loop_detector.reset()
        self.dedup.clear()
        self.externalized_count = 0

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Drive the agent loop for a single user turn."""
        self.messages.append({"role": "user", "content": user_input})
        yield AgentEvent("user", {"content": user_input})

        for step in range(self.config.max_steps):
            if self.budget.exceeded:
                yield AgentEvent(
                    "error",
                    {"message": f"token budget exhausted ({self.budget.used}/{self.budget.limit})"},
                )
                return

            self._compact_history()
            yield AgentEvent("step", {"n": step + 1, "max": self.config.max_steps})

            # Apply elision on a copy of messages - keep state intact.
            model_messages = elide_superseded_writes(self.messages)

            try:
                completion = await self.client.chat(model_messages, system=SYSTEM_PROMPT)
            except Exception as exc:
                yield AgentEvent("error", {"message": f"LLM call failed: {exc}"})
                return

            self.usage.add(completion.usage)
            self.budget.add(int(completion.usage.get("total_tokens") or 0))

            msg = completion.message
            content = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls")

            history_msg: dict = {"role": "assistant", "content": msg.get("content") or ""}
            if tool_calls:
                history_msg["tool_calls"] = tool_calls
            self.messages.append(history_msg)

            if content:
                yield AgentEvent("assistant", {"content": content})

            if not tool_calls:
                yield AgentEvent("done", {"answer": msg.get("content") or ""})
                return

            parsed_calls: list[tuple[dict, str, dict]] = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                parsed_calls.append((tc, name, args))

            for tc, name, args in parsed_calls:
                yield AgentEvent(
                    "tool_call",
                    {"id": tc.get("id", ""), "name": name, "arguments": args},
                )

            for tc, name, args in parsed_calls:
                should_stop, reason = self.loop_detector.check(name, args)
                if should_stop:
                    yield AgentEvent("error", {"message": f"loop detected: {reason}"})
                    return

            if self.config.parallel_tools and len(parsed_calls) > 1:
                coros = [
                    self._execute_one(name, args, tc.get("id", ""))
                    for tc, name, args in parsed_calls
                ]
                results = await gather_tool_results(coros)
            else:
                results = []
                for tc, name, args in parsed_calls:
                    results.append(await self._execute_one(name, args, tc.get("id", "")))

            done_answer: str | None = None
            for (tc, name, args), result in zip(parsed_calls, results):
                tc_id = tc.get("id", "")
                display_result = result
                if isinstance(result, str) and result.startswith(DONE_MARKER):
                    done_answer = result[len(DONE_MARKER):]
                    display_result = "(task completed)"
                elif isinstance(result, str) and len(result) > self.output_store.threshold:
                    new_result, _path = self.output_store.maybe_externalize(result, name)
                    if new_result != result:
                        self.externalized_count += 1
                        display_result = new_result

                yield AgentEvent(
                    "tool_result",
                    {"id": tc_id, "name": name, "output": display_result},
                )
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": display_result if isinstance(display_result, str) else str(display_result),
                })

            if done_answer is not None:
                yield AgentEvent("done", {"answer": done_answer})
                return

        yield AgentEvent(
            "error",
            {"message": f"reached max steps ({self.config.max_steps}) without finishing."},
        )

    async def _execute_one(self, name: str, args: dict, tc_id: str) -> str:
        """Run one tool call, applying dedup + externalisation."""
        if name in ("read_file", "list_files"):
            cached = self.dedup.get(name, args)
            if cached is not None:
                return cached

        result = await execute_tool(name, args, agent=self)

        if isinstance(result, str) and name in ("read_file", "list_files"):
            if not result.startswith("ERROR:"):
                self.dedup.put(name, args, result)

        return result

    async def run_subagent(
        self,
        prompt: str,
        subagent_type: str = "general",
        max_turns: int | None = None,
    ) -> str:
        """Spawn an isolated sub-agent for a subtask.

        Returns a single string answer (truncated + externalised if huge).
        The sub-agent has its own LLM client (optionally a cheaper model),
        fresh context, bounded turns, and a focused system prompt.
        """
        bound = max_turns or self.config.subagent_default_max_turns

        from dataclasses import replace

        if self.config.subagent_model and self.config.subagent_model != self.config.model:
            child_config = replace(
                self.config,
                model=self.config.subagent_model,
                max_steps=bound,
                token_budget=max(2000, self.config.token_budget // 10),
            )
        else:
            child_config = replace(self.config, max_steps=bound)

        child_client = LLMClient(child_config)
        child = Agent(child_config, child_client)

        answer = ""
        try:
            async for ev in child.run(prompt):
                if ev.type == "done":
                    answer = ev.payload.get("answer", "")
                    break
                if ev.type == "error":
                    answer = f"sub-agent error: {ev.payload.get('message', 'unknown')}"
                    break
        finally:
            await child_client.aclose()

        if not answer:
            answer = "(sub-agent returned no answer)"

        if len(answer) > self.output_store.threshold:
            ext, _path = self.output_store.maybe_externalize(answer, f"subagent-{subagent_type}")
            return ext
        if len(answer) > 2000:
            answer = answer[:2000] + f"\n... [sub-agent answer truncated, {len(answer)} chars total]"
        return answer

    def _compact_history(self) -> None:
        """Bound the conversation history to keep input tokens low."""
        window = self.config.history_window
        if len(self.messages) <= window:
            return

        first_user_idx = 0
        for i, m in enumerate(self.messages):
            if m.get("role") == "user":
                first_user_idx = i
                break

        head = self.messages[: first_user_idx + 1]
        tail_size = window - len(head)
        if tail_size <= 0:
            tail_size = window // 2
            head = self.messages[:tail_size]
            tail_size = window - tail_size

        tail = self.messages[-tail_size:]
        if len(head) + len(tail) >= len(self.messages):
            return
        self.messages = head + tail
