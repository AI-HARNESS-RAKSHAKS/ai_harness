"""Agent loop: drives the LLM and dispatches tool calls.

Optimised for low token/cost:
- Parallel tool execution (Pi-style).
- Tool output externalisation for oversized results.
- Superseded write elision before each LLM call.
- Token-budget hard-stop (strips tool_calls at the cap).
- Two-layer loop detection.
- Sub-agent spawning via the ``task`` tool with isolated context.
- Per-step timing + cost accumulation for the FlowPanel.

A ``FlowObserver`` callback can be attached to receive lightweight updates
for visualisation (UI panels, logs, etc.) without coupling the agent to
any particular UI.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

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
from .pricing import estimate_cost
from .tools import DONE_MARKER, execute_tool


SYSTEM_PROMPT = """You are AI Harness, a coding agent that operates inside a Textual TUI.

You have exactly these tools available:
- read_file(path)             read a file (output auto-truncated; huge outputs externalised)
- write_file(path, content)   write content to a file (overwrites existing)
- edit_file(path, old_string, new_string, replace_all=false)   in-place edit
- list_files(path=".")        list a directory (head-truncated)
- bash(command, timeout=30)   run a shell command (errors tail-kept)
- task(prompt, subagent_type="general", max_turns=N)
                                spawn an isolated sub-agent for a bounded subtask
- done(answer)                finish and report the final answer

Rules of engagement:
1. EXPLORE FIRST. Use list_files then read_file before any edit. Never guess file contents.
2. PREFER edit_file OVER write_file for existing files - it is safer and avoids full-file re-sends.
3. KEEP BASH COMMANDS SMALL AND FOCUSED. No destructive operations (no rm -rf, no git push, no installs without need).
4. DELEGATE bounded subtasks with task() instead of doing everything yourself.
5. BE CONCISE. No narration, no "let me think about this" prose. The user sees only your tool calls + done summary.
6. LARGE OUTPUTS: anything over a few KB is auto-externalised. The tool result tells you the path; call read_file if you need the full content.
7. WHEN THE TASK IS COMPLETE, call done(answer) with a short summary of what you did.

You will be told the budget (cumulative token cap) and the max steps in the system context. Do not exceed either."""


SUBAGENT_SYSTEM_PROMPT = """You are a focused sub-agent spawned by a parent agent.

You have the same tools as the parent (read_file, write_file, edit_file, list_files, bash, done) but NOT the `task` tool - do not spawn further sub-agents.

Rules:
- Solve the specific subtask given. Do not explore unrelated code.
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


# Type alias: a lightweight callback for UI observers
FlowObserver = Callable[[AgentEvent], None]


class Agent:
    def __init__(
        self,
        config: Config,
        client: LLMClient,
        observer: FlowObserver | None = None,
    ):
        self.config = config
        self.client = client
        self.observer = observer
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
        self.dedup_hits = 0
        self.subagent_calls = 0

        # Timing
        self.turn_started: float = 0.0
        self.last_step_duration: float = 0.0
        self.step_token_rates: list[float] = []  # tok/s per step

    def _emit(self, ev: AgentEvent) -> None:
        if self.observer is not None:
            try:
                self.observer(ev)
            except Exception:
                pass

    def reset(self) -> None:
        self.messages = []
        self.usage = TokenUsage()
        self.budget.reset()
        self.loop_detector.reset()
        self.dedup.clear()
        self.externalized_count = 0
        self.dedup_hits = 0
        self.subagent_calls = 0
        self.turn_started = 0.0
        self.last_step_duration = 0.0
        self.step_token_rates = []

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Drive the agent loop for a single user turn."""
        self.turn_started = time.monotonic()
        self.step_token_rates = []

        self.messages.append({"role": "user", "content": user_input})
        ev = AgentEvent("user", {"content": user_input})
        yield ev
        self._emit(ev)

        for step in range(self.config.max_steps):
            if self.budget.exceeded:
                ev = AgentEvent(
                    "error",
                    {"message": f"token budget exhausted ({self.budget.used}/{self.budget.limit})"},
                )
                yield ev
                self._emit(ev)
                return

            self._compact_history()
            step_start = time.monotonic()

            ev = AgentEvent("step", {"n": step + 1, "max": self.config.max_steps})
            yield ev
            self._emit(ev)

            model_messages = elide_superseded_writes(self.messages)

            try:
                completion = await self.client.chat(model_messages, system=SYSTEM_PROMPT)
            except Exception as exc:
                ev = AgentEvent("error", {"message": f"LLM call failed: {exc}"})
                yield ev
                self._emit(ev)
                return

            step_duration = time.monotonic() - step_start
            self.last_step_duration = step_duration

            self.usage.add(completion.usage)
            self.budget.add(int(completion.usage.get("total_tokens") or 0))

            step_in = int(completion.usage.get("prompt_tokens") or 0)
            step_out = int(completion.usage.get("completion_tokens") or 0)
            step_cached = int((completion.usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)

            # Compute cost for this step
            step_cost_in, step_cost_out, step_cost_total = estimate_cost(
                self.config.model, step_in, step_out, step_cached
            )
            # Accumulate cost in usage via a sidecar dict
            if not hasattr(self, "_cost_in"):
                self._cost_in = 0.0
                self._cost_out = 0.0
                self._cost_total = 0.0
            self._cost_in += step_cost_in
            self._cost_out += step_cost_out
            self._cost_total += step_cost_total

            # Tokens per second for this step
            tps = step_out / step_duration if step_duration > 0 else 0.0
            self.step_token_rates.append(tps)
            avg_tps = (
                sum(self.step_token_rates) / len(self.step_token_rates)
                if self.step_token_rates
                else 0.0
            )

            msg = completion.message
            content = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls")

            history_msg: dict = {"role": "assistant", "content": msg.get("content") or ""}
            if tool_calls:
                history_msg["tool_calls"] = tool_calls
            self.messages.append(history_msg)

            ev = AgentEvent(
                "assistant",
                {
                    "content": content,
                    "step_in": step_in,
                    "step_out": step_out,
                    "step_cached": step_cached,
                    "step_duration": step_duration,
                    "tokens_per_second": tps,
                    "avg_tokens_per_second": avg_tps,
                    "elapsed": time.monotonic() - self.turn_started,
                    "step_cost": step_cost_total,
                    "cost_in": self._cost_in,
                    "cost_out": self._cost_out,
                    "cost_total": self._cost_total,
                },
            )
            yield ev
            self._emit(ev)

            if content:
                pass  # content rendered via event above

            if not tool_calls:
                ev = AgentEvent("done", {"answer": msg.get("content") or ""})
                yield ev
                self._emit(ev)
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
                ev = AgentEvent(
                    "tool_call",
                    {"id": tc.get("id", ""), "name": name, "arguments": args},
                )
                yield ev
                self._emit(ev)

            for tc, name, args in parsed_calls:
                should_stop, reason = self.loop_detector.check(name, args)
                if should_stop:
                    ev = AgentEvent("error", {"message": f"loop detected: {reason}"})
                    yield ev
                    self._emit(ev)
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

                ev = AgentEvent(
                    "tool_result",
                    {
                        "id": tc_id,
                        "name": name,
                        "output": display_result,
                        "size": len(display_result) if isinstance(display_result, str) else 0,
                    },
                )
                yield ev
                self._emit(ev)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": display_result if isinstance(display_result, str) else str(display_result),
                })

            if done_answer is not None:
                ev = AgentEvent("done", {"answer": done_answer})
                yield ev
                self._emit(ev)
                return

        ev = AgentEvent(
            "error",
            {"message": f"reached max steps ({self.config.max_steps}) without finishing."},
        )
        yield ev
        self._emit(ev)

    async def _execute_one(self, name: str, args: dict, tc_id: str) -> str:
        if name in ("read_file", "list_files"):
            cached = self.dedup.get(name, args)
            if cached is not None:
                self.dedup_hits += 1
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

        self.subagent_calls += 1

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
