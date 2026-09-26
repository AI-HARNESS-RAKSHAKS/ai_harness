"""Textual TUI for the AI Harness.

Layout:
  ┌── status bar (top, 1 line) ───────────────────────────────────┐
  │                                                              │
  │  ┌── main log (left, takes remaining space) ──┐ ┌── flow ─┐│
  │  │                                            │ │ panel  ││
  │  │  user input + tool calls + final answer    │ │ (right ││
  │  │                                            │ │ side)  ││
  │  │                                            │ │        ││
  │  └────────────────────────────────────────────┘ └────────┘│
  │                                                              │
  ├── input box (bottom) ────────────────────────────────────────┤
  └── footer ────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Input, RichLog, Static

from . import tools
from .agent import Agent, AgentEvent
from .config import load_config
from .flow import DEFAULT_SIDEBAR_WIDTH, FlowPanel, FlowState
from .llm import LLMClient
from .optimize import ToolOutputStore
from .pricing import get_context_window


class HarnessApp(App):
    CSS = """
    Screen {
        background: #0e1117;
    }
    #status {
        dock: top;
        height: 1;
        background: #161b22;
        color: #8b949e;
        padding: 0 1;
    }
    #main {
        height: 1fr;
        width: 1fr;
    }
    #log {
        background: #0e1117;
        color: #c9d1d9;
        border: round #30363d;
        padding: 1 2;
    }
    #flow {
        dock: right;
        width: 46;
        background: #0d1117;
        color: #c9d1d9;
        border: round #30363d;
        padding: 0 0;
        margin: 0 1 0 0;
    }
    #input {
        dock: bottom;
        background: #161b22;
        border: round #30363d;
    }
    Input > .input--cursor {
        background: #58a6ff;
        color: #0e1117;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("ctrl+r", "reset", "Reset"),
        Binding("ctrl+k", "compact", "Compact"),
        Binding("ctrl+d", "toggle_flow", "Hide/Show flow"),
        Binding("ctrl+b", "toggle_sidebar", "Dock bottom"),
    ]

    TITLE = "AI Harness"

    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()
        tools.configure_limits(
            max_chars=self.config.max_tool_output_chars,
            max_lines=self.config.max_tool_output_lines,
        )
        tools.configure_output_store(
            ToolOutputStore.default(threshold=self.config.externalize_threshold)
        )
        ctx_window = get_context_window(self.config.model)
        self.flow_state = FlowState(
            model=self.config.model,
            base_url=self.config.base_url,
            budget_limit=self.config.token_budget,
            max_steps=self.config.max_steps,
            context_window=ctx_window,
            sidebar_width=DEFAULT_SIDEBAR_WIDTH,
        )
        self.client = LLMClient(self.config)
        self.agent = Agent(self.config, self.client, observer=self._observe)
        self._busy = False
        self._flow_visible = True
        self._sidebar_docked = "right"

    def compose(self) -> ComposeResult:
        yield Static(self._status_text("ready"), id="status")
        with Horizontal():
            with Vertical(id="main"):
                yield RichLog(
                    id="log", wrap=True, highlight=True, markup=True, max_lines=5000
                )
            yield FlowPanel(self.flow_state)
        yield Input(
            placeholder="Describe a task... (Enter submit · Ctrl+D hide flow · Ctrl+C quit)",
            id="input",
        )
        yield Footer()

    def _status_text(self, state: str) -> str:
        u = self.agent.usage
        b = self.agent.budget
        pct = int(b.fraction * 100)
        cost = getattr(self.agent, "_cost_total", 0.0)
        mode = "stream" if self.agent.streaming else "batch"
        return (
            f"model: {self.config.model}   "
            f"in {u.input_tokens} (cached {u.cached_input_tokens}) | out {u.output_tokens}   "
            f"cost ${cost:.4f}   "
            f"budget {b.used}/{b.limit} ({pct}%)   "
            f"mode {mode}   "
            f"{state}"
        )

    def on_mount(self) -> None:
        log = self.query_one("#log", RichLog)
        log.write("[bold cyan]AI Harness[/bold cyan] [dim]- text-only coding agent[/dim]")
        log.write(f"[dim]Model: {self.config.model}   Base: {self.config.base_url}[/dim]")
        ctx = get_context_window(self.config.model)
        log.write(
            f"[dim]Context: {ctx:,} tok · Budget: {self.config.token_budget:,} tok · "
            f"Sub-agent: {self.config.subagent_model or '(inherit)'} · "
            f"Parallel: {self.config.parallel_tools}[/dim]"
        )
        log.write(
            "[dim]Auto-retry: 408/409/425/429/500/502/503/504 (5 attempts, honours Retry-After).[/dim]"
        )
        log.write(
            "[dim]Shortcuts: Ctrl+L clear · Ctrl+R reset · Ctrl+K compact · "
            "Ctrl+D hide flow · Ctrl+B dock bottom · Ctrl+C quit[/dim]\n"
        )

    # --- FlowPanel observer (runs synchronously, never raises) ---------------

    def _observe(self, ev: AgentEvent) -> None:
        """Push AgentEvents into the FlowState and refresh the panel."""
        s = self.flow_state
        if ev.type == "user":
            s.user_input = ev.payload.get("content", "")
            s.final_answer = ""
            s.error = ""
            s.step_in = 0
            s.step_out = 0
            s.step_cached = 0
            s.pending_tools = []
            s.finished_tools = []
            s.streaming_text = ""
            s.is_thinking = False
        elif ev.type == "step":
            s.step = ev.payload.get("n", 0)
            s.max_steps = ev.payload.get("max", s.max_steps)
            s.step_in = 0
            s.step_out = 0
            s.step_cached = 0
            s.pending_tools = []
            s.finished_tools = []
            s.streaming_text = ""
            s.last_step_duration = 0.0
            s.is_thinking = False
        elif ev.type == "thinking":
            s.is_thinking = True
            s.streaming_text = ""
            s.elapsed = ev.payload.get("elapsed", s.elapsed)
        elif ev.type == "content_delta":
            s.is_thinking = False
            text = ev.payload.get("text", "")
            s.streaming_text += text
            if len(s.streaming_text) > 1000:
                s.streaming_text = "…" + s.streaming_text[-500:]
        elif ev.type == "assistant":
            s.is_thinking = False
            s.streaming_text = ""
            s.last_assistant = ev.payload.get("content", "")[:200]
            s.step_in = ev.payload.get("step_in", s.step_in)
            s.step_out = ev.payload.get("step_out", s.step_out)
            s.step_cached = ev.payload.get("step_cached", s.step_cached)
            s.last_step_duration = ev.payload.get("step_duration", s.last_step_duration)
            s.last_ttft = ev.payload.get("ttft", s.last_ttft)
            s.avg_tokens_per_second = ev.payload.get("avg_tokens_per_second", s.avg_tokens_per_second)
            s.elapsed = ev.payload.get("elapsed", s.elapsed)
            s.is_streaming = ev.payload.get("streamed", s.is_streaming)
            s.cost_in = ev.payload.get("cost_in", s.cost_in)
            s.cost_out = ev.payload.get("cost_out", s.cost_out)
            s.cost_total = ev.payload.get("cost_total", s.cost_total)
        elif ev.type == "tool_call":
            s.is_thinking = False
            name = ev.payload.get("name", "")
            s.pending_tools.append(name)
        elif ev.type == "tool_result":
            if s.pending_tools:
                s.finished_tools = s.pending_tools
                s.pending_tools = []
        elif ev.type == "done":
            s.final_answer = ev.payload.get("answer", "")
            s.pending_tools = []
            s.is_thinking = False
        elif ev.type == "error":
            s.error = ev.payload.get("message", "error")
            s.is_thinking = False

        s.total_in = self.agent.usage.input_tokens
        s.total_out = self.agent.usage.output_tokens
        s.total_cached = self.agent.usage.cached_input_tokens
        s.total_calls = self.agent.usage.calls
        s.budget_used = self.agent.budget.used
        s.budget_limit = self.agent.budget.limit
        s.externalized = self.agent.externalized_count
        s.dedup_hits = self.agent.dedup_hits
        s.subagent_calls = self.agent.subagent_calls
        if self.agent.usage.input_tokens > 0:
            s.context_used = self.agent.usage.input_tokens

        try:
            self.query_one("#flow", FlowPanel).refresh()
        except Exception:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._busy:
            return
        text = event.value.strip()
        if not text:
            return
        self.query_one("#input", Input).value = ""
        await self._handle_user_input(text)

    async def _handle_user_input(self, text: str) -> None:
        self._busy = True
        log = self.query_one("#log", RichLog)
        log.write(f"[bold green]>[/bold green] {text}")

        try:
            async for ev in self.agent.run(text):
                await self._render_event(ev)
        except Exception as exc:  # pragma: no cover - defensive
            log.write(f"[bold red]x error[/bold red] {exc}")

        ext = self.agent.externalized_count
        ext_note = f"  · externalized: {ext}" if ext else ""
        cost = getattr(self.agent, "_cost_total", 0.0)
        log.write(
            f"[dim]↳ {self.agent.usage.summary()} · ${cost:.4f}{ext_note}[/dim]\n"
        )
        self._busy = False
        self.query_one("#status", Static).update(self._status_text("ready"))

    async def _render_event(self, ev: AgentEvent) -> None:
        log = self.query_one("#log", RichLog)
        status = self.query_one("#status", Static)

        if ev.type == "user":
            return
        if ev.type == "step":
            n, mx = ev.payload["n"], ev.payload["max"]
            status.update(self._status_text(f"step {n}/{mx}"))
            return
        if ev.type == "assistant":
            content = ev.payload["content"].strip()
            if content:
                dur = ev.payload.get("step_duration", 0.0)
                tok = ev.payload.get("step_out", 0)
                tps = ev.payload.get("tokens_per_second", 0.0)
                cost = ev.payload.get("step_cost", 0.0)
                ttft = ev.payload.get("ttft", 0.0)
                dur_s = f"{dur:.1f}s" if dur >= 1 else f"{int(dur * 1000)}ms"
                ttft_s = f"{int(ttft * 1000)}ms" if ttft else "—"
                log.write(
                    f"[bold magenta]assistant[/bold magenta] "
                    f"[dim]({dur_s} · ttft {ttft_s} · {tok} tok · {tps:.0f} tok/s · ${cost:.4f})[/dim]\n{content}"
                )
            return
        if ev.type == "tool_call":
            name = ev.payload["name"]
            args = ev.payload["arguments"]
            args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
            if name == "task":
                log.write(
                    f"[bold yellow]⚙ task[/bold yellow] "
                    f"[dim](subagent={args.get('subagent_type', 'general')}, "
                    f"prompt={repr(args.get('prompt', ''))[:80]}...)[/dim]"
                )
            else:
                log.write(f"[bold yellow]⚙ {name}[/bold yellow] [dim]({args_str})[/dim]")
            return
        if ev.type == "tool_result":
            output = ev.payload["output"]
            sz = ev.payload.get("size", 0)
            externalized = isinstance(output, str) and output.startswith("[externalized:")
            label = "↳ externalized →" if externalized else f"↳ output ({sz} chars)"
            log.write(f"[dim]{label}[/dim]")
            for line in str(output).splitlines()[:40]:
                log.write(f"    {line}")
            return
        if ev.type == "done":
            log.write(f"[bold cyan]✓ done[/bold cyan] [dim]{ev.payload.get('answer', '')}[/dim]")
            return
        if ev.type == "error":
            log.write(f"[bold red]x error[/bold red] {ev.payload['message']}")
            return

    async def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

    async def action_reset(self) -> None:
        self.agent.reset()
        self.flow_state.reset()
        self.flow_state.model = self.config.model
        self.flow_state.base_url = self.config.base_url
        self.flow_state.budget_limit = self.config.token_budget
        self.flow_state.max_steps = self.config.max_steps
        self.flow_state.context_window = get_context_window(self.config.model)
        log = self.query_one("#log", RichLog)
        log.clear()
        log.write("[dim]Conversation reset (usage + budget + dedup + cost cleared).[/dim]")
        self.query_one("#status", Static).update(self._status_text("ready"))
        self.query_one("#flow", FlowPanel).refresh()

    async def action_compact(self) -> None:
        before = len(self.agent.messages)
        self.agent._compact_history()
        after = len(self.agent.messages)
        log = self.query_one("#log", RichLog)
        log.write(f"[dim]compacted history: {before} → {after} messages[/dim]")

    async def action_toggle_flow(self) -> None:
        self._flow_visible = not self._flow_visible
        try:
            self.query_one("#flow", FlowPanel).display = self._flow_visible
        except Exception:
            pass

    async def action_toggle_sidebar(self) -> None:
        """Toggle FlowPanel between right-side dock and bottom dock."""
        try:
            flow = self.query_one("#flow", FlowPanel)
        except Exception:
            return
        # Remove existing style and apply new dock
        flow.set_class(False, "docked-right", "docked-bottom")
        if self._sidebar_docked == "right":
            self._sidebar_docked = "bottom"
            flow.set_class(True, "docked-bottom")
            flow.styles.width = "100%"
            flow.styles.height = "30%"
            flow.styles.dock = "bottom"
        else:
            self._sidebar_docked = "right"
            flow.styles.width = "46"
            flow.styles.height = "100%"
            flow.styles.dock = "right"

    async def on_unmount(self) -> None:
        await self.client.aclose()


def main() -> None:
    HarnessApp().run()


if __name__ == "__main__":
    main()
