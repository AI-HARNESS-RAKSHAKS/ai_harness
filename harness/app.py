"""Textual TUI for the AI Harness."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Input, RichLog, Static

from . import tools
from .agent import Agent
from .config import load_config
from .llm import LLMClient
from .optimize import ToolOutputStore


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
    #log {
        background: #0e1117;
        color: #c9d1d9;
        border: round #30363d;
        padding: 1 2;
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
    ]

    TITLE = "AI Harness"

    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()
        tools.configure_limits(
            max_chars=self.config.max_tool_output_chars,
            max_lines=self.config.max_tool_output_lines,
        )
        tools.configure_output_store(ToolOutputStore.default(threshold=self.config.externalize_threshold))
        self.client = LLMClient(self.config)
        self.agent = Agent(self.config, self.client)
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Static(self._status_text("ready"), id="status")
        with Vertical():
            yield RichLog(id="log", wrap=True, highlight=True, markup=True, max_lines=5000)
        yield Input(placeholder="Describe a task... (Enter submit · Ctrl+K compact · Ctrl+C quit)", id="input")
        yield Footer()

    def _status_text(self, state: str) -> str:
        u = self.agent.usage
        b = self.agent.budget
        pct = int(b.fraction * 100)
        return (
            f"model: {self.config.model}   "
            f"tokens: in {u.input_tokens} (cached {u.cached_input_tokens}) | out {u.output_tokens}   "
            f"budget: {b.used}/{b.limit} ({pct}%)   "
            f"state: {state}"
        )

    def on_mount(self) -> None:
        log = self.query_one("#log", RichLog)
        log.write("[bold cyan]AI Harness[/bold cyan] [dim]- text-only coding agent[/dim]")
        log.write(f"[dim]Model: {self.config.model}   Base: {self.config.base_url}[/dim]")
        log.write(
            f"[dim]Budget {self.config.token_budget} tok · externalize >={self.config.externalize_threshold} chars · "
            f"max {self.config.max_output_tokens} out · history <={self.config.history_window} msgs[/dim]"
        )
        log.write(
            f"[dim]Sub-agent model: {self.config.subagent_model or '(inherit)'} · "
            f"parallel tools: {self.config.parallel_tools}[/dim]"
        )
        log.write("[dim]Shortcuts: Ctrl+L clear · Ctrl+R reset · Ctrl+K compact history · Ctrl+C quit[/dim]\n")

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
                await self._render(ev)
        except Exception as exc:  # pragma: no cover - defensive
            log.write(f"[bold red]x error[/bold red] {exc}")

        ext = self.agent.externalized_count
        ext_note = f"  · externalized: {ext}" if ext else ""
        log.write(f"[dim]↳ {self.agent.usage.summary()}{ext_note}[/dim]\n")
        self._busy = False
        self.query_one("#status", Static).update(self._status_text("ready"))

    async def _render(self, ev) -> None:
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
                log.write(f"[bold magenta]assistant[/bold magenta]\n{content}")
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
            externalized = isinstance(output, str) and output.startswith("[externalized:")
            label = "↳ externalized →" if externalized else "↳ output:"
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
        log = self.query_one("#log", RichLog)
        log.clear()
        log.write("[dim]Conversation reset (usage + budget + dedup cleared).[/dim]")
        self.query_one("#status", Static).update(self._status_text("ready"))

    async def action_compact(self) -> None:
        before = len(self.agent.messages)
        self.agent._compact_history()
        after = len(self.agent.messages)
        log = self.query_one("#log", RichLog)
        log.write(f"[dim]compacted history: {before} → {after} messages[/dim]")

    async def on_unmount(self) -> None:
        await self.client.aclose()


def main() -> None:
    HarnessApp().run()


if __name__ == "__main__":
    main()
