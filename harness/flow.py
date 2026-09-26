"""Visual data-flow panel for the TUI.

Renders the live pipeline showing:
- Where data is going (USER -> AGENT -> LLM -> TOOLS -> ...)
- How much data is flowing at each step (chars, tokens, call count)
- The active step, model, and budget usage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from textual.reactive import reactive
from textual.widgets import Static


@dataclass
class FlowState:
    """Mutable state used to render the FlowPanel."""

    model: str = ""
    base_url: str = ""
    user_input: str = ""
    step: int = 0
    max_steps: int = 0
    # Aggregate counters (across the whole turn)
    total_in: int = 0
    total_out: int = 0
    total_cached: int = 0
    budget_used: int = 0
    budget_limit: int = 0
    # Per-step state
    step_in: int = 0
    step_out: int = 0
    last_assistant: str = ""
    pending_tools: list[str] = field(default_factory=list)
    finished_tools: list[str] = field(default_factory=list)
    final_answer: str = ""
    error: str = ""
    # Streaming buffer (text arriving token-by-token)
    streaming_text: str = ""
    # Externalized count
    externalized: int = 0

    def reset(self) -> None:
        self.user_input = ""
        self.step = 0
        self.max_steps = 0
        self.total_in = 0
        self.total_out = 0
        self.total_cached = 0
        self.budget_used = 0
        self.budget_limit = 0
        self.step_in = 0
        self.step_out = 0
        self.last_assistant = ""
        self.pending_tools = []
        self.finished_tools = []
        self.final_answer = ""
        self.error = ""
        self.streaming_text = ""
        self.externalized = 0


def _trunc(s: str, n: int = 36) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


class FlowPanel(Static):
    """Live visualisation of the agent's data flow.

    Reads from a FlowState and re-renders itself on every change. The agent
    loop pushes updates into ``self.state`` from its event handler.
    """

    def __init__(self, state: FlowState, **kwargs: Any) -> None:
        super().__init__(id="flow", markup=True, **kwargs)
        self.state = state

    def render(self) -> str:
        return _render_flow(self.state)


def _render_flow(s: FlowState) -> str:
    """Render the FlowState as rich text. Pure function for testability."""
    box_w = 76
    sep = "─" * box_w
    lines: list[str] = []

    def row(content: str) -> str:
        return content

    # Header
    lines.append(f"[bold cyan]┌─ DATA FLOW PIPELINE {'─' * (box_w - 22)}[/bold cyan]")

    # USER -> AGENT
    if s.user_input:
        lines.append("[bold cyan]│[/bold cyan]")
        lines.append(
            f"[bold cyan]│[/bold cyan]  [bold green]USER[/bold green]  ── \"{_trunc(s.user_input, 50)}\" ──▶"
        )
        lines.append(f"[bold cyan]│[/bold cyan]        [dim]{len(s.user_input)} chars in[/dim]")

    # AGENT -> LLM (with current step)
    lines.append("[bold cyan]│[/bold cyan]")
    step_label = f"step {s.step}/{s.max_steps}" if s.max_steps else f"step {s.step}"
    in_tok = f"{s.step_in:,}" if s.step_in else "—"
    out_tok = f"{s.step_out:,}" if s.step_out else "—"
    lines.append(
        f"[bold cyan]│[/bold cyan]  [bold blue]AGENT[/bold blue]  ── {step_label} ──▶"
    )
    lines.append(
        f"[bold cyan]│[/bold cyan]        [dim]in: {in_tok} tok · out: {out_tok} tok[/dim]"
    )

    # LLM
    lines.append("[bold cyan]│[/bold cyan]        [dim]│[/dim]")
    lines.append("[bold cyan]│[/bold cyan]        [dim]▼[/dim]")
    lines.append(
        f"[bold cyan]│[/bold cyan]  [bold magenta]LLM[/bold magenta]  [dim]{_trunc(s.model or 'model?', 40)}[/dim]"
    )
    base = _trunc(s.base_url.replace("https://", "").replace("http://", ""), 60)
    if base:
        lines.append(f"[bold cyan]│[/bold cyan]        [dim]↳ {base}[/dim]")

    # streaming text (if any)
    if s.streaming_text:
        lines.append(f"[bold cyan]│[/bold cyan]        [dim]streaming:[/dim] \"{_trunc(s.streaming_text, 56)}\"")

    if s.last_assistant and not s.streaming_text:
        lines.append(
            f"[bold cyan]│[/bold cyan]        [dim]assistant:[/dim] \"{_trunc(s.last_assistant, 56)}\""
        )

    # LLM -> TOOLS
    if s.pending_tools or s.finished_tools:
        lines.append("[bold cyan]│[/bold cyan]        [dim]│[/dim]")
        lines.append("[bold cyan]│[/bold cyan]        [dim]▼[/dim]")
        if s.pending_tools:
            tools_str = ", ".join(s.pending_tools[:5])
            more = f" +{len(s.pending_tools) - 5}" if len(s.pending_tools) > 5 else ""
            lines.append(
                f"[bold cyan]│[/bold cyan]  [bold yellow]TOOLS[/bold yellow] ── ⚙ {tools_str}{more} ──▶"
            )
        elif s.finished_tools:
            tools_str = ", ".join(s.finished_tools[:5])
            more = f" +{len(s.finished_tools) - 5}" if len(s.finished_tools) > 5 else ""
            lines.append(
                f"[bold cyan]│[/bold cyan]  [bold yellow]TOOLS[/bold yellow] ── ✓ {tools_str}{more}[/bold yellow]"
            )

    # TOOLS -> USER (final)
    if s.final_answer:
        lines.append("[bold cyan]│[/bold cyan]        [dim]│[/dim]")
        lines.append("[bold cyan]│[/bold cyan]        [dim]▼[/dim]")
        lines.append(
            f"[bold cyan]│[/bold cyan]  [bold green]USER[/bold green]  ◀── \"{_trunc(s.final_answer, 60)}\""
        )

    if s.error:
        lines.append("[bold cyan]│[/bold cyan]")
        lines.append(f"[bold cyan]│[/bold cyan]  [bold red]✗ {s.error}[/bold red]")

    # Footer stats
    lines.append(f"[bold cyan]│[/bold cyan]")
    pct = int((s.budget_used / s.budget_limit) * 100) if s.budget_limit else 0
    ext = f" · ext:{s.externalized}" if s.externalized else ""
    lines.append(
        f"[bold cyan]│[/bold cyan]  [dim]Σ in: {s.total_in:,} (cached {s.total_cached:,})"
        f"  Σ out: {s.total_out:,}{ext}[/dim]"
    )
    bar_len = 30
    filled = int(bar_len * pct / 100) if s.budget_limit else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    bar_color = "green" if pct < 60 else ("yellow" if pct < 85 else "red")
    lines.append(
        f"[bold cyan]│[/bold cyan]  [dim]budget:[/dim] [{bar_color}]{bar}[/{bar_color}] "
        f"[dim]{s.budget_used:,}/{s.budget_limit:,} ({pct}%)[/dim]"
    )

    lines.append(f"[bold cyan]└{'─' * box_w}[/bold cyan]")

    return "\n".join(lines)
