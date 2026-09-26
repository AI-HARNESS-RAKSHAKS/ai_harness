"""Visual data-flow panel for the TUI.

Renders the live pipeline showing:
- Where data is going (USER -> AGENT -> LLM -> TOOLS -> ...)
- How much data is flowing at each step (chars, tokens, call count)
- Detailed metrics: cost, context usage, speed, dedup, sub-agents
- The active step, model, and budget usage
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from textual.widgets import Static


@dataclass
class FlowState:
    """Mutable state used to render the FlowPanel."""

    # Identity
    model: str = ""
    base_url: str = ""
    context_window: int = 0

    # Per-turn
    user_input: str = ""
    step: int = 0
    max_steps: int = 0

    # Aggregate counters (across the whole turn)
    total_in: int = 0
    total_out: int = 0
    total_cached: int = 0
    total_calls: int = 0
    cost_in: float = 0.0
    cost_out: float = 0.0
    cost_total: float = 0.0
    budget_used: int = 0
    budget_limit: int = 0

    # Per-step state
    step_in: int = 0
    step_out: int = 0
    step_cached: int = 0
    last_step_duration: float = 0.0
    last_ttft: float = 0.0  # time-to-first-token
    avg_tokens_per_second: float = 0.0
    elapsed: float = 0.0
    is_thinking: bool = False
    is_streaming: bool = True

    # Recent activity
    last_assistant: str = ""
    pending_tools: list[str] = field(default_factory=list)
    finished_tools: list[str] = field(default_factory=list)
    final_answer: str = ""
    error: str = ""

    # Streaming buffer
    streaming_text: str = ""

    # Externalised / dedup / sub-agents
    externalized: int = 0
    dedup_hits: int = 0
    subagent_calls: int = 0

    # Context used (last observed prompt_tokens)
    context_used: int = 0

    def reset(self) -> None:
        self.user_input = ""
        self.step = 0
        self.total_in = 0
        self.total_out = 0
        self.total_cached = 0
        self.total_calls = 0
        self.cost_in = 0.0
        self.cost_out = 0.0
        self.cost_total = 0.0
        self.budget_used = 0
        self.step_in = 0
        self.step_out = 0
        self.step_cached = 0
        self.last_step_duration = 0.0
        self.last_ttft = 0.0
        self.avg_tokens_per_second = 0.0
        self.elapsed = 0.0
        self.is_thinking = False
        self.is_streaming = True
        self.last_assistant = ""
        self.pending_tools = []
        self.finished_tools = []
        self.final_answer = ""
        self.error = ""
        self.streaming_text = ""
        self.externalized = 0
        self.dedup_hits = 0
        self.subagent_calls = 0
        self.context_used = 0


def _trunc(s: str, n: int = 36) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _bar(fraction: float, length: int = 24, width: int = 80) -> tuple[str, str]:
    """Render a horizontal bar. Returns (color, bar_string)."""
    pct = max(0.0, min(1.0, fraction))
    filled = int(length * pct)
    bar = "█" * filled + "░" * (length - filled)
    color = "green" if pct < 0.6 else ("yellow" if pct < 0.85 else "red")
    return color, bar


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_dollars(amount: float) -> str:
    if amount >= 1.0:
        return f"${amount:.2f}"
    if amount >= 0.01:
        return f"${amount:.3f}"
    if amount >= 0.0001:
        return f"${amount:.4f}"
    return f"${amount:.6f}"


def _fmt_time(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m{s}s"


class FlowPanel(Static):
    """Live visualisation of the agent's data flow + detailed metrics.

    Reads from a FlowState and re-renders itself on every change. The agent
    loop pushes updates into ``self.state`` from its event handler.
    """

    DEFAULT_CSS = ""

    def __init__(self, state: FlowState, **kwargs: Any) -> None:
        super().__init__(id="flow", markup=True, **kwargs)
        self.state = state

    def render(self) -> str:
        return _render_flow(self.state)


def _render_flow(s: FlowState) -> str:
    """Render the FlowState as rich text. Pure function for testability."""
    box_w = 80
    lines: list[str] = []

    lines.append(f"[bold cyan]┌─ DATA FLOW PIPELINE {'─' * (box_w - 22)}[/bold cyan]")

    # USER -> AGENT
    if s.user_input:
        lines.append("[bold cyan]│[/bold cyan]")
        lines.append(
            f"[bold cyan]│[/bold cyan]  [bold green]USER[/bold green]  ── \"{_trunc(s.user_input, 56)}\" ──▶"
        )
        lines.append(f"[bold cyan]│[/bold cyan]        [dim]{len(s.user_input)} chars in[/dim]")

    # AGENT step
    lines.append("[bold cyan]│[/bold cyan]")
    step_label = f"step {s.step}/{s.max_steps}" if s.max_steps else f"step {s.step}"
    in_tok = f"{s.step_in:,}" if s.step_in else "—"
    out_tok = f"{s.step_out:,}" if s.step_out else "—"
    cached_tok = f" (cached {s.step_cached:,})" if s.step_cached else ""

    # Show thinking spinner or step info
    if s.is_thinking:
        spinner = "◐◓◑◒"[int(s.elapsed * 4) % 4] if s.elapsed else "◐"
        lines.append(
            f"[bold cyan]│[/bold cyan]  [bold blue]AGENT[/bold blue]  ── {step_label} "
            f"[yellow]{spinner} thinking...[/yellow] ──▶"
        )
        lines.append(
            f"[bold cyan]│[/bold cyan]        [dim]waiting for first token · "
            f"{_fmt_time(s.elapsed)}[/dim]"
        )
    else:
        lines.append(
            f"[bold cyan]│[/bold cyan]  [bold blue]AGENT[/bold blue]  ── {step_label} ──▶"
        )
        ttft_str = f" · ttft {_fmt_time(s.last_ttft)}" if s.last_ttft else ""
        lines.append(
            f"[bold cyan]│[/bold cyan]        [dim]in: {in_tok}{cached_tok} tok · out: {out_tok} tok · "
            f"{_fmt_time(s.last_step_duration)}{ttft_str}[/dim]"
        )

    # LLM
    lines.append("[bold cyan]│[/bold cyan]        [dim]│[/dim]")
    lines.append("[bold cyan]│[/bold cyan]        [dim]▼[/dim]")
    lines.append(
        f"[bold cyan]│[/bold cyan]  [bold magenta]LLM[/bold magenta]  [dim]{_trunc(s.model or 'model?', 50)}[/dim]"
    )
    base = _trunc(s.base_url.replace("https://", "").replace("http://", ""), 64)
    if base:
        lines.append(f"[bold cyan]│[/bold cyan]        [dim]↳ {base}[/dim]")

    # streaming text or assistant text or thinking placeholder
    if s.is_streaming and s.streaming_text:
        lines.append(
            f"[bold cyan]│[/bold cyan]        [green]▍ streaming:[/green] "
            f"\"{_trunc(s.streaming_text, 60)}\""
        )
    elif s.is_thinking:
        lines.append(
            f"[bold cyan]│[/bold cyan]        [yellow]▍ generating...[/yellow]"
        )
    elif s.last_assistant:
        lines.append(
            f"[bold cyan]│[/bold cyan]        [dim]assistant:[/dim] \"{_trunc(s.last_assistant, 60)}\""
        )

    # TOOLS
    if s.pending_tools or s.finished_tools:
        lines.append("[bold cyan]│[/bold cyan]        [dim]│[/dim]")
        lines.append("[bold cyan]│[/bold cyan]        [dim]▼[/dim]")
        if s.pending_tools:
            tools_str = ", ".join(s.pending_tools[:6])
            more = f" +{len(s.pending_tools) - 6}" if len(s.pending_tools) > 6 else ""
            lines.append(
                f"[bold cyan]│[/bold cyan]  [bold yellow]TOOLS[/bold yellow] ── ⚙ {tools_str}{more} ──▶"
            )
        elif s.finished_tools:
            tools_str = ", ".join(s.finished_tools[:6])
            more = f" +{len(s.finished_tools) - 6}" if len(s.finished_tools) > 6 else ""
            lines.append(
                f"[bold cyan]│[/bold cyan]  [bold yellow]TOOLS[/bold yellow] ── ✓ {tools_str}{more}[/bold yellow]"
            )

    # Final answer
    if s.final_answer:
        lines.append("[bold cyan]│[/bold cyan]        [dim]│[/dim]")
        lines.append("[bold cyan]│[/bold cyan]        [dim]▼[/dim]")
        lines.append(
            f"[bold cyan]│[/bold cyan]  [bold green]USER[/bold green]  ◀── \"{_trunc(s.final_answer, 64)}\""
        )

    if s.error:
        lines.append("[bold cyan]│[/bold cyan]")
        lines.append(f"[bold cyan]│[/bold cyan]  [bold red]✗ {s.error}[/bold red]")

    # ─── Detail panel ──────────────────────────────────────────────────────
    lines.append(f"[bold cyan]│[/bold cyan]")
    lines.append(f"[bold cyan]│[/bold cyan]  [bold]─── DETAIL ───[/bold]")

    # tokens row
    cached_pct = (s.total_cached / s.total_in * 100) if s.total_in else 0
    cache_color = "green" if cached_pct > 30 else ("yellow" if cached_pct > 5 else "dim")
    lines.append(
        f"[bold cyan]│[/bold cyan]  [dim]tokens   Σ in:[/dim] {_fmt_tokens(s.total_in)} "
        f"[{cache_color}](cached: {_fmt_tokens(s.total_cached)} = {cached_pct:.0f}%)[/{cache_color}] "
        f"[dim]| Σ out:[/dim] {_fmt_tokens(s.total_out)} "
        f"[dim]| calls:[/dim] {s.total_calls}"
    )

    # cost row
    lines.append(
        f"[bold cyan]│[/bold cyan]  [dim]cost     total:[/dim] {_fmt_dollars(s.cost_total)} "
        f"[dim](in {_fmt_dollars(s.cost_in)} + out {_fmt_dollars(s.cost_out)})[/dim]"
    )

    # context row with bar
    if s.context_window:
        ctx_pct = (s.context_used / s.context_window) if s.context_window else 0
        ctx_color, ctx_bar = _bar(ctx_pct, length=24)
        lines.append(
            f"[bold cyan]│[/bold cyan]  [dim]context  ~{_fmt_tokens(s.context_used)} / "
            f"{_fmt_tokens(s.context_window)} ({ctx_pct * 100:.1f}%)[/dim] "
            f"[{ctx_color}]{ctx_bar}[/{ctx_color}] [dim]({_fmt_tokens(s.context_window - s.context_used)} free)[/dim]"
        )
    else:
        lines.append(f"[bold cyan]│[/bold cyan]  [dim]context  unknown (set AI_CONTEXT_WINDOW)[/dim]")

    # budget row with bar
    if s.budget_limit:
        bgt_pct = (s.budget_used / s.budget_limit) if s.budget_limit else 0
        bgt_color, bgt_bar = _bar(bgt_pct, length=24)
        lines.append(
            f"[bold cyan]│[/bold cyan]  [dim]budget   {_fmt_tokens(s.budget_used)} / "
            f"{_fmt_tokens(s.budget_limit)} ({bgt_pct * 100:.1f}%)[/dim] "
            f"[{bgt_color}]{bgt_bar}[/{bgt_color}]"
        )

    # speed row
    ttft_info = f" · ttft: {_fmt_time(s.last_ttft)}" if s.last_ttft else ""
    mode = "streaming" if s.is_streaming else "batch"
    lines.append(
        f"[bold cyan]│[/bold cyan]  [dim]speed    avg: {s.avg_tokens_per_second:.0f} tok/s · "
        f"last step: {_fmt_time(s.last_step_duration)}{ttft_info} · "
        f"elapsed: {_fmt_time(s.elapsed)} · {mode}[/dim]"
    )

    # misc row
    misc = []
    if s.externalized:
        misc.append(f"ext:{s.externalized}")
    if s.dedup_hits:
        misc.append(f"dedup:{s.dedup_hits}")
    if s.subagent_calls:
        misc.append(f"sub-agents:{s.subagent_calls}")
    misc_str = " · ".join(misc) if misc else "(none)"
    lines.append(f"[bold cyan]│[/bold cyan]  [dim]extras   {misc_str}[/dim]")

    lines.append(f"[bold cyan]└{'─' * box_w}[/bold cyan]")

    return "\n".join(lines)
