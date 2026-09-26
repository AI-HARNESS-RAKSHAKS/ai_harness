"""Visual data-flow panel for the TUI.

Renders the live pipeline showing:
- Where data is going (USER -> AGENT -> LLM -> TOOLS -> ...)
- How much data is flowing at each step (chars, tokens, call count)
- Detailed metrics: cost, context usage, speed, dedup, sub-agents

Designed as a narrow sidebar (default 46 cols) so the main chat log
stays uncluttered. All truncation widths and bar lengths adapt to the
panel width.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from textual.widgets import Static


# Sidebar width. The FlowPanel renders content to fit this many columns.
DEFAULT_SIDEBAR_WIDTH = 46


@dataclass
class FlowState:
    """Mutable state used to render the FlowPanel."""

    # Identity
    model: str = ""
    base_url: str = ""
    context_window: int = 0
    sidebar_width: int = DEFAULT_SIDEBAR_WIDTH

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
    last_ttft: float = 0.0
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


def _trunc(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _bar(fraction: float, length: int = 12) -> tuple[str, str]:
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
    return f"${amount:.5f}"


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

    Renders compact output suitable for a 46-column sidebar.
    """

    def __init__(self, state: FlowState, **kwargs: Any) -> None:
        super().__init__(id="flow", markup=True, **kwargs)
        self.state = state

    def render(self) -> str:
        return _render_flow(self.state)


def _render_flow(s: FlowState) -> str:
    """Render the FlowState as rich text. Pure function for testability."""
    # Sidebar layout - compact, vertical.
    w = max(28, s.sidebar_width)            # total width including borders
    inner = w - 2                            # content width inside │ │
    bar_len = max(8, inner - 18)             # bar length (leave room for numbers)
    tr = max(12, inner - 8)                  # truncation width for flowing text

    sep_top = "┌" + "─" * (w - 2) + "┐"
    sep_bot = "└" + "─" * (w - 2) + "┘"

    def row(content: str) -> str:
        return f"[bold cyan]│[/bold cyan] {content}"

    def blank() -> str:
        return "[bold cyan]│[/bold cyan]"

    lines: list[str] = [
        f"[bold cyan]{sep_top}[/bold cyan]",
        row("[bold]DATA FLOW[/bold]"),
    ]

    # ── Pipeline ────────────────────────────────────────────────────────
    if s.user_input:
        lines.append(blank())
        lines.append(row(
            f"[bold green]USER[/bold green] ── \"{_trunc(s.user_input, tr - 8)}\""
        ))
        lines.append(row(f"[dim]  {len(s.user_input)} chars in[/dim]"))

    lines.append(blank())
    step_label = f"step {s.step}/{s.max_steps}" if s.max_steps else f"step {s.step}"

    if s.is_thinking:
        spinner = "◐◓◑◒"[int(s.elapsed * 4) % 4] if s.elapsed else "◐"
        lines.append(row(
            f"[bold blue]AGENT[/bold blue] {step_label} [yellow]{spinner}[/yellow]"
        ))
        lines.append(row(f"[dim]  thinking · {_fmt_time(s.elapsed)}[/dim]"))
    else:
        lines.append(row(f"[bold blue]AGENT[/bold blue] {step_label}"))
        in_tok = f"{s.step_in:,}" if s.step_in else "—"
        out_tok = f"{s.step_out:,}" if s.step_out else "—"
        cached_tok = f"/c{s.step_cached:,}" if s.step_cached else ""
        ttft_str = f" · ttft {_fmt_time(s.last_ttft)}" if s.last_ttft else ""
        lines.append(row(
            f"[dim]  in:{in_tok}{cached_tok} out:{out_tok}[/dim]"
        ))
        lines.append(row(
            f"[dim]  {_fmt_time(s.last_step_duration)}{ttft_str}[/dim]"
        ))

    lines.append(row("[dim]  │[/dim]"))
    lines.append(row("[dim]  ▼[/dim]"))
    lines.append(row(f"[bold magenta]LLM[/bold magenta] [dim]{_trunc(s.model or '?', tr)}[/dim]"))

    if s.is_streaming and s.streaming_text:
        lines.append(row(f"[green]▍ {_trunc(s.streaming_text, tr)}[/green]"))
    elif s.is_thinking:
        lines.append(row("[yellow]▍ generating...[/yellow]"))
    elif s.last_assistant:
        lines.append(row(f"[dim]▍ {_trunc(s.last_assistant, tr)}[/dim]"))

    if s.pending_tools:
        tools_str = ", ".join(s.pending_tools[:3])
        more = f"+{len(s.pending_tools) - 3}" if len(s.pending_tools) > 3 else ""
        lines.append(row("[dim]  │[/dim]"))
        lines.append(row("[dim]  ▼[/dim]"))
        lines.append(row(
            f"[bold yellow]TOOLS[/bold yellow] [dim]⚙ {_trunc(tools_str + more, tr - 8)}[/dim]"
        ))
    elif s.finished_tools:
        tools_str = ", ".join(s.finished_tools[:3])
        more = f"+{len(s.finished_tools) - 3}" if len(s.finished_tools) > 3 else ""
        lines.append(row("[dim]  │[/dim]"))
        lines.append(row("[dim]  ▼[/dim]"))
        lines.append(row(
            f"[bold yellow]TOOLS[/bold yellow] [dim]✓ {_trunc(tools_str + more, tr - 8)}[/dim]"
        ))

    if s.final_answer:
        lines.append(row("[dim]  │[/dim]"))
        lines.append(row("[dim]  ▼[/dim]"))
        lines.append(row(f"[bold green]USER[/bold green] ◀ \"{_trunc(s.final_answer, tr - 8)}\""))

    if s.error:
        lines.append(blank())
        lines.append(row(f"[bold red]✗ {s.error[:inner - 2]}[/bold red]"))

    # ── Detail ──────────────────────────────────────────────────────────
    lines.append(blank())
    lines.append(row("[bold]── DETAIL ──[/bold]"))

    # Tokens row
    cached_pct = (s.total_cached / s.total_in * 100) if s.total_in else 0
    cache_color = "green" if cached_pct > 30 else ("yellow" if cached_pct > 5 else "dim")
    lines.append(row(
        f"[dim]tok[/dim] {_fmt_tokens(s.total_in)} "
        f"[{cache_color}]({cached_pct:.0f}%↻)[/{cache_color}]"
    ))
    lines.append(row(
        f"[dim]  out:[/dim] {_fmt_tokens(s.total_out)}  "
        f"[dim]calls:[/dim] {s.total_calls}"
    ))

    # Cost row
    lines.append(row(
        f"[dim]$[/dim]  {_fmt_dollars(s.cost_total)} "
        f"[dim](in {_fmt_dollars(s.cost_in)} + out {_fmt_dollars(s.cost_out)})[/dim]"
    ))

    # Context row
    if s.context_window:
        ctx_pct = s.context_used / s.context_window if s.context_window else 0
        ctx_color, ctx_bar = _bar(ctx_pct, length=bar_len)
        lines.append(row(
            f"[dim]ctx[/dim] {_fmt_tokens(s.context_used)}/{_fmt_tokens(s.context_window)} "
            f"[{ctx_color}]{ctx_bar}[/{ctx_color}]"
        ))
        lines.append(row(f"[dim]    ({ctx_pct * 100:.1f}%)[/dim]"))

    # Budget row
    if s.budget_limit:
        bgt_pct = s.budget_used / s.budget_limit if s.budget_limit else 0
        bgt_color, bgt_bar = _bar(bgt_pct, length=bar_len)
        lines.append(row(
            f"[dim]bgt[/dim] {_fmt_tokens(s.budget_used)}/{_fmt_tokens(s.budget_limit)} "
            f"[{bgt_color}]{bgt_bar}[/{bgt_color}]"
        ))
        lines.append(row(f"[dim]    ({bgt_pct * 100:.1f}%)[/dim]"))

    # Speed row
    ttft_info = f" ttft:{_fmt_time(s.last_ttft)}" if s.last_ttft else ""
    mode = "stream" if s.is_streaming else "batch"
    lines.append(row(
        f"[dim]spd[/dim] {s.avg_tokens_per_second:.0f} tok/s "
        f"[dim]{_fmt_time(s.last_step_duration)}{ttft_info}[/dim]"
    ))
    lines.append(row(
        f"[dim]    {_fmt_time(s.elapsed)} · {mode}[/dim]"
    ))

    # Extras row
    misc = []
    if s.externalized:
        misc.append(f"ext:{s.externalized}")
    if s.dedup_hits:
        misc.append(f"dedup:{s.dedup_hits}")
    if s.subagent_calls:
        misc.append(f"sub:{s.subagent_calls}")
    misc_str = " · ".join(misc) if misc else "—"
    lines.append(row(f"[dim]+[/dim] {misc_str}"))

    lines.append(f"[bold cyan]{sep_bot}[/bold cyan]")

    return "\n".join(lines)
