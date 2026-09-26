# AI Harness

A text-only coding agent that runs inside a Terminal User Interface (TUI). Built for the AI Harness Hackathon 2026.

The agent receives a natural-language task, uses tools (`read_file`, `write_file`, `edit_file`, `list_files`, `bash`, `task`) to act on a working directory, and streams back what it did. It speaks any OpenAI-compatible chat-completions API.

## Quickstart

```bash
export AI_API_KEY="<your-key>"
export AI_BASE_URL="<openai-compatible-endpoint>"
export AI_MODEL="<model-name>"
make setup
make run
```

### Example providers

| Provider | `AI_BASE_URL` | `AI_MODEL` | Notes |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | default |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` | **judges will use this** |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-flash` | cheapest, very fast |
| **Groq** | `https://api.groq.com/openai/v1` | `qwen/qwen3.8-27b` | free tier, judges use this |
| **Qwen** (DashScope) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-turbo` | **judges will use this** |
| **Qwen** (DashScope) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-coder-plus` | best for code |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` | |
| xAI Grok | `https://api.x.ai/v1` | `grok-2-latest` | |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` | free tier |

## Why this is faster & cheaper than DeerFlow / Pi

Built-in techniques borrowed from both projects (DeerFlow = bytedance/deer-flow, Pi = earendil-works/pi), plus a few extras.

### Token optimisations

| Technique | Source | Saves |
|---|---|---|
| Truncated tool outputs (head for `read`, tail for `bash`) | Pi | caps per-result cost |
| Tool output externalisation (>threshold chars → disk + synopsis) | DeerFlow | ~90% on huge outputs |
| Superseded write elision (`write_file` content shrunk if later read/edit of same path) | DeerFlow | ~70% on multi-write flows |
| Token-budget hard-stop (strips tool_calls at the per-run cap) | DeerFlow | prevents runaway cost |
| Conversation history window (drop middle, keep head + tail) | Pi | bounds context growth |
| Output token cap (`max_tokens` per call) | this repo | caps completion cost |
| Static system prompt + terse tool schemas → prompt caching | both | auto-cached prefix |
| Short-window call dedup (read_file / list_files) | this repo | re-reads are free |
| Sub-agent returns trimmed + externalised if huge | this repo | keeps parent context small |

### Speed & reliability

| Technique | Source | Effect |
|---|---|---|
| **Parallel tool execution** (`asyncio.gather` over concurrent calls) | Pi | ~Nx speedup when assistant emits multiple calls |
| Sub-agent tool (`task`) with isolated context, bounded turns, optionally cheaper model | both — Pi doesn't have it | delegate bounded subtasks |
| Two-layer loop detection (identical-call hash + per-tool frequency) | DeerFlow | stops runaway loops |
| **Auto-retry with exponential backoff** for 408/409/425/429/500/502/503/504 | this repo | survives transient failures (Google 503, OpenAI 429) |
| **SSE streaming** for assistant responses | this repo | text appears token-by-token as the LLM generates |
| Retry-After header honoured | this repo | respects provider pacing |

### Recommended model choices

| Provider    | Cheapest capable model           | Set with              |
|-------------|----------------------------------|-----------------------|
| **DeepSeek** | `deepseek-flash` ($0.014/M in) | `AI_MODEL=deepseek-flash` |
| **DeepSeek** | `deepseek-chat` (V3, smart) | `AI_MODEL=deepseek-chat` |
| **Qwen** | `qwen-turbo` ($0.05/M in) | `AI_MODEL=qwen-turbo` |
| **Qwen** | `qwen-coder-plus` (best for code) | `AI_MODEL=qwen-coder-plus` |
| OpenAI      | `gpt-4.1-nano` (~33% cheaper than `gpt-4o-mini`) | `AI_MODEL=gpt-4.1-nano` |
| OpenAI      | `gpt-4o-mini` (default)           | —                     |
| Groq        | `llama-3.1-8b-instant` (free tier) | `AI_MODEL=llama-3.1-8b-instant` `AI_BASE_URL=https://api.groq.com/openai/v1` |
| OpenRouter  | `meta-llama/llama-3.1-8b-instruct` | `AI_BASE_URL=https://openrouter.ai/api/v1` |

Set `AI_SUBAGENT_MODEL=deepseek-flash` (or `qwen-turbo`) to make sub-agents use a cheaper model than the parent.

## Standard evaluation interface

```bash
git clone <repo>
cd <repo>
export AI_API_KEY="<provided-key>"
make setup
make run
```

The Makefile also exposes `make test` (126 tests) and `make clean`.

## Environment variables

| Variable                       | Default                          | Purpose                                  |
|--------------------------------|----------------------------------|------------------------------------------|
| `AI_API_KEY`                   | —                                | Bearer token (required)                  |
| `AI_BASE_URL`                  | `https://api.openai.com/v1`      | OpenAI-compatible base URL               |
| `AI_MODEL`                     | `gpt-4o-mini`                    | Model name                               |
| `AI_MAX_STEPS`                 | `20`                             | Max tool-use iterations per turn         |
| `AI_MAX_OUTPUT_TOKENS`         | `2048`                           | Per-call output token cap                |
| `AI_HISTORY_WINDOW`            | `40`                             | Message count before history is dropped  |
| `AI_MAX_TOOL_CHARS`            | `4000`                           | Truncate tool outputs past this many chars |
| `AI_MAX_TOOL_LINES`            | `200`                            | Truncate tool outputs past this many lines |
| `AI_WORKDIR`                   | `.`                              | Root for file tool paths                 |
| `AI_EXTERNALIZE_THRESHOLD`     | `8000`                           | Outputs larger than this go to disk      |
| `AI_TOKEN_BUDGET`              | `200000`                         | Hard cap on cumulative tokens per run    |
| `AI_LOOP_WINDOW`               | `20`                             | Sliding window for loop detection        |
| `AI_LOOP_IDENTICAL`            | `5`                              | Stop after this many identical calls     |
| `AI_LOOP_FREQ`                 | `50`                             | Stop after this many total calls of one tool |
| `AI_PARALLEL_TOOLS`            | `true`                           | Run concurrent tool calls via `asyncio.gather` |
| `AI_SUBAGENT_MODEL`            | (inherit)                        | Optional cheaper model for sub-agents    |
| `AI_SUBAGENT_MAX_STEPS`        | `15`                             | Max steps for any sub-agent run          |
| `AI_SUBAGENT_DEFAULT_MAX_TURNS`| `10`                             | Default `max_turns` arg of `task` tool   |

No credentials are committed. See `.env.example`.

## Tools available to the agent

| Tool         | Purpose                                                |
|--------------|--------------------------------------------------------|
| `read_file`  | Read a file (head-truncated; huge outputs externalised)|
| `write_file` | Write content to a file (overwrites)                   |
| `edit_file`  | Replace `old_string` with `new_string`                 |
| `list_files` | List a directory (head-truncated)                      |
| `bash`       | Run a shell command (tail-truncated; errors kept)      |
| `task`       | Spawn an isolated sub-agent with the given prompt      |
| `done`       | Signal completion and return a final answer            |

## Project layout

```
.
├── Makefile                # required: setup / run / test / clean
├── README.md
├── requirements.txt
├── .env.example
├── harness/
│   ├── __init__.py
│   ├── __main__.py         # entry: python -m harness
│   ├── config.py           # env loading
│   ├── llm.py              # OpenAI-compatible client + retry + streaming
│   ├── optimize.py         # budget / loops / externalisation / dedup / truncate
│   ├── flow.py             # FlowPanel - visual data-flow widget
│   ├── pricing.py          # per-model pricing + context windows
│   ├── tools.py            # tool implementations + schemas
│   ├── agent.py            # agent loop (parallel exec, sub-agents, elision)
│   └── app.py              # Textual TUI
└── tests/
    ├── test_smoke.py
    ├── test_tools.py
    ├── test_agent.py
    ├── test_optimize.py
    ├── test_flow.py
    ├── test_llm.py
    ├── test_pricing.py
    └── test_providers.py   # end-to-end with DeepSeek + Qwen response shapes
```

## Visual data-flow panel

While the harness runs, a live panel at the top of the TUI shows exactly where data is flowing, how much, and the cost:

```
┌─ DATA FLOW PIPELINE ─────────────────────────────────────────────────────┐
│                                                                         │
│  USER  ── "find the bug in a.js" ──▶                                    │
│        20 chars in                                                      │
│                                                                         │
│  AGENT ── step 4/20 ──▶                                                 │
│        in: 1,850 (cached 1,200) tok · out: 142 tok · 1.8s              │
│        │                                                                │
│        ▼                                                                │
│  LLM   gemini-2.5-flash                                                 │
│        ↳ generativelanguage.googleapis.com/v1beta/openai                │
│        │                                                                │
│        ▼                                                                │
│  TOOLS ── ⚙ read_file, bash ──▶                                        │
│                                                                         │
│  ─── DETAIL ───                                                          │
│  tokens   Σ in: 8.9k (cached: 5.2k = 58%) | Σ out: 612 | calls: 4      │
│  cost     total: $0.0006  (in $0.0004 + out $0.0002)                    │
│  context  ~1.9k / 1.05M (0.2%)  ░░░░░░░░░░░░░░░░░░░░░░░░ (1.05M free)  │
│  budget   9.5k / 200.0k (4.8%)  █░░░░░░░░░░░░░░░░░░░░░░░               │
│  speed    avg: 78 tok/s · last step: 1.8s · elapsed: 7.4s               │
│  extras   ext:1 · dedup:2 · sub-agents:1                               │
└─────────────────────────────────────────────────────────────────────────┘
```

**What the detail section shows:**

| Field | Meaning |
|---|---|
| `tokens` | Total input/output across all calls. Cache % turns **green** when >30%, yellow at 5–30%, dim otherwise |
| `cost` | USD estimate based on per-model pricing (see `harness/pricing.py`). Cached input shown separately |
| `context` | Estimated prompt size vs the model's context window. Bar turns red at 85% |
| `budget` | Cumulative token cap (default 200k). Bar yellow at 60%, red at 85% |
| `speed` | Average tokens/sec, last step duration, total elapsed |
| `extras` | Externalised outputs, dedup hits, sub-agent invocations |

Every `AgentEvent` updates the panel:
- `user` → your input + char count
- `step` → step counter + per-step token spend
- `assistant` → model name + endpoint + per-step timing/cost
- `tool_call` → tool names being executed
- `tool_result` → finished tools (✓)
- `done` → final answer
- `error` → surfaced as ✗ with message

Pricing data covers 25+ models (OpenAI, Gemini, Grok, Groq, Anthropic). Override per-session via env:
```bash
export AI_INPUT_PRICE=0.5     # USD per 1M input tokens
export AI_OUTPUT_PRICE=2.0    # USD per 1M output tokens
export AI_CACHED_PRICE=0.1    # USD per 1M cached input tokens
export AI_CONTEXT_WINDOW=500000
```

## Keyboard shortcuts inside the TUI

- `Enter` — submit the current input
- `Ctrl+L` — clear the log
- `Ctrl+R` — reset the conversation, usage, budget, and cost counters
- `Ctrl+K` — compact history (drop middle messages, keep head + tail)
- `Ctrl+D` — hide / show the FlowPanel
- `Ctrl+C` — quit
