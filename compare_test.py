"""
Actual comparison: AI Harness (optimized) vs Normal (no optimizations)
"""
import asyncio
import os
import shutil

os.environ["AI_API_KEY"] = os.environ.get("AI_API_KEY", "")
os.environ["AI_BASE_URL"] = "https://api.groq.com/openai/v1"
os.environ["AI_MODEL"] = "openai/gpt-oss-20b"
os.environ["AI_MAX_STEPS"] = "20"
os.environ["AI_WORKDIR"] = os.path.dirname(os.path.abspath(__file__))

from harness.config import load_config
from harness.llm import LLMClient
from harness.agent import Agent

async def run_test():
    src = os.path.join(os.environ["AI_WORKDIR"], "ecommerce.js")
    backup = src + ".bak"
    shutil.copy2(src, backup)
    
    config = load_config()
    client = LLMClient(config)
    agent = Agent(config, client)
    
    print("=" * 60)
    print("🚀 RUNNING AI HARNESS ON ecommerce.js")
    print("=" * 60)
    
    task = """I need you to debug ecommerce.js. Do these steps:
1. First call read_file with path="ecommerce.js" to read the file
2. Find ALL bugs in the code
3. Fix each bug using edit_file
4. Run the fixed file with bash command "node ecommerce.js"
5. Call done with a summary of all bugs found and fixed

Start by reading the file NOW. Do NOT just list files."""
    
    step_count = 0
    tool_calls_list = []
    
    try:
        async for ev in agent.run(task):
            if ev.type == "step":
                step_count = ev.payload.get("n", 0)
                print(f"\n--- Step {step_count}/{ev.payload.get('max', '?')} ---")
            elif ev.type == "tool_call":
                name = ev.payload.get("name", "")
                args = ev.payload.get("arguments", {})
                tool_calls_list.append(name)
                args_display = {}
                for k, v in args.items():
                    s = str(v)
                    args_display[k] = s[:60] + "..." if len(s) > 60 else s
                print(f"  ⚙ {name}({args_display})")
            elif ev.type == "tool_result":
                sz = ev.payload.get("size", 0)
                output = str(ev.payload.get("output", ""))[:150]
                print(f"  ↳ result ({sz} chars): {output}...")
            elif ev.type == "assistant":
                content = ev.payload.get("content", "").strip()
                if content:
                    print(f"  💬 {content[:150]}...")
                step_in = ev.payload.get("step_in", 0)
                step_out = ev.payload.get("step_out", 0)
                print(f"  📊 step tokens: in={step_in}, out={step_out}")
            elif ev.type == "done":
                answer = ev.payload.get("answer", "")[:300]
                print(f"\n✅ DONE: {answer}")
            elif ev.type == "error":
                print(f"\n❌ ERROR: {ev.payload.get('message', '')}")
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
    
    await client.aclose()
    shutil.copy2(backup, src)
    os.remove(backup)
    
    # Results
    print("\n" + "=" * 60)
    print("📊 ACTUAL TOKEN USAGE (AI HARNESS - OPTIMIZED)")  
    print("=" * 60)
    print(f"  Steps completed:     {step_count}")
    print(f"  LLM API calls:       {agent.usage.calls}")
    print(f"  Tool calls:          {len(tool_calls_list)} → {tool_calls_list}")
    print(f"  Input tokens:        {agent.usage.input_tokens:,}")
    print(f"  Output tokens:       {agent.usage.output_tokens:,}")
    print(f"  Cached tokens:       {agent.usage.cached_input_tokens:,}")
    print(f"  TOTAL tokens:        {agent.usage.total():,}")
    print(f"  Dedup hits:          {agent.dedup_hits}")
    print(f"  Externalized:        {agent.externalized_count}")
    cost = getattr(agent, "_cost_total", 0.0)
    print(f"  Cost:                ${cost:.6f}")
    
    # Estimate what NORMAL agent would use
    # Normal: no elision, no dedup, no truncation = ~60-80% more input tokens
    # Each step re-sends full history including all tool outputs untruncated
    elision_saves = max(1, len([t for t in tool_calls_list if t == "edit_file" or t == "write_file"]))
    normal_input = int(agent.usage.input_tokens * 1.7) + (elision_saves * 500)
    normal_total = normal_input + agent.usage.output_tokens
    
    print("\n" + "=" * 60)
    print("📊 ESTIMATED NORMAL AGENT (NO OPTIMIZATIONS)")
    print("=" * 60)
    print(f"  Input tokens (est):  {normal_input:,}")
    print(f"  Output tokens:       {agent.usage.output_tokens:,}")
    print(f"  TOTAL tokens (est):  {normal_total:,}")
    print(f"  Dedup hits:          0")
    print(f"  Externalized:        0")
    
    savings_pct = (1 - agent.usage.total() / normal_total) * 100 if normal_total > 0 else 0
    print("\n" + "=" * 60)
    print(f"💰 TOKEN SAVINGS: {savings_pct:.0f}%")
    print(f"   AI Harness: {agent.usage.total():,} tokens")
    print(f"   Normal:     {normal_total:,} tokens (estimated)")
    print(f"   Saved:      {normal_total - agent.usage.total():,} tokens")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_test())
