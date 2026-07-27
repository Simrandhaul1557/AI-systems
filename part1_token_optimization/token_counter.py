"""
Part 1 — Token Counter: Before / After comparison
==================================================
Run this script directly to see the token savings from both optimizations.

  python part1_token_optimization/token_counter.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tiktoken

# ── Import baseline ──────────────────────────────────────────────────────────
from part1_token_optimization.baseline_agent import (
    build_naive_prompt,
    count_tokens as count_baseline,
)

# ── Import optimized ─────────────────────────────────────────────────────────
from part1_token_optimization.optimized_agent import (
    OptimizedAgent,
    count_tokens as count_optimized,
)


SAMPLE_QUERY = "What is the expense reimbursement limit for client dinners?"
MODEL = "gpt-4o"


def separator(char: str = "─", width: int = 60) -> str:
    return char * width


def main() -> None:
    print(separator("═"))
    print("  Part 1 — Token Usage: BEFORE vs AFTER Optimization")
    print(separator("═"))
    print(f"\n  Sample query: \"{SAMPLE_QUERY}\"\n")

    # ── BEFORE ───────────────────────────────────────────────────────────────
    baseline_msgs = build_naive_prompt(SAMPLE_QUERY)
    baseline_tokens = count_baseline(baseline_msgs, MODEL)

    print(separator())
    print("  BEFORE (baseline naive agent)")
    print(separator())
    print(f"  Total input tokens : {baseline_tokens:>10,}")
    print(f"  Breakdown by source:")

    enc = tiktoken.encoding_for_model(MODEL)

    from part1_token_optimization.baseline_agent import (
        FULL_DOCUMENT_CORPUS,
        FEW_SHOT_EXAMPLES,
        SIMULATED_CONVERSATION_HISTORY,
    )

    corpus_tokens = len(enc.encode(FULL_DOCUMENT_CORPUS))
    fewshot_tokens = len(enc.encode(FEW_SHOT_EXAMPLES))
    history_tokens = sum(
        len(enc.encode(m["content"])) for m in SIMULATED_CONVERSATION_HISTORY
    )
    query_tokens = len(enc.encode(SAMPLE_QUERY))
    overhead = baseline_tokens - corpus_tokens - fewshot_tokens - history_tokens - query_tokens

    print(f"    Full document corpus  : {corpus_tokens:>10,} tokens")
    print(f"    Few-shot examples     : {fewshot_tokens:>10,} tokens")
    print(f"    Conversation history  : {history_tokens:>10,} tokens")
    print(f"    Current query         : {query_tokens:>10,} tokens")
    print(f"    Prompt overhead       : {overhead:>10,} tokens")

    # Estimated cost at gpt-4o input pricing ($5 / 1M tokens)
    baseline_cost = (baseline_tokens / 1_000_000) * 5.0
    print(f"\n  Estimated cost (gpt-4o @ $5/1M): ${baseline_cost:.4f} per query")
    print(f"  At 10 000 queries/day           : ${baseline_cost * 10_000:.2f}/day")

    # ── AFTER ────────────────────────────────────────────────────────────────
    agent = OptimizedAgent()
    optimized_msgs = agent.build_messages(SAMPLE_QUERY)
    optimized_tokens = count_optimized(optimized_msgs, MODEL)

    print(f"\n{separator()}")
    print("  AFTER (optimized agent — both optimizations applied)")
    print(separator())
    print(f"  Total input tokens : {optimized_tokens:>10,}")
    print(f"  Breakdown by source:")

    from part1_token_optimization.optimized_agent import (
        RETRIEVED_CHUNKS,
        RETRIEVED_FEW_SHOT,
        ROLLING_SUMMARY,
        RECENT_TURNS,
    )

    rag_tokens = len(enc.encode(RETRIEVED_CHUNKS))
    opt_fewshot_tokens = len(enc.encode(RETRIEVED_FEW_SHOT))
    summary_tokens = len(enc.encode(ROLLING_SUMMARY))
    recent_tokens = sum(len(enc.encode(m["content"])) for m in RECENT_TURNS)

    print(f"    RAG retrieved chunks  : {rag_tokens:>10,} tokens  (was {corpus_tokens:,})")
    print(f"    Few-shot (1 example)  : {opt_fewshot_tokens:>10,} tokens  (was {fewshot_tokens:,})")
    print(f"    Rolling summary       : {summary_tokens:>10,} tokens  (was {history_tokens:,} history)")
    print(f"    Recent turns (last 3) : {recent_tokens:>10,} tokens")
    print(f"    Current query         : {query_tokens:>10,} tokens")

    opt_cost = (optimized_tokens / 1_000_000) * 5.0
    print(f"\n  Estimated cost (gpt-4o @ $5/1M): ${opt_cost:.6f} per query")
    print(f"  At 10 000 queries/day           : ${opt_cost * 10_000:.2f}/day")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    reduction = (1 - optimized_tokens / baseline_tokens) * 100
    cost_saved_daily = (baseline_cost - opt_cost) * 10_000

    print(f"\n{separator('═')}")
    print("  SUMMARY")
    print(separator("═"))
    print(f"  Tokens BEFORE : {baseline_tokens:>10,}")
    print(f"  Tokens AFTER  : {optimized_tokens:>10,}")
    print(f"  Reduction     : {reduction:>9.1f}%")
    print(f"  Daily savings : ${cost_saved_daily:>9.2f}  (at 10K queries/day)")
    print(separator("═"))

    print("""
  QUALITY TRADEOFFS
  ─────────────────
  Optimization 1 (RAG):
    • Quality impact: minimal for single-domain queries.
    • Risk: multi-domain queries may miss a chunk if k is too small.
    • Mitigation: increase k=6 for ambiguous queries; add a topic router.

  Optimization 2 (Sliding-window + rolling summary):
    • Quality impact: model loses verbatim early-turn recall.
    • Risk: nuanced commitments from early turns may be oversimplified.
    • Mitigation: summary is generated by the model itself with a dedicated
      "summarize this conversation accurately" prompt; reviewed before caching.

  Bonus (Prompt caching):
    • Quality impact: zero — identical inputs, server reuses KV cache.
    • Effect: ~75 % cost reduction and ~35 % latency drop on cache hits.
    • No code change needed beyond placing stable content at the prompt head.
""")


if __name__ == "__main__":
    main()
