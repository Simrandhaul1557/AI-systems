"""
Part 1 — Baseline Agent (BEFORE optimization)
=============================================
Simulates a naive agent-based pipeline that burns ~100K input tokens per query.

Problems:
  1. Full document corpus stuffed into every system prompt — no retrieval.
  2. Entire conversation history sent on every turn — no summarization or pruning.
  3. Redundant few-shot examples repeated per call.
  4. No caching: identical sub-queries re-sent to the model each time.

This file exists purely as a reference baseline.  The optimized version is in
optimized_agent.py.  Run token_counter.py to see the before/after numbers.
"""

import tiktoken

# ---------------------------------------------------------------------------
# Simulated "knowledge base" — in the naive pipeline every document is
# stuffed verbatim into the system prompt on every single call.
# ---------------------------------------------------------------------------
FULL_DOCUMENT_CORPUS = """
DOCUMENT 1 — Company Policy Manual (pages 1-120)
=================================================
[... 8 000 words of HR policy, benefits, vacation rules, expense limits ...]

DOCUMENT 2 — Product Specification v3.2 (60 pages)
====================================================
[... 6 000 words of technical specs, API contracts, data models ...]

DOCUMENT 3 — Sales Playbook (45 pages)
=======================================
[... 4 500 words of objection handling, pricing tiers, competitor analysis ...]

DOCUMENT 4 — Engineering Runbook (80 pages)
============================================
[... 8 000 words of incident response, deployment checklists, on-call guides ...]

DOCUMENT 5 — Legal & Compliance Guide (70 pages)
=================================================
[... 7 000 words of regulatory requirements, data handling, GDPR obligations ...]
""" + (
    # Simulate bulk document text: 650 varied prose sections bring the
    # corpus well above 50 000 tokens, matching a real large knowledge base.
    "\n".join(
        "Section {i}: This section covers policy item {i} in detail. "
        "Employees must comply with regulation {i} as outlined in "
        "schedule {s}. Failure to follow guideline {i} may result in "
        "disciplinary action under clause {c}. All records related to "
        "item {i} should be retained for {r} years and submitted to the "
        "compliance team by the end of quarter {q}.".format(
            i=i,
            s=i % 10 + 1,
            c=i % 5 + 1,
            r=i % 7 + 1,
            q=i % 4 + 1,
        )
        for i in range(1, 651)
    )
)

# ---------------------------------------------------------------------------
# Naive few-shot examples — pasted in full on EVERY call, not retrieved.
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """
EXAMPLE 1
User: What is our vacation policy?
Assistant: According to the Company Policy Manual, full-time employees receive
20 days of paid vacation per year...  [200 words]

EXAMPLE 2
User: How do I deploy a hotfix?
Assistant: Per the Engineering Runbook, a hotfix deployment follows these steps...
[300 words]

EXAMPLE 3
User: What is the refund policy for enterprise customers?
Assistant: The Sales Playbook states that enterprise customers are eligible for...
[250 words]
""" * 10  # repeated 10x — simulating copy-paste code smell

# ---------------------------------------------------------------------------
# Grows unbounded: entire chat history resent every turn
# ---------------------------------------------------------------------------
SIMULATED_CONVERSATION_HISTORY = [
    {"role": "user", "content": "Tell me about our Q3 sales targets."},
    {
        "role": "assistant",
        "content": "Q3 targets are outlined in the Sales Playbook... [400 words]",
    },
    {"role": "user", "content": "What about discounts for SMB customers?"},
    {"role": "assistant", "content": "SMB discount tiers are... [350 words]"},
    {
        "role": "user",
        "content": "How does that interact with the GDPR obligations?",
    },
    {
        "role": "assistant",
        "content": "From the Legal Guide, GDPR requires... [500 words]",
    },
    # ... imagine 30 more turns of full verbose responses
] * 15  # simulating a long session


def build_naive_prompt(user_query: str) -> list[dict]:
    """
    Naive approach: shove everything into every request.
    Returns the messages list as it would be sent to the API.
    """
    system_prompt = (
        "You are a helpful enterprise assistant.\n\n"
        "Here is the complete knowledge base you must reference:\n\n"
        f"{FULL_DOCUMENT_CORPUS}\n\n"
        "Here are examples of how to answer questions:\n\n"
        f"{FEW_SHOT_EXAMPLES}\n"
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(SIMULATED_CONVERSATION_HISTORY)
    messages.append({"role": "user", "content": user_query})
    return messages


def count_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """Count tokens for an OpenAI messages payload using tiktoken."""
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        # 4 tokens overhead per message (role + separators)
        total += 4 + len(enc.encode(msg.get("content", "") or ""))
    total += 2  # reply priming
    return total


if __name__ == "__main__":
    query = "What is the expense reimbursement limit for client dinners?"
    msgs = build_naive_prompt(query)
    tokens = count_tokens(msgs)
    print(f"[BASELINE]  Input tokens for sample query: {tokens:,}")
