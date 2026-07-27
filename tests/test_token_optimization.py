"""
Tests for Part 1 — Token Optimization
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from part1_token_optimization.baseline_agent import build_naive_prompt, count_tokens as count_baseline
from part1_token_optimization.optimized_agent import OptimizedAgent, count_tokens as count_optimized

SAMPLE_QUERY = "What is the expense reimbursement limit for client dinners?"


def test_baseline_token_count_is_high():
    """Baseline should be well above 50K tokens due to full corpus injection."""
    msgs = build_naive_prompt(SAMPLE_QUERY)
    tokens = count_baseline(msgs)
    assert tokens > 50_000, f"Expected baseline > 50K tokens, got {tokens}"


def test_optimized_token_count_is_low():
    """Optimized should be under 1 000 tokens for the same query."""
    agent = OptimizedAgent()
    msgs = agent.build_messages(SAMPLE_QUERY)
    tokens = count_optimized(msgs)
    assert tokens < 1_000, f"Expected optimized < 1K tokens, got {tokens}"


def test_optimization_achieves_90_percent_reduction():
    """Verify the optimized pipeline cuts tokens by at least 90%."""
    baseline_msgs = build_naive_prompt(SAMPLE_QUERY)
    baseline_tokens = count_baseline(baseline_msgs)

    agent = OptimizedAgent()
    opt_msgs = agent.build_messages(SAMPLE_QUERY)
    opt_tokens = count_optimized(opt_msgs)

    reduction = 1 - (opt_tokens / baseline_tokens)
    assert reduction >= 0.90, (
        f"Expected >= 90% reduction, got {reduction:.1%} "
        f"(baseline={baseline_tokens}, optimized={opt_tokens})"
    )


def test_optimized_messages_contain_user_query():
    """The user's query must appear verbatim in the optimized messages."""
    agent = OptimizedAgent()
    msgs = agent.build_messages(SAMPLE_QUERY)
    last_msg = msgs[-1]
    assert last_msg["role"] == "user"
    assert SAMPLE_QUERY in last_msg["content"]


def test_optimized_messages_contain_retrieved_context():
    """System prompt must include the RAG-retrieved chunks."""
    agent = OptimizedAgent()
    msgs = agent.build_messages(SAMPLE_QUERY)
    system_content = msgs[0]["content"]
    assert "Expense Reimbursement" in system_content or "expense" in system_content.lower()
