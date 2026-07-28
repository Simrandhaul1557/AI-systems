"""
Part 2 — The Broken Pipeline
=============================
A multi-step agent workflow with THREE intentional bugs that cause:
  * Intermittent timeouts
  * Malformed JSON output
  * Silent wrong-data failures

This is the "before" state.  See fixed_pipeline.py for the repaired version
and debugging_walkthrough.py for the step-by-step diagnosis.

Pipeline steps:
  1. fetch_user_context  -- retrieve user profile from a (mock) API
  2. generate_summary    -- call LLM to summarise the profile
  3. score_lead          -- call LLM to score the lead 0-100
  4. persist_result      -- write scored lead to a (mock) DB

Bugs introduced:
  BUG A -- fetch_user_context has a race condition: a shared mutable default
           argument accumulates state across calls, causing wrong data to bleed
           between requests (silent wrong-data failure).

  BUG B -- generate_summary has no timeout on the HTTP call and no retry logic,
           so a slow upstream causes the whole pipeline to hang (intermittent
           timeout).

  BUG C -- score_lead parses the LLM response with a brittle regex that breaks
           if the model includes any preamble text, returning None and causing
           the downstream persist step to store null scores (malformed output).
"""

import re
import time
import json
import random
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BUG A -- mutable default argument (shared state across calls)
# ---------------------------------------------------------------------------


def fetch_user_context(user_id: str, _cache: dict = {}) -> dict:  # noqa: B006
    """
    Fetch user context from upstream API.

    BUG: `_cache` is a mutable default argument.  Python creates it ONCE at
    function definition time, so it is shared across ALL calls.  When the same
    function is called concurrently or sequentially, previously cached data
    from other users can be returned for a new user_id if the key happened to
    collide or if the cache was never cleared.  In practice this means query N
    can silently receive user data from query N-1.
    """
    if user_id in _cache:
        logger.debug("Cache hit for user_id=%s", user_id)
        return _cache[user_id]

    logger.info("Fetching context for user_id=%s", user_id)
    context = {
        "user_id": user_id,
        "name": f"User_{user_id}",
        "company": f"Acme_{random.randint(1, 5)}",
        "revenue": random.randint(10_000, 10_000_000),
        "industry": random.choice(["SaaS", "FinTech", "Healthcare", "Retail"]),
    }
    _cache[user_id] = context  # BUG: mutates shared default dict forever
    return context


# ---------------------------------------------------------------------------
# BUG B -- no timeout, no retry (intermittent hang)
# ---------------------------------------------------------------------------


def _mock_llm_call(prompt: str) -> str:
    """
    Simulates an LLM API call.  Randomly introduces a 30-second delay 20% of
    the time to simulate a slow/unresponsive upstream.
    """
    if random.random() < 0.2:
        logger.warning("Simulating slow LLM response (30s delay)...")
        time.sleep(30)  # This will hang the whole pipeline with no timeout
    return f"Summary of: {prompt[:80]}"


def generate_summary(context: dict) -> str:
    """
    Summarise the user context via LLM.

    BUG: No timeout is set on the _mock_llm_call.  If the upstream is slow,
    this blocks indefinitely -- causing the whole pipeline to hang and the
    caller to receive a timeout error only if their own HTTP server times out
    first (usually resulting in a confusing 504 rather than a clear error).
    """
    prompt = f"Summarise this lead in 2 sentences: {json.dumps(context)}"
    summary = _mock_llm_call(prompt)  # BUG: no timeout parameter
    return summary


# ---------------------------------------------------------------------------
# BUG C -- brittle regex, silent None on parse failure
# ---------------------------------------------------------------------------


def _mock_llm_score(prompt: str) -> str:
    """
    Returns an LLM-style response.  Occasionally wraps the score in preamble
    text (as real LLMs do) to trigger the parsing bug.
    """
    score = random.randint(10, 95)
    if random.random() < 0.4:
        # Realistic LLM response WITH preamble -- breaks the brittle regex
        return f"Based on the provided information, I would score this lead: {score}/100."
    # Clean response -- the regex happens to work
    return f"Score: {score}"


def score_lead(summary: str, context: dict) -> int | None:
    """
    Score the lead 0-100 via LLM.

    BUG: The regex r"Score:\\s*(\\d+)" only matches if the LLM says exactly
    "Score: <number>".  Any variation (preamble, different wording, the number
    appearing mid-sentence) returns None.  The None then propagates silently to
    the database as a null score -- no exception, no log warning.
    """
    prompt = f"Score this lead 0-100 based on: {summary}. Context: {context}"
    response = _mock_llm_score(prompt)
    logger.debug("LLM score response: %s", response)

    match = re.search(r"Score:\s*(\d+)", response)  # BUG: brittle pattern
    if match:
        return int(match.group(1))
    # BUG: returns None silently -- no warning logged, no fallback
    return None


# ---------------------------------------------------------------------------
# Persist step (no bugs here -- just writes whatever it receives)
# ---------------------------------------------------------------------------


def persist_result(context: dict, summary: str, score: int | None) -> dict:
    """Write the scored lead to the (mock) database."""
    record = {
        "user_id": context["user_id"],
        "company": context["company"],
        "summary": summary,
        "score": score,  # Will be None when BUG C triggers
        "scored_at": time.time(),
    }
    logger.info("Persisted record: %s", record)
    return record


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(user_id: str) -> dict:
    """Run all four steps in sequence."""
    logger.info("Pipeline start for user_id=%s", user_id)

    context = fetch_user_context(user_id)
    summary = generate_summary(context)
    score = score_lead(summary, context)
    result = persist_result(context, summary, score)

    logger.info("Pipeline complete: score=%s", score)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    print("Running broken pipeline (may hang 20% of the time)...")
    for uid in ["user_001", "user_002", "user_001"]:  # note repeated user_001
        result = run_pipeline(uid)
        print(f"Result: {result}\n")
