"""
Tests for Part 2 — Pipeline bugs (regression tests that were written BEFORE fixing)

Each test was written against the broken behaviour, run to confirm it was red,
then the fix was applied and the test went green.  This is the standard TDD
debugging loop.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import re
import time
import pytest

# ── Fixed pipeline imports ──────────────────────────────────────────────────
from part2_debugging.fixed_pipeline import (
    UserContextCache,
    fetch_user_context,
    score_lead,
    _extract_score,
    run_pipeline,
)


# ── BUG A regression ────────────────────────────────────────────────────────

class TestCacheIsolation:
    """FIX A: Each call must get its own user's data; no cross-call bleed."""

    def test_separate_users_get_separate_data(self):
        """Two different user_ids must return different records."""
        cache = UserContextCache()
        ctx_a = fetch_user_context("user_test_A", cache=cache)
        ctx_b = fetch_user_context("user_test_B", cache=cache)
        assert ctx_a["user_id"] == "user_test_A"
        assert ctx_b["user_id"] == "user_test_B"
        assert ctx_a != ctx_b

    def test_cache_ttl_expiry(self):
        """After TTL expires the cache should return a fresh fetch."""
        cache = UserContextCache(ttl_seconds=1)
        ctx1 = fetch_user_context("user_ttl", cache=cache)
        time.sleep(1.1)
        # Force a different result by checking the cache no longer holds the entry
        cached = cache.get("user_ttl")
        assert cached is None, "Cache should have expired after TTL"

    def test_repeated_calls_same_user_return_cached_value(self):
        """Same user_id within TTL window should return identical dict."""
        cache = UserContextCache(ttl_seconds=60)
        ctx1 = fetch_user_context("user_repeat", cache=cache)
        ctx2 = fetch_user_context("user_repeat", cache=cache)
        assert ctx1 == ctx2


# ── BUG C regression ────────────────────────────────────────────────────────

class TestScoreExtraction:
    """FIX C: Score extraction must never return None."""

    @pytest.mark.parametrize("response,expected", [
        ("Score: 72",                                              72),
        ("Based on the information, I'd score this lead: 85/100.", 85),
        ("I give this a 60 out of 100.",                           60),
        ("The lead scores approximately 45 points.",               45),
        ("Score: 100",                                            100),
        ("Score: 0",                                               0),
    ])
    def test_extract_score_parses_varied_formats(self, response: str, expected: int):
        result = _extract_score(response)
        assert result == expected, f"Failed to parse '{response}': got {result}"

    def test_score_lead_never_returns_none(self):
        """score_lead() must always return an int, not None."""
        # Run many times to hit both code paths in the mock
        for _ in range(50):
            score = score_lead("Some summary", {"user_id": "x", "company": "y"})
            assert score is not None, "score_lead() returned None"
            assert isinstance(score, int)
            assert 0 <= score <= 100

    def test_score_lead_falls_back_to_50_on_garbage_input(self, monkeypatch):
        """If LLM returns complete garbage, fall back to 50 (not None)."""
        import part2_debugging.fixed_pipeline as fp
        monkeypatch.setattr(fp, "_mock_llm_score", lambda p: "no numbers here at all!!")
        score = fp.score_lead("summary", {"user_id": "u"})
        assert score == 50


# ── End-to-end fixed pipeline ───────────────────────────────────────────────

class TestFixedPipeline:
    def test_pipeline_returns_non_null_score(self):
        """The full fixed pipeline must always return a non-null integer score."""
        result = run_pipeline("e2e_user_001")
        assert result["score"] is not None
        assert isinstance(result["score"], int)

    def test_pipeline_result_has_required_keys(self):
        result = run_pipeline("e2e_user_002")
        for key in ("user_id", "company", "summary", "score", "scored_at"):
            assert key in result, f"Missing key: {key}"
