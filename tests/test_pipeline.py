"""
Tests for Part 2 -- Pipeline bugs (regression tests written BEFORE fixing)

Each test was written against the broken behaviour, run to confirm it was red,
then the fix was applied and the test went green.  Standard TDD debugging loop.
"""

import time
import pytest

from part2_debugging.fixed_pipeline import (
    UserContextCache,
    fetch_user_context,
    score_lead,
    _extract_score,
    run_pipeline,
)


class TestCacheIsolation:
    """FIX A: Each call must get its own user data; no cross-call bleed."""

    def test_separate_users_get_separate_data(self):
        cache = UserContextCache()
        ctx_a = fetch_user_context("user_test_A", cache=cache)
        ctx_b = fetch_user_context("user_test_B", cache=cache)
        assert ctx_a["user_id"] == "user_test_A"
        assert ctx_b["user_id"] == "user_test_B"
        assert ctx_a != ctx_b

    def test_cache_ttl_expiry(self):
        cache = UserContextCache(ttl_seconds=1)
        fetch_user_context("user_ttl", cache=cache)
        time.sleep(1.1)
        cached = cache.get("user_ttl")
        assert cached is None, "Cache should have expired after TTL"

    def test_repeated_calls_same_user_return_cached_value(self):
        cache = UserContextCache(ttl_seconds=60)
        ctx1 = fetch_user_context("user_repeat", cache=cache)
        ctx2 = fetch_user_context("user_repeat", cache=cache)
        assert ctx1 == ctx2


class TestScoreExtraction:
    """FIX C: Score extraction must never return None."""

    @pytest.mark.parametrize(
        "response,expected",
        [
            ("Score: 72", 72),
            ("Based on the information, I'd score this lead: 85/100.", 85),
            ("I give this a 60 out of 100.", 60),
            ("The lead scores approximately 45 points.", 45),
            ("Score: 100", 100),
            ("Score: 0", 0),
        ],
    )
    def test_extract_score_parses_varied_formats(self, response: str, expected: int):
        result = _extract_score(response)
        assert result == expected, f"Failed to parse '{response}': got {result}"

    def test_score_lead_never_returns_none(self):
        for _ in range(50):
            score = score_lead("Some summary", {"user_id": "x", "company": "y"})
            assert score is not None, "score_lead() returned None"
            assert isinstance(score, int)
            assert 0 <= score <= 100

    def test_score_lead_falls_back_to_50_on_garbage_input(self, monkeypatch):
        import part2_debugging.fixed_pipeline as fp

        monkeypatch.setattr(fp, "_mock_llm_score", lambda p: "no numbers here at all!!")
        score = fp.score_lead("summary", {"user_id": "u"})
        assert score == 50


class TestFixedPipeline:
    def test_pipeline_returns_non_null_score(self, monkeypatch):
        """Patch out the slow-LLM mock so the test is deterministic and fast."""
        import part2_debugging.fixed_pipeline as fp

        monkeypatch.setattr(fp, "_mock_llm_call", lambda p: f"Summary of: {p[:80]}")
        result = run_pipeline("e2e_user_001")
        assert result["score"] is not None
        assert isinstance(result["score"], int)

    def test_pipeline_result_has_required_keys(self, monkeypatch):
        import part2_debugging.fixed_pipeline as fp

        monkeypatch.setattr(fp, "_mock_llm_call", lambda p: f"Summary of: {p[:80]}")
        result = run_pipeline("e2e_user_002")
        for key in ("user_id", "company", "summary", "score", "scored_at"):
            assert key in result, f"Missing key: {key}"
