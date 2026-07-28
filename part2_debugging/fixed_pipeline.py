"""
Part 2 -- Fixed Pipeline
=========================
All three bugs from broken_pipeline.py corrected.

FIX A -- Remove mutable default argument; use an explicit cache object per
         pipeline run (or inject a cache instance for testability).

FIX B -- Wrap the LLM call with a configurable timeout (threading.Timer pattern
         or asyncio.wait_for).  Add exponential-backoff retry via tenacity.

FIX C -- Replace the brittle exact-match regex with a robust number-extraction
         approach.  Fall back to a safe default if no number is found.
         Log a warning -- never silently return None.
"""

import re
import json
import time
import random
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  FIX A -- Explicit cache, no mutable default
# ---------------------------------------------------------------------------


class UserContextCache:
    """Simple TTL-aware in-process cache (replace with Redis in production)."""

    def __init__(self, ttl_seconds: int = 300):
        self._store: dict[str, tuple[dict, float]] = {}
        self._ttl = ttl_seconds

    def get(self, user_id: str) -> Optional[dict]:
        if user_id in self._store:
            value, expires_at = self._store[user_id]
            if time.time() < expires_at:
                return value
            del self._store[user_id]
        return None

    def set(self, user_id: str, value: dict) -> None:
        self._store[user_id] = (value, time.time() + self._ttl)


_context_cache = UserContextCache()


def fetch_user_context(
    user_id: str, cache: Optional[UserContextCache] = None
) -> dict:
    """
    FIX A: Cache is an injected dependency, not a shared mutable default.
    Each caller can pass its own cache, or use the module-level singleton.
    """
    c = cache or _context_cache
    cached = c.get(user_id)
    if cached:
        logger.debug("Cache hit for user_id=%s", user_id)
        return cached

    logger.info("Fetching context for user_id=%s", user_id)
    context = {
        "user_id": user_id,
        "name": f"User_{user_id}",
        "company": f"Acme_{random.randint(1, 5)}",
        "revenue": random.randint(10_000, 10_000_000),
        "industry": random.choice(["SaaS", "FinTech", "Healthcare", "Retail"]),
    }
    c.set(user_id, context)
    return context


# ---------------------------------------------------------------------------
#  FIX B -- LLM call with timeout + exponential-backoff retry
# ---------------------------------------------------------------------------

LLM_TIMEOUT_SECONDS = 10
LLM_MAX_RETRIES = 3


def _mock_llm_call(prompt: str) -> str:
    """Same mock as broken_pipeline but we now handle it with a timeout."""
    if random.random() < 0.2:
        logger.warning("Simulating slow LLM response...")
        time.sleep(30)
    return f"Summary of: {prompt[:80]}"


def _call_with_timeout(prompt: str, timeout: int = LLM_TIMEOUT_SECONDS) -> str:
    """
    FIX B: Run the LLM call on a daemon thread; join with a timeout.
    Raises TimeoutError if the call does not complete in time.
    """
    result_box: list[Optional[str]] = [None]
    error_box: list[Optional[Exception]] = [None]

    def target():
        try:
            result_box[0] = _mock_llm_call(prompt)
        except Exception as exc:
            error_box[0] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutError(f"LLM call did not complete within {timeout}s")
    if error_box[0]:
        raise error_box[0]  # type: ignore[misc]
    return result_box[0]  # type: ignore[return-value]


def generate_summary(context: dict, retries: int = LLM_MAX_RETRIES) -> str:
    """
    FIX B: Timeout + simple exponential-backoff retry loop.
    (In production, use tenacity.retry with wait_exponential.)
    """
    prompt = f"Summarise this lead in 2 sentences: {json.dumps(context)}"
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            return _call_with_timeout(prompt, timeout=LLM_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            last_error = exc
            wait = 2**attempt
            logger.warning(
                "LLM timeout on attempt %d/%d; retrying in %ds",
                attempt,
                retries,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"generate_summary failed after {retries} retries"
    ) from last_error


# ---------------------------------------------------------------------------
#  FIX C -- Robust score extraction, explicit fallback, no silent None
# ---------------------------------------------------------------------------

_SCORE_PATTERN = re.compile(r"\b([0-9]{1,3})\b")


def _mock_llm_score(prompt: str) -> str:
    score = random.randint(10, 95)
    if random.random() < 0.4:
        return (
            f"Based on the provided information, I would score this lead: {score}/100."
        )
    return f"Score: {score}"


def _extract_score(response: str) -> Optional[int]:
    """
    FIX C: Find the first number in [0, 100].
    More robust than an exact-string regex.
    """
    for match in _SCORE_PATTERN.finditer(response):
        candidate = int(match.group(1))
        if 0 <= candidate <= 100:
            return candidate
    return None


def score_lead(summary: str, context: dict) -> int:
    """
    FIX C: Extract score robustly; if still not found, log a warning and
    return a safe default (50) rather than None.  Never silently fail.
    """
    prompt = f"Score this lead 0-100. Respond with ONLY a number. Context: {summary}"
    response = _mock_llm_score(prompt)
    logger.debug("LLM score response: %s", response)

    score = _extract_score(response)
    if score is None:
        logger.warning(
            "Could not parse score from response=%r; defaulting to 50", response
        )
        score = 50  # safe default -- never stored as None

    return score


# ---------------------------------------------------------------------------
#  Pipeline orchestrator
# ---------------------------------------------------------------------------


def persist_result(context: dict, summary: str, score: int) -> dict:
    record = {
        "user_id": context["user_id"],
        "company": context["company"],
        "summary": summary,
        "score": score,
        "scored_at": time.time(),
    }
    logger.info("Persisted record: %s", record)
    return record


def run_pipeline(user_id: str) -> dict:
    """Fixed pipeline -- all three bugs resolved."""
    logger.info("Pipeline start for user_id=%s", user_id)
    context = fetch_user_context(user_id)
    summary = generate_summary(context)
    score = score_lead(summary, context)
    result = persist_result(context, summary, score)
    logger.info("Pipeline complete: score=%d", score)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Running FIXED pipeline...")
    for uid in ["user_001", "user_002", "user_001"]:
        result = run_pipeline(uid)
        print(f"Result: {result}\n")
