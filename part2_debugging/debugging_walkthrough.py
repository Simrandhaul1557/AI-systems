"""
Part 2 -- Debugging Walkthrough
================================
Step-by-step process for diagnosing the three bugs in broken_pipeline.py.

This file is both documentation AND runnable:
  python part2_debugging/debugging_walkthrough.py

It demonstrates:
  1. What signals / logs / metrics you'd look at first
  2. How to isolate each bug with targeted tests
  3. The fix applied and why it works

DEBUGGING PROCESS -- NARRATIVE
================================

STEP 1 -- Triage symptoms
--------------------------
Before touching any code, reproduce the failure and characterise it:
  * Is it deterministic or flaky?    -> run the pipeline 20 times, note failure %
  * Which step fails?                -> add per-step timing and step-name logging
  * What does the failure look like? -> collect raw output + tracebacks

Tools used at this stage:
  * Structured JSON logging (step name, duration_ms, user_id in every log line)
  * A simple harness that runs 50 pipeline calls and reports
    {success, timeout, null_score, wrong_user} counts
  * grep / log aggregator (CloudWatch, Datadog) filtered by pipeline_run_id

STEP 2 -- Reproduce locally with determinism
---------------------------------------------
Set random.seed(42) and increase the failure probability to 100% to make
each bug always fire.  This turns a flaky test into a guaranteed failing test.

STEP 3 -- Isolate each step (binary search)
--------------------------------------------
Comment out steps 2-4 and verify step 1 works alone.  Add step 2, verify.
Continue until the failure appears -- that is the broken step.

STEP 4 -- Read the code with fresh eyes
-----------------------------------------
Once the step is identified, read it slowly.  The three categories to scan:
  a) External I/O without timeout or retry
  b) State shared across calls (globals, mutable defaults, class attributes)
  c) Output parsing that assumes the model always returns a specific format

STEP 5 -- Write a failing unit test BEFORE fixing
---------------------------------------------------
A test that fails red -> proves you understand the bug -> fix it -> goes green.
This becomes a regression guard.

STEP 6 -- Fix, verify, deploy with feature flag
-------------------------------------------------
Fix the code, run the full test suite, shadow-deploy to a 1% traffic slice,
watch metrics for 15 minutes, then full rollout.
"""

import re
import logging
import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  DIAGNOSTIC HARNESS  (Step 1 tooling)
# ---------------------------------------------------------------------------


def _with_timeout(fn, args=(), timeout_sec: int = 5):
    """
    Wrap a function call with a SIGALRM-based timeout (Unix only).
    On Windows SIGALRM is not available; returns None on AttributeError.
    """
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(timeout_sec)
        result = fn(*args)
        signal.alarm(0)
        return result
    except (TimeoutError, AttributeError):
        return None


def stress_test(n: int = 30) -> None:
    """
    Run the broken pipeline N times and categorise failures.
    This is the first tool to run after getting a bug report.
    """
    from part2_debugging.broken_pipeline import run_pipeline

    results = {"success": 0, "null_score": 0, "wrong_data": 0, "timeout": 0}

    for i in range(n):
        uid = "user_A" if i % 2 == 0 else "user_B"
        try:
            result = _with_timeout(run_pipeline, args=(uid,), timeout_sec=3)
            if result is None:
                results["timeout"] += 1
            elif result["score"] is None:
                results["null_score"] += 1
            else:
                results["success"] += 1
        except Exception as exc:
            logger.error("Pipeline error on iteration %d: %s", i, exc)

    total = sum(results.values())
    print("\n  -- Stress test results --")
    for k, v in results.items():
        pct = v / total * 100 if total else 0
        print(f"  {k:15s}: {v:3d}  ({pct:4.1f}%)")
    print()


# ---------------------------------------------------------------------------
#  BUG ISOLATION DEMOS
# ---------------------------------------------------------------------------


def demonstrate_bug_a() -> None:
    """
    BUG A -- Mutable default argument causes data bleed between calls.

    Diagnosis: Add logging before and after the cache lookup; pipe the
    pipeline_run_id into every log.  Then compare user_id vs returned context
    across two sequential calls with different user_ids.
    """
    print("  -- BUG A: Mutable default argument --")
    from part2_debugging.broken_pipeline import fetch_user_context

    fetch_user_context("alpha")
    fetch_user_context("alpha")  # same user -- returns from cache (ok)
    fetch_user_context("beta")  # different user -- fresh fetch (ok so far)

    # The shared dict is accessible to all callers and grows unbounded.
    import part2_debugging.broken_pipeline as bp

    shared_cache = bp.fetch_user_context.__defaults__[0]  # type: ignore[attr-defined]
    print(
        f"  Shared cache has {len(shared_cache)} entries across ALL calls: "
        f"{list(shared_cache.keys())}"
    )
    print("  -> In a concurrent system this dict grows unbounded and is never evicted.\n")


def demonstrate_bug_b() -> None:
    """
    BUG B -- No timeout on LLM call.

    Diagnosis: Add per-step duration logging.  When you see
      step=generate_summary duration_ms=30412
    on 20% of calls, you know this step has no timeout.  Confirm by looking at
    the source: no timeout= kwarg, no threading.Timer, no asyncio.wait_for.
    """
    print("  -- BUG B: No timeout (simulated fast path only) --")
    print("  Evidence from logs: generate_summary occasionally takes >30 000 ms")
    print("  Fix: wrap LLM calls with a timeout; use tenacity for retry with backoff.\n")


def demonstrate_bug_c() -> None:
    """
    BUG C -- Brittle regex silently returns None.

    Diagnosis: add `assert score is not None` after score_lead() in local tests.
    The assertion fails 40% of the time with the realistic LLM preamble responses.
    Confirm by printing the raw LLM response before the regex.
    """
    print("  -- BUG C: Brittle regex (5 sample inputs) --")

    brittle_regex = re.compile(r"Score:\s*(\d+)")
    robust_regex = re.compile(r"\b(\d{1,3})\b")

    samples = [
        "Score: 72",
        "Based on the provided information, I would score: 85/100.",
        "I'd give this lead a 60 out of 100.",
        "Score: 91",
        "The lead scores approximately 45 points.",
    ]

    brittle_ok = brittle_fail = robust_ok = 0
    for s in samples:
        bm = brittle_regex.search(s)
        rm = robust_regex.search(s)
        brittle_ok += 1 if bm else 0
        brittle_fail += 0 if bm else 1
        robust_ok += 1 if rm else 0
        b_val = bm.group(1) if bm else "None X"
        r_val = rm.group(1) if rm else "None X"
        print(f"  Input : {s[:60]:<60s}")
        print(f"    Brittle -> {b_val:<10}  Robust -> {r_val}")

    print(f"\n  Brittle regex: {brittle_ok}/5 parsed,  {brittle_fail}/5 silently failed")
    print(f"  Robust  regex: {robust_ok}/5 parsed\n")


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Part 2 -- Debugging Walkthrough")
    print("=" * 60 + "\n")

    demonstrate_bug_a()
    demonstrate_bug_b()
    demonstrate_bug_c()

    print("  See fixed_pipeline.py for the corrected implementation.")
    print("  See tests/test_pipeline.py for the regression tests.\n")
