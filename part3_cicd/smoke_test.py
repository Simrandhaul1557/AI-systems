"""
Part 3 — Smoke Test
====================
Runs after every staging deploy to confirm the service is alive and the
critical API endpoints respond correctly.

Called by deploy_staging.yml as a post-deploy gate.
"""

import os
import sys
import json
import httpx

STAGING_URL = os.environ.get("STAGING_URL", "http://localhost:8000")
TIMEOUT = 10


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    if not condition:
        # Accumulate failures; caller decides whether to exit
        check._failures.append(name)  # type: ignore[attr-defined]


check._failures = []  # type: ignore[attr-defined]


def main() -> None:
    print(f"\n  Smoke test against: {STAGING_URL}\n")

    client = httpx.Client(base_url=STAGING_URL, timeout=TIMEOUT)

    # ── Health check ────────────────────────────────────────────────────────
    try:
        r = client.get("/health")
        check("GET /health returns 200", r.status_code == 200, str(r.status_code))
        body = r.json()
        check("Health body has 'status' key", "status" in body)
        check("Status is 'ok'", body.get("status") == "ok", body.get("status"))
    except Exception as exc:
        check("GET /health reachable", False, str(exc))

    # ── Version endpoint ─────────────────────────────────────────────────────
    try:
        r = client.get("/version")
        check("GET /version returns 200", r.status_code == 200, str(r.status_code))
        body = r.json()
        check("Version body has 'version' key", "version" in body)
    except Exception as exc:
        check("GET /version reachable", False, str(exc))

    # ── Pipeline endpoint ────────────────────────────────────────────────────
    try:
        payload = {"user_id": "smoke_test_user"}
        r = client.post("/pipeline/run", json=payload)
        check("POST /pipeline/run returns 200", r.status_code == 200, str(r.status_code))
        body = r.json()
        check("Response has 'score' key", "score" in body)
        check(
            "Score is an integer 0-100",
            isinstance(body.get("score"), int) and 0 <= body["score"] <= 100,
            str(body.get("score")),
        )
    except Exception as exc:
        check("POST /pipeline/run reachable", False, str(exc))

    print()
    failures = check._failures  # type: ignore[attr-defined]
    if failures:
        print(f"  SMOKE TEST FAILED — {len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    else:
        print("  All smoke tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
