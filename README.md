# AI Systems Assignment

Three-part technical assignment covering cost-aware AI systems, debugging under pressure, and deployment discipline.

---

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # 18 tests, all green
python part1_token_optimization/token_counter.py   # before/after token report
python part2_debugging/debugging_walkthrough.py    # bug isolation demos
```

---

## Part 1 — Token / Cost Optimization

**Problem:** a naive agent pipeline stuffs the entire document corpus, all few-shot examples, and the full conversation history into every API call — burning ~51 000 input tokens per query.

### Optimization 1 — Retrieval-Augmented Generation (RAG)

Instead of injecting all documents (~49 000 tokens) into the system prompt on every call, chunks are embedded at index-time and only the top-k relevant chunks (~200 tokens) are retrieved at query-time.

### Optimization 2 — Sliding-window + rolling summary for history

Instead of resending the full conversation history (grows unbounded), only the last N turns are sent verbatim. Older turns are replaced with a single compressed summary (~60 tokens) generated once and cached.

### Before / After

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Input tokens | 51 338 | 404 | **−99.2 %** |
| Cost per query (gpt-4o @ $5/1M) | $0.2567 | $0.0020 | **−99.2 %** |
| Cost at 10K queries/day | $2 567/day | $20/day | **−$2 547/day** |

Run the comparison yourself:
```bash
python part1_token_optimization/token_counter.py
```

**Quality tradeoffs:**

- *RAG:* near-zero for single-domain queries. Multi-domain queries can miss a relevant chunk if k is too small — mitigated by raising k for ambiguous queries and adding a topic router.
- *Sliding-window summary:* the model loses word-for-word recall of early turns but retains all decisions and facts via the summary. Nuanced early-turn commitments are the edge case to watch.
- *Bonus — prompt caching:* stable prompt sections placed at the head get OpenAI server-side KV cache hits. Zero quality impact; ~75 % cost reduction and ~35 % latency drop on cached prefixes.

**Files:**
- `part1_token_optimization/baseline_agent.py` — naive pipeline (before)
- `part1_token_optimization/optimized_agent.py` — optimized pipeline (after)
- `part1_token_optimization/token_counter.py` — before/after comparison script

---

## Part 2 — Debugging

**Problem:** a multi-step agent pipeline intermittently times out, returns malformed output, and silently succeeds with wrong data.

### The three bugs

| Bug | Symptom | Root cause |
|-----|---------|------------|
| A | Wrong user data returned silently | Mutable default argument `_cache: dict = {}` — shared across ALL calls |
| B | Pipeline hangs 20% of the time | No timeout on the LLM HTTP call — slow upstream blocks forever |
| C | `score` stored as `null` in DB | Brittle regex `Score:\s*(\d+)` fails when LLM includes preamble text; returns `None` silently |

### Debugging process

**Step 1 — Triage:** reproduce and characterise. Run the pipeline 30 times, count `{success, null_score, wrong_user, timeout}`. Know the failure rate before touching code.

**Step 2 — Determinism:** set `random.seed(42)` and raise failure probability to 100% for the specific bug being investigated. Turns a flaky test into a guaranteed-failing test.

**Step 3 — Isolate by step:** comment out steps 2-4, confirm step 1 works alone. Add each step back in sequence. The broken step is where failures start appearing — binary search rather than reading all the code at once.

**Step 4 — Scan for three patterns:** (a) external I/O without timeout/retry, (b) state shared across calls (mutable defaults, globals, class attributes), (c) output parsing that assumes a fixed model response format.

**Step 5 — Write the failing test FIRST:** a red test proves you understand the bug. Fix the code, test goes green, becomes a permanent regression guard.

**Step 6 — Deploy with a feature flag:** 1% traffic slice → watch metrics for 15 min → full rollout.

### Fixes applied

- **Bug A:** replace mutable default with an injected `UserContextCache` instance — proper TTL, per-caller isolation, no shared global state.
- **Bug B:** run the LLM call on a daemon thread with `thread.join(timeout)`. Raise `TimeoutError` on hang. Retry with exponential backoff (max 3 attempts).
- **Bug C:** replace `Score:\s*(\d+)` with a general `\b([0-9]{1,3})\b` scan that finds any number in [0, 100]. Fall back to 50 and log a warning rather than returning `None`.

**Files:**
- `part2_debugging/broken_pipeline.py` — original broken code (annotated with bug markers)
- `part2_debugging/fixed_pipeline.py` — corrected implementation
- `part2_debugging/debugging_walkthrough.py` — runnable step-by-step diagnosis
- `tests/test_pipeline.py` — regression tests (18 pass)

---

## Part 3 — CI/CD and Deployment

### Pipeline overview

Two GitHub Actions workflows:

**`ci.yml`** — triggers on every push and PR to any branch:
1. `lint` job: flake8 + black (check mode)
2. `test` job (runs only if lint passes): pytest with coverage, fails if coverage < 80%

**`deploy_staging.yml`** — triggers only on merge to `main`:
1. Re-runs the test gate
2. Builds a Docker image and pushes to GitHub Container Registry
3. Deploys to staging via Fly.io rolling deploy
4. Runs a smoke test against the live staging URL

### Secrets / API key handling

Keys are **never** hardcoded or echoed. The rules enforced:

1. All secrets live in **GitHub Actions Secrets** (encrypted at rest, masked in logs). They are injected only as environment variables into specific jobs — not as build args, not baked into the Docker image.
2. **Least privilege:** the `FLY_API_TOKEN` is scoped to the single staging app, not the org. The `OPENAI_API_KEY` is available only in the jobs that need it.
3. **Runtime injection:** at container startup, `flyctl secrets set` injects keys via the platform's secret store. The Docker image itself contains no credentials.
4. **Secret scanning:** GitHub's native secret scanning is enabled. A `grep -r "sk-"` lint step can be added as a pre-commit check.
5. **Separate staging/prod secrets:** staging and production use separate tokens, separate app names, and separate GitHub Environments with different protection rules.

See `part3_cicd/rollback_plan.md` for the full secrets policy table.

### Rollback plan — first 5 minutes

**T+0:00 — Confirm scope first (30 seconds)**
```bash
flyctl logs --app my-prod-app | Select-Object -Last 100
```
Is it a complete outage or partial degradation? This determines whether you rollback immediately or can take 2 more minutes to investigate.

**T+0:30 — Rollback, don't investigate yet**
```bash
flyctl releases list --app my-prod-app    # find last good SHA
flyctl deploy --image <previous-sha> --strategy rolling --app my-prod-app
```
Target: production back to last known good state within **3 minutes**. If a feature flag exists for the broken change, disable it first — instant, no redeploy.

**T+2:00 — Confirm recovery**
```bash
python part3_cicd/smoke_test.py   # STAGING_URL=https://my-prod-app.fly.dev
```

**T+3:00 — Communicate**
Post to incident Slack channel. Update status page to "Investigating". The root cause analysis happens AFTER users are unblocked, not during the outage.

**T+5:00+ — Root cause**
`git diff <good-sha> <bad-sha>`, pull structured logs for the outage window, reproduce in staging, write a regression test, fix, merge.

**Files:**
- `.github/workflows/ci.yml`
- `.github/workflows/deploy_staging.yml`
- `Dockerfile`
- `part3_cicd/app.py` — minimal FastAPI app with `/health`, `/version`, `/pipeline/run`
- `part3_cicd/smoke_test.py` — post-deploy smoke test
- `part3_cicd/rollback_plan.md` — full rollback + secrets policy

---

## Project structure

```
ai-systems-assignment/
├── part1_token_optimization/
│   ├── baseline_agent.py       # naive 51K-token pipeline
│   ├── optimized_agent.py      # RAG + sliding-window (404 tokens)
│   └── token_counter.py        # before/after comparison
├── part2_debugging/
│   ├── broken_pipeline.py      # 3 intentional bugs
│   ├── fixed_pipeline.py       # all bugs resolved
│   └── debugging_walkthrough.py
├── part3_cicd/
│   ├── app.py                  # FastAPI app
│   ├── smoke_test.py           # post-deploy smoke test
│   └── rollback_plan.md        # secrets + rollback policy
├── tests/
│   ├── test_token_optimization.py   # 5 tests
│   └── test_pipeline.py             # 13 tests
├── .github/workflows/
│   ├── ci.yml                  # lint + test on every push
│   └── deploy_staging.yml      # build + deploy on merge to main
├── Dockerfile
└── requirements.txt
```
