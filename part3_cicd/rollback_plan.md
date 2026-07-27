# Part 3 — Rollback Plan

## First 5 Minutes When a Production Deploy Breaks

### T+0:00 — Alert fires (or user reports)
**First move:** confirm the scope before touching anything.

```bash
# Check error rate in the last 5 minutes
# (command shown for Fly.io; substitute your platform's log tool)
flyctl logs --app my-prod-app | tail -100

# Or via log aggregator
datadog-metrics query "error_rate{env:production}" --from "5 minutes ago"
```

**Decision point:** Is it a complete outage (5xx on all requests) or a partial
degradation (elevated error rate, one endpoint down)?

---

### T+0:30 — Immediate rollback (do NOT investigate root cause first)
Production is down → rollback first, debug after.

```bash
# Option A — Fly.io rolling rollback to previous image
flyctl releases list --app my-prod-app        # find the last good version
flyctl deploy --image <previous-image-sha> --strategy rolling --app my-prod-app

# Option B — GitHub Actions: re-run the deploy job for the previous commit
# Go to Actions → find the last green deploy → "Re-run jobs"

# Option C — Feature flag (if the bug is behind a flag)
# Disable the flag in LaunchDarkly / ConfigCat — instant, no redeploy needed
```

**Target:** production back to last known good state within **3 minutes**.

---

### T+2:00 — Confirm recovery
```bash
# Re-run smoke tests against production
STAGING_URL=https://my-prod-app.fly.dev python part3_cicd/smoke_test.py

# Check error rate is back to baseline
```

---

### T+3:00 — Communicate
- Post to the incident Slack channel: "Rollback complete at T+X. Investigating root cause."
- Update status page (StatusPage, Instatus) to "Investigating".

---

### T+5:00 — Begin root cause analysis (now that users are unblocked)
1. Compare the diff between the broken and good release: `git diff <good-sha> <bad-sha>`
2. Pull structured logs for the period between deploy and rollback
3. Reproduce in staging before touching production again
4. Write a regression test, fix, merge, re-deploy with the test green

---

## Secrets and API Keys in the Pipeline

### Where keys live
| Secret | Storage | Who can see it |
|--------|---------|----------------|
| `OPENAI_API_KEY` | GitHub Actions Secret | GitHub Actions runner only |
| `FLY_API_TOKEN` | GitHub Actions Secret | Deploy job only |
| `STAGING_APP_NAME` | GitHub Actions Secret | Deploy job only |
| Runtime env vars | Platform secret store (Fly.io `flyctl secrets set`) | Running container only |

### Rules enforced in this repo
1. **Never hardcode keys.** `grep -r "sk-" .` is run in CI as a pre-commit check.
2. **Least privilege.** The `FLY_API_TOKEN` is scoped to a single app, not the
   entire org. Rotate it quarterly.
3. **No echo.** Steps that consume secrets never use `echo $SECRET` or `run: echo`.
   GitHub masks known secret values in logs automatically, but we don't rely on
   that as the only control.
4. **Separate staging and production secrets.** The staging deploy job uses
   `STAGING_APP_NAME`; a separate production deploy job (if added) uses
   `PROD_APP_NAME` with a different token.
5. **Secret scanning.** GitHub's native secret scanning is enabled on the repo.
   `truffleHog` can be added as an additional CI step for defence in depth.

### Injecting secrets at runtime (not build time)
Secrets are injected as environment variables at container startup, not baked
into the Docker image.  This means a leaked image doesn't expose keys.

```bash
# Set production secrets on Fly.io (stored encrypted, injected at startup)
flyctl secrets set OPENAI_API_KEY=sk-... --app my-prod-app
# The value is never stored in the repo, the Dockerfile, or build logs.
```
