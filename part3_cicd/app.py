"""
Part 3 — Minimal FastAPI app
==============================
Provides the /health, /version, and /pipeline/run endpoints that the
smoke test validates after every staging deploy.
"""

import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from part2_debugging.fixed_pipeline import run_pipeline

app = FastAPI(title="AI Systems Assignment API", version="1.0.0")


class PipelineRequest(BaseModel):
    user_id: str


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/version")
def version():
    return {"version": app.version, "environment": os.getenv("ENVIRONMENT", "local")}


@app.post("/pipeline/run")
def pipeline_run(req: PipelineRequest):
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    result = run_pipeline(req.user_id)
    return result
