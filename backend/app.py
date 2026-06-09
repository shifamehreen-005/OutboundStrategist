"""FastAPI server.

Two streaming endpoints (NDJSON over POST so the UI can render the agent's
steps live) plus static hosting for the single-page frontend.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import Agent
from .config import config
from .schemas import SenderRequest, TargetRequest

app = FastAPI(title="Outbound Strategist")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


def _stream(generator):
    def gen():
        try:
            for event in generator:
                yield json.dumps(event) + "\n"
        except Exception as e:  # noqa: BLE001
            yield json.dumps(
                {"step": "error", "status": "error", "detail": str(e)}
            ) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/health")
def health():
    return {"ok": True, "mock_mode": config.MOCK_MODE, "model": config.MODEL}


@app.post("/api/sender")
def sender(req: SenderRequest):
    agent = Agent()
    return _stream(agent.run_sender(req.url, req.max_pages))


@app.post("/api/target")
def target(req: TargetRequest):
    agent = Agent()
    return _stream(agent.run_target(
        req.url, req.persona_role, req.persona_seniority, req.sender, req.max_pages
    ))


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
