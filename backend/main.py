"""FastAPI backend: serves precomputed game_state.json, runs the agent loop,
and serves clips + the final report.

Deliberately does NOT run the perception pipeline (YOLO/homography/possession
segmentation) — that runs offline via scripts/run_pipeline.sh so the live
demo never waits on GPU inference. The agent loop (Claude Haiku tool-calling)
is cheap enough to run live.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from sse_starlette.sse import EventSourceResponse

load_dotenv()

from agent.agent import run_agent_loop
from agent.report import build_report_document
from perception.schemas import GameState

GAME_STATE_PATH = os.environ.get("GAME_STATE_PATH", "data/processed/game_state.json")
CLIPS_DIR = os.environ.get("CLIPS_DIR", "data/processed/clips")

app = FastAPI(title="hoop-scout backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory run registry: agent runs are short-lived and single-clip, no DB needed
_runs: dict[str, dict] = {}


def _load_game_state() -> GameState:
    if not Path(GAME_STATE_PATH).exists():
        raise HTTPException(404, f"game_state.json not found at {GAME_STATE_PATH} — run the perception pipeline first")
    return GameState.model_validate_json(Path(GAME_STATE_PATH).read_text())


@app.get("/api/game-state")
def get_game_state() -> dict:
    return json.loads(Path(GAME_STATE_PATH).read_text())


RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _range_response(request: Request, path: Path, media_type: str) -> Response:
    """Serves a video file honoring HTTP Range requests.

    Browsers seek by requesting a byte range starting near the target playback
    position — without a real 206 Partial Content response here, <video> seeks
    silently fail (currentTime resets to 0), which is exactly what broke the
    citation-seeking feature in manual testing. FastAPI's plain FileResponse
    doesn't reliably do this, so it's handled explicitly.
    """
    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(path, media_type=media_type)

    match = RANGE_RE.match(range_header)
    if not match:
        raise HTTPException(416, "Invalid Range header")

    start = int(match.group(1)) if match.group(1) else 0
    end = int(match.group(2)) if match.group(2) else file_size - 1
    end = min(end, file_size - 1)
    chunk_size = end - start + 1

    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(chunk_size)

    return Response(
        content=data,
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
        },
    )


@app.get("/api/video")
def get_video(request: Request) -> Response:
    """Serves the raw source video — the main player seeks against this using the
    global timestamps in game_state.json, which is what makes citation-seeking work."""
    game_state = _load_game_state()
    video_path = Path(game_state.video.path)
    if not video_path.exists():
        raise HTTPException(404, f"source video not found at {video_path}")
    return _range_response(request, video_path, "video/mp4")


@app.get("/api/clips/{possession_id}")
def get_clip(possession_id: str, request: Request) -> Response:
    clip_path = Path(CLIPS_DIR) / f"{possession_id}.mp4"
    if not clip_path.exists():
        raise HTTPException(404, f"No clip for possession {possession_id}")
    return _range_response(request, clip_path, "video/mp4")


@app.post("/api/agent/runs")
def start_agent_run() -> dict:
    """Kicks off the agent loop in a background thread; poll/stream via run_id."""
    game_state = _load_game_state()
    run_id = str(uuid.uuid4())
    _runs[run_id] = {"status": "running", "events": [], "result": None}

    def on_event(entry: dict) -> None:
        _runs[run_id]["events"].append(entry)

    def worker() -> None:
        try:
            result = run_agent_loop(game_state, video_path=game_state.video.path, on_event=on_event)
            _runs[run_id]["result"] = build_report_document(game_state, result)
            _runs[run_id]["status"] = "done"
        except Exception as exc:
            _runs[run_id]["status"] = "error"
            _runs[run_id]["error"] = str(exc)

    threading.Thread(target=worker, daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/agent/runs/{run_id}/stream")
async def stream_agent_run(run_id: str) -> EventSourceResponse:
    """Server-sent events of the tool-call trail as it happens — this is the
    reasoning trail the frontend renders live."""
    if run_id not in _runs:
        raise HTTPException(404, "unknown run_id")

    async def event_generator():
        sent = 0
        while True:
            run = _runs[run_id]
            events = run["events"]
            for entry in events[sent:]:
                yield {"event": "tool_call", "data": json.dumps(entry, default=str)}
            sent = len(events)
            if run["status"] in ("done", "error"):
                yield {"event": run["status"], "data": json.dumps(run.get("result") or run.get("error"), default=str)}
                break
            import asyncio
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.get("/api/agent/runs/{run_id}")
def get_agent_run(run_id: str) -> dict:
    if run_id not in _runs:
        raise HTTPException(404, "unknown run_id")
    return _runs[run_id]
