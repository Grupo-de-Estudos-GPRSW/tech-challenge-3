"""FastAPI server exposing the LangGraph medical pipeline as a chat interface.

Run with:  python run_ui.py     (or: uvicorn webapp.server:api --reload)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp import pipeline

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Kick off the (slow) model + vector store load without blocking the server.
    pipeline.start_loading()
    yield


api = FastAPI(title="GPRSW Medical Assistant", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@api.get("/api/status")
async def status() -> Dict[str, Any]:
    return pipeline.get_status()


@api.post("/api/session/reset")
async def reset(payload: Dict[str, str]) -> Dict[str, str]:
    session_id = payload.get("session_id") or ""
    pipeline.reset_session(session_id)
    return {"session_id": str(uuid.uuid4())}


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@api.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Stream the graph execution back as Server-Sent Events."""
    session_id = request.session_id or str(uuid.uuid4())
    session = pipeline.get_session(session_id)
    message = request.message.strip()

    async def event_stream():
        yield _sse({"type": "session", "session_id": session_id})

        state = pipeline.get_status()
        if state["state"] != "ready":
            detail = ("The pipeline is still loading: " + state["detail"]
                      if state["state"] == "loading"
                      else "The pipeline failed to load: " + state["detail"])
            yield _sse({"type": "error", "message": detail})
            return
        if not message:
            yield _sse({"type": "error", "message": "Empty message."})
            return

        session.messages.append({"role": "user", "content": message})

        # `run_turn` is a blocking generator (the graph runs in its own thread);
        # pull from it without stalling the event loop.
        iterator = pipeline.run_turn(session, message)
        while True:
            event = await asyncio.to_thread(lambda: next(iterator, None))
            if event is None:
                break
            if event["type"] == "result":
                session.messages.append({"role": "assistant", "content": event["answer"]})
            yield _sse(event)
        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #

@api.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
