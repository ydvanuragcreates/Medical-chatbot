"""MediBot API: FastAPI app serving the chat UI and a streaming RAG endpoint.

Run locally:  uvicorn main:app --reload --port 8080
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import get_settings
from src.rag import RAGService

BASE_DIR = Path(__file__).resolve().parent
settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("medibot")


# ---------- Models ----------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)


class Source(BaseModel):
    page: str | None
    topic: str | None = None
    section: str | None = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------- Lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = None
    app.state.startup_error = None

    missing = settings.missing_keys()
    if missing:
        app.state.startup_error = f"Missing environment variables: {', '.join(missing)}"
        logger.error(app.state.startup_error)
    else:
        try:
            # Loading the embedding model is slow and blocking; keep it off the event loop.
            app.state.rag = await asyncio.to_thread(RAGService, settings)
            logger.info("MediBot is ready (model=%s)", settings.openai_model)
        except Exception as exc:  # noqa: BLE001 - surfaced through /api/health
            app.state.startup_error = f"Could not initialise the knowledge base: {exc}"
            logger.exception("Startup failed")
    yield


app = FastAPI(title="MediBot API", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def get_rag(request: Request) -> RAGService:
    rag = request.app.state.rag
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.startup_error or "Service is not ready yet.",
        )
    return rag


def friendly_error(exc: Exception) -> str:
    """Map provider errors to messages that are safe to show to users."""
    if isinstance(exc, openai.AuthenticationError):
        return "The AI service rejected the API key. Check OPENAI_API_KEY in your .env file."
    if isinstance(exc, openai.RateLimitError):
        return "The AI service is rate limited or out of credit. Please try again shortly."
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return "Couldn't reach the AI service. Check your internet connection and try again."
    if isinstance(exc, openai.NotFoundError):
        return f"The model '{settings.openai_model}' isn't available for this API key."
    return "Something went wrong while generating the answer. Please try again."


# ---------- Routes ----------

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/api/health")
async def health(request: Request):
    ready = request.app.state.rag is not None
    return {
        "status": "ok" if ready else "error",
        "ready": ready,
        "model": settings.openai_model,
        "error": request.app.state.startup_error,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    """Non-streaming answer, handy for scripts and tests."""
    rag = get_rag(request)
    history = [m.model_dump() for m in body.history]
    answer, sources = [], []
    try:
        async for event in rag.stream(body.message.strip(), history):
            if event["type"] == "sources":
                sources = event["sources"]
            else:
                answer.append(event["content"])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat request failed")
        raise HTTPException(status_code=502, detail=friendly_error(exc)) from exc
    return ChatResponse(answer="".join(answer), sources=sources)


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    """Server-sent events: sources, token…, then done (or error)."""
    rag = get_rag(request)
    history = [m.model_dump() for m in body.history]
    question = body.message.strip()
    logger.info("Question: %s", question)

    async def event_source():
        def sse(event: dict) -> str:
            return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        try:
            async for event in rag.stream(question, history):
                if await request.is_disconnected():
                    logger.info("Client disconnected; stopping generation")
                    return
                yield sse(event)
            yield sse({"type": "done"})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming chat failed")
            yield sse({"type": "error", "message": friendly_error(exc)})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
