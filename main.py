"""
FastAPI Backend for Agent Team Playground
Provides streaming API endpoints for multi-agent team orchestration.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from models.session_models import SessionRequest, SessionResponse
from services.model_service import ModelService
from services.orchestration_service import TeamEngine, create_session
from services.supabase_service import SupabaseService
from comms.sync import SupabaseSync
from utils.sse import SSEBroadcaster
from utils.metrics import TokenTracker

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agent Team Playground API",
    description="Multi-agent team orchestration with SSE streaming",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Global state ----------
model_service = ModelService()
supabase_service = SupabaseService()
supabase_sync = SupabaseSync(supabase_service)
sse_broadcaster = SSEBroadcaster()
token_tracker = TokenTracker()

# Active sessions: session_id -> TeamEngine
sessions: Dict[str, TeamEngine] = {}


# ---------- Request models ----------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message text")
    target_agent: Optional[str] = Field(None, description="Agent to send to (defaults to lead)")


class LLMTestRequest(BaseModel):
    provider: str = Field(..., description="Provider name: anthropic, openai, kimi, ollama")
    api_key: str = Field(..., description="API key to validate")
    model: Optional[str] = Field(None, description="Specific model to test")


# ---------- Health ----------

@app.get("/")
@app.head("/")
async def root():
    return {
        "message": "Agent Team Playground API is running",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
@app.head("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_sessions": len(sessions),
        "services": {
            "model_service": "active",
            "sse_broadcaster": "active",
        },
    }


# ---------- Sessions ----------

@app.post("/api/sessions", response_model=SessionResponse)
async def create_new_session(request: SessionRequest):
    """Start a new agent team session."""
    # Merge env-var API keys as fallbacks
    env_keys = {
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "kimi": os.getenv("KIMI_API_KEY", ""),
    }
    for provider, env_key in env_keys.items():
        if env_key and not request.api_keys.get(provider):
            request.api_keys[provider] = env_key

    # Validate models
    for agent in request.agents:
        if not model_service.validate_model(agent.model):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {agent.model}",
            )
        if agent.provider not in request.api_keys and agent.provider != "ollama":
            raise HTTPException(
                status_code=400,
                detail=f"Missing API key for provider: {agent.provider}. Set it in .env or pass in request.",
            )

    async def emit_sse(event):
        await sse_broadcaster.broadcast(session_id, event)

    session_id, engine = create_session(
        request=request,
        emit_sse=emit_sse,
        supabase_sync=supabase_sync,
    )
    sessions[session_id] = engine

    # Persist to Supabase (fire-and-forget, strip API keys)
    safe_config = request.model_dump()
    safe_config.pop("api_keys", None)
    await supabase_service.save_session(
        session_id,
        [a.name for a in request.agents],
        safe_config,
    )

    # Start orchestration
    await engine.start()

    return SessionResponse(
        session_id=session_id,
        agents=[a.name for a in request.agents],
        status="running",
    )


@app.delete("/api/sessions/{session_id}")
async def stop_session(session_id: str):
    """Stop and clean up a session."""
    engine = sessions.get(session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")

    await engine.stop()
    engine.cleanup()
    sse_broadcaster.cleanup(session_id)
    token_tracker.clear_session(session_id)
    del sessions[session_id]

    await supabase_service.end_session(session_id)

    return {"status": "stopped", "session_id": session_id}


# ---------- Chat ----------

@app.post("/api/sessions/{session_id}/chat")
async def send_chat(session_id: str, request: ChatRequest):
    """User sends a message to the team."""
    engine = sessions.get(session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")

    await engine.send_user_message(request.message, request.target_agent)
    return {"status": "sent", "session_id": session_id}


# ---------- SSE Stream ----------

@app.get("/api/sessions/{session_id}/stream")
async def stream_events(session_id: str):
    """SSE stream of all events for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    queue = sse_broadcaster.subscribe(session_id)

    return StreamingResponse(
        sse_broadcaster.event_generator(session_id, queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        },
    )


# ---------- LLM Test ----------

@app.post("/api/llm/test")
async def test_llm_key(request: LLMTestRequest):
    """Validate an API key by making a minimal LLM call."""
    from llm.factory import get_provider

    # Fallback to env var if no key provided
    api_key = request.api_key
    if not api_key:
        env_map = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "kimi": "KIMI_API_KEY"}
        api_key = os.getenv(env_map.get(request.provider, ""), "")

    try:
        provider = get_provider(request.provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        response = await provider.chat(
            messages=[
                {"role": "system", "content": "Reply with exactly: OK"},
                {"role": "user", "content": "Test"},
            ],
            api_key=api_key,
            model=request.model,
        )
        return {
            "status": "valid",
            "provider": request.provider,
            "model": request.model,
            "response": response.content[:100],
        }
    except Exception as e:
        return {
            "status": "invalid",
            "provider": request.provider,
            "error": str(e),
        }


# ---------- Models ----------

@app.get("/api/models")
async def get_models():
    """Get available models grouped by provider."""
    return model_service.get_available_models()


# ---------- History ----------

@app.get("/api/history")
async def get_history(limit: int = 20):
    """Get past sessions from Supabase."""
    sessions_list = await supabase_service.get_sessions(limit=limit)
    return {"sessions": sessions_list}


@app.get("/api/history/{session_id}")
async def get_session_history(session_id: str):
    """Get full replay data for a session."""
    detail = await supabase_service.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Session not found in history")
    return detail


@app.delete("/api/history/{session_id}")
async def delete_session_history(session_id: str):
    """Delete a session and all its data from Supabase."""
    deleted = await supabase_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# ---------- Entry point ----------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    reload = os.getenv("ENVIRONMENT", "production") == "development"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload, log_level="info")
