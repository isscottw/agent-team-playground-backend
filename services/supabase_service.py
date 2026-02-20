"""
SupabaseService — write-only sync to Supabase for history / replay.

All writes are best-effort (failures are logged, never raised).
If SUPABASE_URL / SUPABASE_KEY are not set, the service is a no-op.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_supabase_client = None


def _get_client():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        logger.info("SUPABASE_URL/SUPABASE_KEY not set — Supabase sync disabled")
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as e:
        logger.warning(f"Failed to create Supabase client: {e}")
        return None


class SupabaseService:

    async def save_session(self, session_id: str, agents: List[str], config: Dict[str, Any]) -> None:
        client = _get_client()
        if not client:
            return
        try:
            client.table("sessions").insert({
                "id": session_id,
                "agents": agents,
                "config": config,
                "status": "running",
                "created_at": datetime.utcnow().isoformat(),
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to save session to Supabase: {e}")

    async def end_session(self, session_id: str) -> None:
        client = _get_client()
        if not client:
            return
        try:
            client.table("sessions").update({
                "status": "ended",
                "ended_at": datetime.utcnow().isoformat(),
            }).eq("id", session_id).execute()
        except Exception as e:
            logger.warning(f"Failed to end session in Supabase: {e}")

    async def save_message(self, session_id: str, agent: str, message: Dict[str, Any]) -> None:
        client = _get_client()
        if not client:
            return
        try:
            client.table("messages_history").insert({
                "session_id": session_id,
                "agent": agent,
                "from_agent": message.get("from", ""),
                "text": message.get("text", ""),
                "summary": message.get("summary", ""),
                "timestamp": message.get("timestamp", datetime.utcnow().isoformat()),
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to save message to Supabase: {e}")

    async def save_task(self, session_id: str, task: Dict[str, Any]) -> None:
        client = _get_client()
        if not client:
            return
        try:
            client.table("tasks_history").insert({
                "session_id": session_id,
                "task_id": task.get("id", ""),
                "subject": task.get("subject", ""),
                "status": task.get("status", ""),
                "owner": task.get("owner"),
                "snapshot": task,
                "timestamp": datetime.utcnow().isoformat(),
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to save task to Supabase: {e}")

    async def save_agent_turn(self, session_id: str, agent: str, turn_data: Dict[str, Any]) -> None:
        client = _get_client()
        if not client:
            return
        try:
            client.table("agent_turns").insert({
                "session_id": session_id,
                "agent": agent,
                "event_type": turn_data.get("type", ""),
                "data": turn_data.get("data", {}),
                "timestamp": turn_data.get("timestamp", datetime.utcnow().isoformat()),
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to save agent turn to Supabase: {e}")

    async def get_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        client = _get_client()
        if not client:
            return []
        try:
            result = client.table("sessions").select("*").order(
                "created_at", desc=True
            ).limit(limit).execute()
            return result.data
        except Exception as e:
            logger.warning(f"Failed to get sessions from Supabase: {e}")
            return []

    async def get_session_detail(self, session_id: str) -> Optional[Dict[str, Any]]:
        client = _get_client()
        if not client:
            return None
        try:
            session = client.table("sessions").select("*").eq("id", session_id).single().execute()
            messages = client.table("messages_history").select("*").eq(
                "session_id", session_id
            ).order("timestamp").execute()
            turns = client.table("agent_turns").select("*").eq(
                "session_id", session_id
            ).order("timestamp").execute()
            return {
                "session": session.data,
                "messages": messages.data,
                "turns": turns.data,
            }
        except Exception as e:
            logger.warning(f"Failed to get session detail from Supabase: {e}")
            return None
