"""
Async fire-and-forget sync to Supabase on each JSON mutation.
This runs as background tasks and never blocks the main agent loop.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SupabaseSync:
    """Write-through cache — every local JSON write also fires to Supabase."""

    def __init__(self, supabase_service: Optional[Any] = None):
        self._supa = supabase_service

    async def sync_message(self, session_id: str, agent: str, message: Dict[str, Any]) -> None:
        """Fire-and-forget: persist a message to Supabase."""
        if self._supa is None:
            return
        try:
            await self._supa.save_message(session_id, agent, message)
        except Exception as e:
            logger.warning(f"Supabase sync failed for message in session {session_id}: {e}")

    async def sync_task(self, session_id: str, task: Dict[str, Any]) -> None:
        """Fire-and-forget: persist a task update to Supabase."""
        if self._supa is None:
            return
        try:
            await self._supa.save_task(session_id, task)
        except Exception as e:
            logger.warning(f"Supabase sync failed for task in session {session_id}: {e}")

    async def sync_agent_turn(self, session_id: str, agent: str, turn_data: Dict[str, Any]) -> None:
        """Fire-and-forget: persist an agent turn to Supabase."""
        if self._supa is None:
            return
        try:
            await self._supa.save_agent_turn(session_id, agent, turn_data)
        except Exception as e:
            logger.warning(f"Supabase sync failed for agent turn in session {session_id}: {e}")
