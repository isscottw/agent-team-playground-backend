"""
TeamEngine — manages the full session lifecycle.

- Creates JSON dirs (inboxes, tasks) for a session
- Spawns AgentRunners for each configured agent
- Routes user chat messages to the lead agent's inbox
- Coordinates turn loops across agents
- Handles stop / cleanup
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

from models.session_models import AgentConfig, SessionRequest
from comms.json_store import InboxStore
from comms.task_store import TaskStore
from comms.sync import SupabaseSync
from services.agent_service import AgentRunner

logger = logging.getLogger(__name__)

MAX_ROUNDS = 20  # max orchestration rounds before auto-stop


class TeamEngine:
    """Orchestrates a team of agents for a single session."""

    def __init__(
        self,
        session_id: str,
        agents: List[AgentConfig],
        api_keys: Dict[str, str],
        emit_sse: Optional[Callable[..., Coroutine]] = None,
        supabase_sync: Optional[SupabaseSync] = None,
    ):
        self.session_id = session_id
        self.agent_configs = agents
        self.api_keys = api_keys
        self.emit_sse = emit_sse
        self.supabase_sync = supabase_sync

        self.inbox_store = InboxStore(session_id)
        self.task_store = TaskStore(session_id)

        self.agent_names = [a.name for a in agents]
        self.runners: Dict[str, AgentRunner] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def _build_runners(self) -> None:
        for cfg in self.agent_configs:
            api_key = self.api_keys.get(cfg.provider, "")
            runner = AgentRunner(
                agent_name=cfg.name,
                provider_name=cfg.provider,
                model=cfg.model,
                api_key=api_key,
                system_prompt=cfg.system_prompt,
                inbox_store=self.inbox_store,
                task_store=self.task_store,
                team_agents=self.agent_names,
                emit_sse=self._wrap_sse,
            )
            self.runners[cfg.name] = runner

    async def _wrap_sse(self, event: Dict[str, Any]) -> None:
        """Add session_id and timestamp, then forward to the SSE emitter."""
        event["session_id"] = self.session_id
        event["timestamp"] = datetime.utcnow().isoformat() + "Z"
        if self.emit_sse:
            await self.emit_sse(event)
        # Fire-and-forget Supabase sync
        if self.supabase_sync and event.get("type") in ("agent_response", "tool_call", "tool_result"):
            asyncio.create_task(
                self.supabase_sync.sync_agent_turn(self.session_id, event.get("agent", ""), event)
            )

    async def start(self) -> None:
        """Initialize stores and start the orchestration loop."""
        self._build_runners()
        self._running = True

        await self._wrap_sse({
            "type": "session_start",
            "data": {
                "session_id": self.session_id,
                "agents": self.agent_names,
            },
        })

        self._task = asyncio.create_task(self._orchestration_loop())

    async def _orchestration_loop(self) -> None:
        """Main loop: run agent turns round-robin until stopped."""
        round_num = 0
        try:
            while self._running and round_num < MAX_ROUNDS:
                round_num += 1

                # Check if any agent has unread messages
                any_unread = False
                for name in self.agent_names:
                    msgs = await self.inbox_store.read_all(name)
                    unread = [m for m in msgs if not m.get("read", False)]
                    if unread:
                        any_unread = True
                        break

                if not any_unread and round_num > 1:
                    # No pending messages — wait briefly then check again
                    await asyncio.sleep(1.0)
                    continue

                # Run each agent that has unread messages
                for name in self.agent_names:
                    if not self._running:
                        break
                    msgs = await self.inbox_store.read_all(name)
                    unread = [m for m in msgs if not m.get("read", False)]
                    if unread:
                        runner = self.runners[name]
                        try:
                            await runner.run_turn()
                        except Exception as e:
                            logger.error(f"Agent {name} turn failed: {e}")
                            await self._wrap_sse({
                                "type": "error",
                                "agent": name,
                                "data": {"message": str(e)},
                            })

                # Brief pause between rounds
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info(f"Session {self.session_id} orchestration cancelled")
        except Exception as e:
            logger.error(f"Orchestration loop error: {e}")
            await self._wrap_sse({
                "type": "error",
                "data": {"message": f"Orchestration error: {e}"},
            })
        finally:
            await self._wrap_sse({
                "type": "session_end",
                "data": {"session_id": self.session_id, "rounds": round_num},
            })

    async def send_user_message(self, text: str, target_agent: Optional[str] = None) -> None:
        """Route a user chat message to an agent's inbox.

        If target_agent is not specified, sends to the first agent (leader).
        """
        target = target_agent or self.agent_names[0]
        await self.inbox_store.append_message(
            agent=target,
            from_agent="user",
            text=text,
            summary=text[:80],
        )

    async def stop(self) -> None:
        """Stop the orchestration loop and clean up."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def cleanup(self) -> None:
        """Remove all session data from disk."""
        self.inbox_store.cleanup()


def create_session(
    request: SessionRequest,
    emit_sse: Optional[Callable[..., Coroutine]] = None,
    supabase_sync: Optional[SupabaseSync] = None,
) -> tuple[str, TeamEngine]:
    """Factory: create a session ID and TeamEngine from a SessionRequest."""
    session_id = str(uuid.uuid4())
    engine = TeamEngine(
        session_id=session_id,
        agents=request.agents,
        api_keys=request.api_keys,
        emit_sse=emit_sse,
        supabase_sync=supabase_sync,
    )
    return session_id, engine
