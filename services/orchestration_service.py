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
from services.protocol_service import ProtocolService

logger = logging.getLogger(__name__)

AGENT_COLORS = ["blue", "green", "orange", "purple"]
IDLE_TIMEOUT = 300  # 5 minutes with no activity before auto-stop
LEADER_NUDGE_INTERVAL = 60  # seconds idle before nudging leader to check on team


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
        self._has_activity = asyncio.Event()
        self._last_active: Dict[str, float] = {}  # agent_name → timestamp of last turn

    def _build_runners(self) -> None:
        # Build role lookup for hierarchy detection
        role_lookup = {cfg.name: cfg.role for cfg in self.agent_configs}

        # Top-level leader: first agent with role="leader", else first agent
        top_leader = None
        for cfg in self.agent_configs:
            if cfg.role == "leader":
                top_leader = cfg.name
                break
        if not top_leader and self.agent_names:
            top_leader = self.agent_names[0]

        # Determine per-agent lead based on hierarchy:
        # - Top leader has no lead
        # - Sub-leaders report to a leader in their connections
        # - Teammates report to a leader in their connections
        agent_lead: Dict[str, Optional[str]] = {}
        for cfg in self.agent_configs:
            if cfg.name == top_leader:
                agent_lead[cfg.name] = None  # Top leader has no lead
            else:
                # Find a leader in this agent's connections
                my_lead = None
                for conn_name in (cfg.connections or []):
                    if role_lookup.get(conn_name) == "leader" and conn_name != cfg.name:
                        my_lead = conn_name
                        break
                if not my_lead:
                    my_lead = top_leader  # fallback to top leader
                agent_lead[cfg.name] = my_lead
        logger.info(f"Hierarchy: top_leader={top_leader}, agent_lead={agent_lead}")

        # Build full roster lookup for filtering per-agent
        full_roster = {
            cfg.name: {
                "name": cfg.name,
                "role": cfg.role,
                "description": cfg.system_prompt[:200] if cfg.system_prompt else "",
            }
            for cfg in self.agent_configs
        }

        for i, cfg in enumerate(self.agent_configs):
            # Use per-agent connections if provided, otherwise all teammates
            if cfg.connections:
                team_agents = [cfg.name] + cfg.connections
            else:
                team_agents = self.agent_names

            # Only pass roster entries for this agent's direct connections
            agent_roster = [
                full_roster[name] for name in team_agents
                if name in full_roster and name != cfg.name
            ]
            logger.info(f"Agent '{cfg.name}' role={cfg.role} lead={agent_lead[cfg.name]} team={team_agents}")

            api_key = self.api_keys.get(cfg.provider, "")
            runner = AgentRunner(
                agent_name=cfg.name,
                provider_name=cfg.provider,
                model=cfg.model,
                api_key=api_key,
                system_prompt=cfg.system_prompt,
                inbox_store=self.inbox_store,
                task_store=self.task_store,
                team_agents=team_agents,
                emit_sse=self._wrap_sse,
                lead_agent=agent_lead[cfg.name],
                is_leader=(cfg.role == "leader"),
                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                team_roster=agent_roster,
                session_id=self.session_id,
                supabase_sync=self.supabase_sync,
            )
            self.runners[cfg.name] = runner

    async def _wrap_sse(self, event: Dict[str, Any]) -> None:
        """Add session_id and timestamp, then forward to the SSE emitter."""
        event["session_id"] = self.session_id
        event["timestamp"] = datetime.utcnow().isoformat() + "Z"
        if self.emit_sse:
            await self.emit_sse(event)
        if not self.supabase_sync:
            return
        etype = event.get("type", "")
        # Sync agent turns
        if etype in ("agent_response", "tool_call", "tool_result"):
            asyncio.create_task(
                self.supabase_sync.sync_agent_turn(self.session_id, event.get("agent", ""), event)
            )
        # Sync agent text responses as messages
        if etype == "agent_response":
            agent = event.get("agent", "")
            asyncio.create_task(
                self.supabase_sync.sync_message(self.session_id, agent, {
                    "from": agent,
                    "text": event.get("data", {}).get("content", ""),
                    "summary": event.get("data", {}).get("content", "")[:80],
                    "timestamp": event.get("timestamp", ""),
                })
            )
        # Sync agent-to-agent messages
        if etype == "agent_message":
            data = event.get("data", {})
            asyncio.create_task(
                self.supabase_sync.sync_message(self.session_id, data.get("to", ""), {
                    "from": event.get("agent", ""),
                    "text": data.get("text", ""),
                    "summary": data.get("summary", ""),
                    "timestamp": event.get("timestamp", ""),
                })
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

    async def _has_incomplete_tasks(self) -> bool:
        """Check if there are any non-completed tasks."""
        tasks = await self.task_store.list_tasks()
        return any(t.get("status") not in ("completed", "deleted") for t in tasks)

    async def _nudge_leaders(self) -> None:
        """Send status-check messages to each leader about their own reports' tasks."""
        import time
        now = time.time()
        tasks = await self.task_store.list_tasks()
        incomplete = [t for t in tasks if t.get("status") not in ("completed", "deleted")]
        if not incomplete:
            return

        # Build leader → direct reports mapping from the agent_lead hierarchy
        role_lookup = {cfg.name: cfg.role for cfg in self.agent_configs}
        # Determine each agent's lead (same logic as _build_runners)
        top_leader = self._get_lead_agent()
        agent_lead: Dict[str, Optional[str]] = {}
        for cfg in self.agent_configs:
            if cfg.name == top_leader:
                agent_lead[cfg.name] = None
            else:
                my_lead = None
                for conn_name in (cfg.connections or []):
                    if role_lookup.get(conn_name) == "leader" and conn_name != cfg.name:
                        my_lead = conn_name
                        break
                agent_lead[cfg.name] = my_lead or top_leader

        # Build leader → reports
        leader_reports: Dict[str, List[str]] = {}
        for agent_name, lead in agent_lead.items():
            if lead:
                leader_reports.setdefault(lead, []).append(agent_name)

        # Nudge each leader about their own reports' incomplete tasks
        for leader_name, reports in leader_reports.items():
            report_tasks = [t for t in incomplete if t.get("owner") in reports or (not t.get("owner") and leader_name == top_leader)]
            if not report_tasks:
                continue

            lines = []
            for t in report_tasks:
                owner = t.get("owner", "unassigned")
                status = t.get("status", "pending")
                # Check if the owner has been idle
                idle_info = ""
                if owner != "unassigned":
                    last_active = self._last_active.get(owner, 0)
                    if last_active > 0:
                        idle_secs = int(now - last_active)
                        if status == "in_progress":
                            idle_info = f" — working (last active {idle_secs}s ago)"
                        elif status == "pending":
                            idle_info = f" — NOT STARTED, idle {idle_secs}s"
                    else:
                        idle_info = " — never ran a turn"
                lines.append(f"  #{t['id']} {t['subject']} [{status}] owner: {owner}{idle_info}")

            task_block = "\n".join(lines)
            await self.inbox_store.append_message(
                agent=leader_name,
                from_agent="system",
                text=f"[Status check] Your team has been idle. Tasks needing attention:\n{task_block}\n\nIf a task is 'in_progress', the teammate may still be working — be patient. If a task is 'pending' and the owner has been idle, follow up or reassign the task.",
                summary="Status check: tasks needing attention",
            )
            logger.info(f"Nudged leader '{leader_name}' about {len(report_tasks)} tasks")

    def _get_lead_agent(self) -> Optional[str]:
        for cfg in self.agent_configs:
            if cfg.role == "leader":
                return cfg.name
        return self.agent_names[0] if self.agent_names else None

    async def _orchestration_loop(self) -> None:
        """Main loop: wait for messages, then run agent turns."""
        idle_seconds = 0
        last_nudge_at = 0  # track when we last nudged the leader
        try:
            while self._running:
                # Check all agents for unread messages
                agents_with_unread = []
                for name in self.agent_names:
                    msgs = await self.inbox_store.read_all(name)
                    unread = [m for m in msgs if not m.get("read", False)]
                    if unread:
                        agents_with_unread.append(name)

                if not agents_with_unread:
                    # No pending messages — wait briefly then check again
                    await asyncio.sleep(1.0)
                    idle_seconds += 1

                    # Nudge leaders if idle too long and tasks remain
                    if (idle_seconds >= LEADER_NUDGE_INTERVAL
                            and idle_seconds - last_nudge_at >= LEADER_NUDGE_INTERVAL
                            and await self._has_incomplete_tasks()):
                        await self._nudge_leaders()
                        last_nudge_at = idle_seconds

                    if idle_seconds >= IDLE_TIMEOUT:
                        logger.info(f"Session {self.session_id} idle timeout after {IDLE_TIMEOUT}s")
                        break
                    continue

                # Reset idle counter
                idle_seconds = 0

                # Run all agents with unread messages in parallel
                async def _run_agent(name: str) -> None:
                    import time
                    runner = self.runners[name]
                    try:
                        logger.info(f"Running turn for agent '{name}' in session {self.session_id}")
                        await runner.run_turn()
                        self._last_active[name] = time.time()
                        logger.info(f"Completed turn for agent '{name}'")
                    except Exception as e:
                        logger.error(f"Agent {name} turn failed: {e}", exc_info=True)
                        await self._wrap_sse({
                            "type": "error",
                            "agent": name,
                            "data": {"message": str(e)},
                        })

                await asyncio.gather(*[_run_agent(n) for n in agents_with_unread])

                # Brief pause between rounds to allow message delivery
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info(f"Session {self.session_id} orchestration cancelled")
        except Exception as e:
            logger.error(f"Orchestration loop error: {e}", exc_info=True)
            await self._wrap_sse({
                "type": "error",
                "data": {"message": f"Orchestration error: {e}"},
            })
        finally:
            await self._wrap_sse({
                "type": "session_end",
                "data": {"session_id": self.session_id},
            })

    async def send_user_message(self, text: str, target_agent: Optional[str] = None) -> None:
        """Route a user chat message to the lead agent's inbox."""
        target = target_agent or self._get_lead_agent() or self.agent_names[0]
        msg = await self.inbox_store.append_message(
            agent=target,
            from_agent="user",
            text=text,
            summary=text[:80],
        )
        logger.info(f"User message delivered to '{target}' in session {self.session_id}")
        if self.supabase_sync:
            asyncio.create_task(
                self.supabase_sync.sync_message(self.session_id, target, msg)
            )

    async def stop(self) -> None:
        """Stop the orchestration loop and clean up."""
        self._running = False

        # Send shutdown_request to each agent
        for name in self.agent_names:
            msg = ProtocolService.create_shutdown_request(from_agent="system", reason="session ending", target=name)
            await self.inbox_store.append_message(
                agent=name, from_agent="system",
                text=msg["text"], summary=msg["summary"],
            )
        await self._wrap_sse({
            "type": "protocol_message",
            "data": {"protocol_type": "shutdown_request", "reason": "session ending"},
        })

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
