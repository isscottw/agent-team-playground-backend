"""
AgentRunner — executes a single agent's turn loop.

1. Read inbox
2. Build context (system prompt + conversation + new messages)
3. Call LLM
4. Process tool calls (execute against stores)
5. Loop if there are tool calls, otherwise finish turn
6. Emit SSE events at each step
"""

import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from llm.base import LLMProvider, LLMResponse
from llm.factory import get_provider
from services.context_service import ContextBuilder
from services.tool_service import ToolExecutor
from services.protocol_service import ProtocolService
from comms.json_store import InboxStore
from comms.task_store import TaskStore
from comms.sync import SupabaseSync

logger = logging.getLogger(__name__)

MAX_TOOL_LOOPS = 10  # prevent infinite loops
MAX_HISTORY_MESSAGES = 40  # trigger compaction when exceeded


class AgentRunner:
    """Runs a single agent's turn: read inbox → LLM → tool calls → repeat."""

    def __init__(
        self,
        agent_name: str,
        provider_name: str,
        model: str,
        api_key: str,
        system_prompt: str,
        inbox_store: InboxStore,
        task_store: TaskStore,
        team_agents: List[str],
        emit_sse: Optional[Callable[..., Coroutine]] = None,
        lead_agent: Optional[str] = None,
        is_leader: bool = False,
        color: Optional[str] = None,
        team_roster: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        supabase_sync: Optional[SupabaseSync] = None,
    ):
        self.agent_name = agent_name
        self.provider: LLMProvider = get_provider(provider_name)
        self.model = model
        self.api_key = api_key
        self.inbox = inbox_store
        self.task_store = task_store
        self.team_agents = team_agents
        self.emit_sse = emit_sse  # async callable(event_dict)
        self.lead_agent = lead_agent
        self.color = color
        self.session_id = session_id
        self.supabase_sync = supabase_sync

        self.context_builder = ContextBuilder(
            inbox_store=inbox_store,
            task_store=task_store,
            agent_name=agent_name,
            agent_system_prompt=system_prompt,
            team_agents=team_agents,
            team_roster=team_roster or [],
            is_leader=is_leader,
            lead_agent=lead_agent,
        )

        self.tool_executor = ToolExecutor(
            inbox_store=inbox_store,
            task_store=task_store,
            agent_name=agent_name,
            team_agents=team_agents,
            on_message_sent=self._on_message_sent,
            on_task_assigned=self._on_task_assigned,
            on_task_completed=self._on_task_completed,
            on_task_changed=self._on_task_changed,
        )

        # Conversation history persisted across turns within a session
        self.conversation_history: List[Dict[str, Any]] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def _on_task_changed(self, task: Dict[str, Any]) -> None:
        """Callback when a task is created or updated — emits task_update SSE."""
        await self._emit("task_update", {
            "id": task["id"],
            "subject": task.get("subject", ""),
            "description": task.get("description", ""),
            "status": task.get("status", "pending"),
            "owner": task.get("owner"),
        })
        if self.supabase_sync and self.session_id:
            import asyncio
            asyncio.create_task(self.supabase_sync.sync_task(self.session_id, task))

    async def _on_task_completed(self, task: Dict[str, Any]) -> None:
        """Callback when a task is completed — sends task_completed protocol to lead."""
        if self.lead_agent and self.lead_agent != self.agent_name:
            msg = ProtocolService.create_task_completed(self.agent_name, task["id"], task["subject"])
            await self.inbox.append_message(
                agent=self.lead_agent,
                from_agent=self.agent_name,
                text=msg["text"],
                summary=msg["summary"],
            )
            await self._emit("protocol_message", {
                "protocol_type": "task_completed",
                "task_id": task["id"],
                "task_subject": task["subject"],
                "from": self.agent_name,
            })

    async def _on_task_assigned(self, owner: str, task: Dict[str, Any]) -> None:
        """Callback when a task is assigned — sends protocol message to assignee."""
        msg = ProtocolService.create_task_assignment(self.agent_name, task["id"], task["subject"])
        await self.inbox.append_message(
            agent=owner,
            from_agent=self.agent_name,
            text=msg["text"],
            summary=msg["summary"],
        )
        await self._emit("protocol_message", {
            "protocol_type": "task_assignment",
            "task_id": task["id"],
            "assigned_to": owner,
        })

    async def _on_message_sent(self, to_agent: str, message: Dict[str, Any]) -> None:
        """Callback when a tool sends a message — emits SSE event."""
        if self.emit_sse:
            await self.emit_sse({
                "type": "agent_message",
                "agent": self.agent_name,
                "data": {
                    "to": to_agent,
                    "text": message.get("text", ""),
                    "summary": message.get("summary", ""),
                },
            })

    async def _emit(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        if self.emit_sse:
            await self.emit_sse({
                "type": event_type,
                "agent": self.agent_name,
                "data": data or {},
            })

    def _maybe_compact_history(self) -> None:
        """Trim conversation_history when it exceeds MAX_HISTORY_MESSAGES.

        Keeps a summary marker + the last 20 entries. The system prompt
        already rebuilds team context + task list every turn, so state
        recovery is automatic.
        """
        if len(self.conversation_history) <= MAX_HISTORY_MESSAGES:
            return
        trimmed_count = len(self.conversation_history) - 20
        summary_marker = {
            "role": "user",
            "content": f"[System: {trimmed_count} earlier messages were compacted to save context. Team context and task list are rebuilt in the system prompt above.]",
        }
        self.conversation_history = [summary_marker] + self.conversation_history[-20:]
        logger.info(f"{self.agent_name}: compacted history, trimmed {trimmed_count} messages")

    async def _check_shutdown_request(self) -> bool:
        """Scan raw inbox for shutdown_request. If found, auto-approve and return True."""
        all_msgs = await self.inbox.read_all(self.agent_name)
        for m in all_msgs:
            if m.get("read"):
                continue
            parsed = ProtocolService.parse_protocol_message(m.get("text", ""))
            if parsed and parsed.get("type") == "shutdown_request":
                # Auto-send shutdown_approved to lead, linking requestId
                req_id = parsed.get("requestId", "")
                if self.lead_agent:
                    approved = ProtocolService.create_shutdown_approved(self.agent_name, request_id=req_id)
                    await self.inbox.append_message(
                        agent=self.lead_agent,
                        from_agent=self.agent_name,
                        text=approved["text"],
                        summary=approved["summary"],
                    )
                    await self._emit("protocol_message", {
                        "protocol_type": "shutdown_approved",
                        "from": self.agent_name,
                    })
                # Mark all messages read (including the shutdown_request)
                await self.inbox.read_unread(self.agent_name)
                return True
        return False

    async def run_turn(self) -> Dict[str, Any]:
        """Execute one full turn (may involve multiple LLM calls if tools are used).

        Returns a summary dict of the turn.
        """
        # Check for shutdown request before doing anything
        if await self._check_shutdown_request():
            await self._emit("turn_end", {"shutdown": True})
            return {"agent": self.agent_name, "content": "", "loops": 0, "shutdown": True}

        await self._emit("turn_start")

        # Compact history if needed
        self._maybe_compact_history()

        messages = await self.context_builder.build_messages(self.conversation_history)
        tools = self.context_builder.get_tool_definitions()

        loop_count = 0
        final_content = ""
        should_stop = False  # set when agent sends shutdown_request

        while loop_count < MAX_TOOL_LOOPS and not should_stop:
            loop_count += 1

            await self._emit("thinking", {"loop": loop_count})

            try:
                response: LLMResponse = await self.provider.chat(
                    messages=messages,
                    tools=tools,
                    api_key=self.api_key,
                    model=self.model,
                )
            except Exception as e:
                logger.error(f"LLM call failed for {self.agent_name}: {e}")
                await self._emit("error", {"message": str(e)})
                break

            self.total_prompt_tokens += response.usage.get("prompt_tokens", 0)
            self.total_completion_tokens += response.usage.get("completion_tokens", 0)

            # If there's text content, emit it
            if response.content:
                final_content = response.content
                await self._emit("agent_response", {"content": response.content})
                # Add assistant message to conversation
                messages.append({"role": "assistant", "content": response.content})
                self.conversation_history.append({"role": "assistant", "content": response.content})

            # If no tool calls, turn is done
            if not response.tool_calls:
                break

            # Process tool calls
            # For Anthropic, we need to build the assistant message with tool_use blocks
            # and then user message with tool_result blocks. For simplicity in our
            # conversation history, we track tool calls as text.
            tool_results_text = []
            for tc in response.tool_calls:
                await self._emit("tool_call", {
                    "tool": tc.name,
                    "arguments": tc.arguments,
                    "call_id": tc.id,
                })

                result = await self.tool_executor.execute(tc.name, tc.arguments)

                await self._emit("tool_result", {
                    "tool": tc.name,
                    "call_id": tc.id,
                    "result": result,
                })

                tool_results_text.append(f"[Tool {tc.name} result]: {result}")

                # Stop looping after shutdown_request — agent is done
                if tc.name == "SendMessage" and tc.arguments.get("type") == "shutdown_request":
                    should_stop = True

            # Inject tool results as a user message for the next loop
            combined = "\n\n".join(tool_results_text)
            messages.append({"role": "user", "content": combined})
            self.conversation_history.append({"role": "user", "content": combined})

        await self._emit("turn_end", {
            "loops": loop_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        })

        # Send idle notification to lead agent
        if self.lead_agent and self.lead_agent != self.agent_name:
            idle_msg = ProtocolService.create_idle_notification(from_agent=self.agent_name)
            await self.inbox.append_message(
                agent=self.lead_agent,
                from_agent=self.agent_name,
                text=idle_msg["text"],
                summary=idle_msg["summary"],
            )
        await self._emit("protocol_message", {
            "protocol_type": "idle_notification",
            "from": self.agent_name,
        })

        return {
            "agent": self.agent_name,
            "content": final_content,
            "loops": loop_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }

    async def inject_user_message(self, text: str) -> None:
        """Inject a user/leader message into this agent's inbox."""
        await self.inbox.append_message(
            agent=self.agent_name,
            from_agent="user",
            text=text,
            summary=text[:80],
        )
