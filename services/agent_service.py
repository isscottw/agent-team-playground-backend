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
from comms.json_store import InboxStore
from comms.task_store import TaskStore

logger = logging.getLogger(__name__)

MAX_TOOL_LOOPS = 10  # prevent infinite loops


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
    ):
        self.agent_name = agent_name
        self.provider: LLMProvider = get_provider(provider_name)
        self.model = model
        self.api_key = api_key
        self.inbox = inbox_store
        self.task_store = task_store
        self.team_agents = team_agents
        self.emit_sse = emit_sse  # async callable(event_dict)

        self.context_builder = ContextBuilder(
            inbox_store=inbox_store,
            task_store=task_store,
            agent_name=agent_name,
            agent_system_prompt=system_prompt,
            team_agents=team_agents,
        )

        self.tool_executor = ToolExecutor(
            inbox_store=inbox_store,
            task_store=task_store,
            agent_name=agent_name,
            team_agents=team_agents,
            on_message_sent=self._on_message_sent,
        )

        # Conversation history persisted across turns within a session
        self.conversation_history: List[Dict[str, Any]] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

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

    async def run_turn(self) -> Dict[str, Any]:
        """Execute one full turn (may involve multiple LLM calls if tools are used).

        Returns a summary dict of the turn.
        """
        await self._emit("turn_start")

        messages = await self.context_builder.build_messages(self.conversation_history)
        tools = self.context_builder.get_tool_definitions()

        loop_count = 0
        final_content = ""

        while loop_count < MAX_TOOL_LOOPS:
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

            # Inject tool results as a user message for the next loop
            combined = "\n\n".join(tool_results_text)
            messages.append({"role": "user", "content": combined})
            self.conversation_history.append({"role": "user", "content": combined})

        await self._emit("turn_end", {
            "loops": loop_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
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
