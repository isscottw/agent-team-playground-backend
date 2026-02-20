"""
ContextBuilder — assembles the system prompt + conversation history for
an agent's LLM call by reading from the inbox JSON and task state.
"""

from typing import Any, Dict, List

from comms.json_store import InboxStore
from comms.task_store import TaskStore
from services.tool_service import TOOL_DEFINITIONS


class ContextBuilder:
    """Builds LLM-ready messages for a single agent turn."""

    def __init__(
        self,
        inbox_store: InboxStore,
        task_store: TaskStore,
        agent_name: str,
        agent_system_prompt: str,
        team_agents: List[str],
    ):
        self.inbox = inbox_store
        self.tasks = task_store
        self.agent_name = agent_name
        self.agent_system_prompt = agent_system_prompt
        self.team_agents = team_agents

    async def build_system_prompt(self) -> str:
        """Build the full system prompt including team context."""
        teammates = [a for a in self.team_agents if a != self.agent_name]
        tasks = await self.tasks.list_tasks()

        task_summary = ""
        if tasks:
            lines = []
            for t in tasks:
                owner = t.get("owner", "unassigned")
                blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
                lines.append(f"  #{t['id']} [{t['status']}] {t['subject']} — owner: {owner}{blocked}")
            task_summary = "\n\nCurrent tasks:\n" + "\n".join(lines)

        return f"""{self.agent_system_prompt}

# Team Context
You are agent "{self.agent_name}" on a team.
Your teammates: {', '.join(teammates) if teammates else '(none)'}

You have the following tools available:
- SendMessage: Send a message to a teammate (type=message) or broadcast to all (type=broadcast)
- TaskCreate: Create a new task in the shared task list
- TaskUpdate: Update a task's status, owner, or details
- TaskList: List all tasks
- TaskGet: Get details of a specific task

When using SendMessage, always specify the recipient by name.
{task_summary}"""

    async def build_messages(
        self,
        conversation_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build the full message list for an LLM call.

        Returns a list of {role, content} dicts starting with the system prompt.
        New inbox messages are injected as the latest user message.
        """
        system_prompt = await self.build_system_prompt()
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Read new inbox messages and append as user context
        unread = await self.inbox.read_unread(self.agent_name)
        if unread:
            inbox_text_parts = []
            for m in unread:
                sender = m.get("from", "unknown")
                text = m.get("text", "")
                inbox_text_parts.append(f"[Message from {sender}]: {text}")
            inbox_block = "\n\n".join(inbox_text_parts)
            messages.append({"role": "user", "content": inbox_block})

        return messages

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return the tool definitions for this agent."""
        return TOOL_DEFINITIONS
