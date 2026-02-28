"""
ContextBuilder — assembles the system prompt + conversation history for
an agent's LLM call by reading from the inbox JSON and task state.
"""

from typing import Any, Dict, List, Optional

from comms.json_store import InboxStore
from comms.task_store import TaskStore
from services.protocol_service import ProtocolService
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
        team_roster: Optional[List[Dict[str, str]]] = None,
        is_leader: bool = False,
        lead_agent: Optional[str] = None,
    ):
        self.inbox = inbox_store
        self.tasks = task_store
        self.agent_name = agent_name
        self.agent_system_prompt = agent_system_prompt
        self.team_agents = team_agents
        self.team_roster = team_roster or []  # [{name, role, description}]
        self.is_leader = is_leader
        self.lead_agent = lead_agent  # who this agent reports to (None for top leader)

    async def build_system_prompt(self) -> str:
        """Build the full system prompt including team context."""
        # Direct reports (for leaders) or teammates (for workers)
        direct_connections = [a for a in self.team_agents if a != self.agent_name]

        all_tasks = await self.tasks.list_tasks()

        # Scope task list: only show tasks relevant to this agent's scope
        # Leaders see tasks owned by themselves or their direct reports
        # Teammates see only their own tasks
        scope_names = {self.agent_name} | set(direct_connections)
        relevant_tasks = [
            t for t in all_tasks
            if t.get("owner", "unassigned") in scope_names or t.get("owner") is None
        ]

        task_summary = ""
        if relevant_tasks:
            lines = []
            for t in relevant_tasks:
                owner = t.get("owner", "unassigned")
                blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
                lines.append(f"  #{t['id']} [{t['status']}] {t['subject']} — owner: {owner}{blocked}")
            task_summary = "\n\nCurrent tasks:\n" + "\n".join(lines)

        # Build connection descriptions — only direct connections
        conn_lines = []
        for name in direct_connections:
            roster_entry = next((r for r in self.team_roster if r["name"] == name), None)
            if roster_entry and roster_entry.get("description"):
                role_label = "leader" if roster_entry.get("role") == "leader" else "teammate"
                conn_lines.append(f"  - {name} ({role_label}): {roster_entry['description']}")
            else:
                conn_lines.append(f"  - {name}")
        conn_block = "\n".join(conn_lines) if conn_lines else "  (none)"

        # Reporting line
        superior_line = ""
        if self.lead_agent:
            superior_line = f"\nYou report to: {self.lead_agent}"

        return f"""{self.agent_system_prompt}

# Team Context
You are agent "{self.agent_name}".{superior_line}
Your direct team:
{conn_block}

Tools available:
- SendMessage: Send a message (type=message, recipient=name) or broadcast (type=broadcast)
- TaskCreate: Create a new task
- TaskUpdate: Update task status/owner
- TaskList: List all tasks
- TaskGet: Get task details

IMPORTANT: Before creating tasks, check the "Current tasks" list below. Do NOT create tasks that already exist. Use each agent's name exactly as shown above.
{self._role_instructions()}
{task_summary}"""

    def _role_instructions(self) -> str:
        """Generate role-specific instructions based on hierarchy position.

        Leaders are recursive — any leader delegates to its direct reports.
        A leader with a parent leader is automatically a sub-leader.
        """
        if self.is_leader:
            # Identify direct reports (everyone in team_agents except self and parent)
            direct_reports = [
                a for a in self.team_agents
                if a != self.agent_name and a != self.lead_agent
            ]
            reports_str = ", ".join(f'"{r}"' for r in direct_reports) if direct_reports else "your teammates"

            # Reporting line: top leader receives from user, sub-leader receives from parent
            if self.lead_agent:
                reporting = f"""You report to "{self.lead_agent}".
When you receive tasks from "{self.lead_agent}", delegate them to your team."""
                completion = f"""After ALL your reports have completed and shut down:
7. Compile your team's deliverables into a COMPLETE report and send it to "{self.lead_agent}" via SendMessage. Include ALL the actual content from your reports — your lead cannot see what your teammates sent you, only what you explicitly forward. Do NOT just say "work is done" — send the full compiled output.
8. Then request shutdown: SendMessage with type="shutdown_request", recipient="{self.lead_agent}", content="All tasks complete" """
            else:
                reporting = "You receive requests directly from the user."
                completion = """After ALL your reports have completed and shut down:
7. Write a comprehensive FINAL REPORT as your text response (not via SendMessage). This report is shown directly to the user and should include:
   - A summary of what was accomplished
   - Key results or findings from each teammate
   - Any issues encountered
   This is CRITICAL — the user cannot see inter-agent messages, so your final text response is their ONLY way to see the results."""

            return f"""
## Leader Responsibilities
You are a LEADER who manages: {reports_str}.
{reporting}

CRITICAL: You must NEVER do the work yourself. Your job is to DELEGATE to your team.

1. Break tasks into sub-tasks using TaskCreate — one per report
2. Assign each sub-task using TaskUpdate (set owner to their exact name)
3. Send each report a message via SendMessage explaining their assignment
4. WAIT for your reports to complete — do NOT do their work
5. When a report sends a shutdown_request, approve it: SendMessage with type="shutdown_response", recipient=their name
6. Make sure every task is marked as completed (status="completed") using TaskUpdate
{completion}

## Handling Unresponsive Teammates
The system will send you status checks when the team is idle. Pay attention to task status:
- Task is "in_progress" → the teammate is actively working. Be patient, do NOT interrupt.
- Task is "pending" and owner has been idle → the teammate has NOT started. Send them a follow-up message.
- If a teammate still doesn't respond after a follow-up, REASSIGN the task to another teammate using TaskUpdate (change owner).
- As a LAST RESORT only — if no teammates are available, you may do the work yourself."""

        # Regular teammate
        lead_name = self.lead_agent or "the leader"
        return f"""
## Teammate Responsibilities
You are a TEAMMATE. You report to "{lead_name}". You must:
1. When you receive a task, do the work described — produce the FULL deliverable
2. Send your COMPLETE work product back to "{lead_name}" via SendMessage. Include ALL of your output in the message — your lead cannot see your thinking, only what you explicitly send. Do NOT just say "done" or "task complete" — send the actual content (spec, analysis, design, code, etc.)
3. Mark your assigned task as completed using TaskUpdate (status="completed")
4. After completing ALL your tasks, request shutdown by sending: SendMessage with type="shutdown_request", recipient="{lead_name}", content="All tasks complete"
"""

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
                protocol = ProtocolService.parse_protocol_message(text)
                if protocol:
                    ptype = protocol.get("type", "unknown")
                    inbox_text_parts.append(f"[Protocol: {ptype} from {sender}]")
                else:
                    inbox_text_parts.append(f"[Message from {sender}]: {text}")
            inbox_block = "\n\n".join(inbox_text_parts)
            messages.append({"role": "user", "content": inbox_block})

        return messages

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return the tool definitions for this agent."""
        return TOOL_DEFINITIONS
