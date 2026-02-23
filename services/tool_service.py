"""
Tool definitions and executor for agent tools.

Provides the five core team tools (SendMessage, TaskCreate, TaskUpdate,
TaskList, TaskGet) and executes tool calls against the JSON stores.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from comms.json_store import InboxStore
from comms.task_store import TaskStore
from services.protocol_service import ProtocolService

logger = logging.getLogger(__name__)


# ---------- Tool schemas (Anthropic / OpenAI compatible) ----------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "SendMessage",
        "description": "Send a message to another agent on the team.",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["message", "broadcast", "shutdown_request", "shutdown_response", "plan_approval_request", "plan_approval_response"],
                    "description": "message = DM, broadcast = all, or protocol message types",
                },
                "recipient": {
                    "type": "string",
                    "description": "Name of the recipient agent (required for type=message)",
                },
                "content": {
                    "type": "string",
                    "description": "The message text",
                },
                "summary": {
                    "type": "string",
                    "description": "Short 5-10 word summary",
                },
                "request_id": {
                    "type": "string",
                    "description": "Request ID (for shutdown_response, plan_approval_request/response)",
                },
                "approve": {
                    "type": "boolean",
                    "description": "Whether to approve (for shutdown_response, plan_approval_response)",
                },
            },
            "required": ["type"],
        },
    },
    {
        "name": "TaskCreate",
        "description": "Create a new task in the shared task list.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Brief task title"},
                "description": {"type": "string", "description": "Detailed description"},
                "activeForm": {"type": "string", "description": "Present continuous form for spinner"},
                "metadata": {"type": "object", "description": "Arbitrary metadata to attach to the task"},
            },
            "required": ["subject", "description"],
        },
    },
    {
        "name": "TaskUpdate",
        "description": "Update an existing task's status, owner, or details.",
        "parameters": {
            "type": "object",
            "properties": {
                "taskId": {"type": "string", "description": "ID of the task to update"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "deleted"]},
                "owner": {"type": "string"},
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "activeForm": {"type": "string"},
                "addBlockedBy": {"type": "array", "items": {"type": "string"}},
                "addBlocks": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object", "description": "Metadata keys to merge (set key to null to delete)"},
            },
            "required": ["taskId"],
        },
    },
    {
        "name": "TaskList",
        "description": "List all tasks in the shared task list.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "TaskGet",
        "description": "Get a single task by ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "taskId": {"type": "string", "description": "The task ID"},
            },
            "required": ["taskId"],
        },
    },
]


class ToolExecutor:
    """Executes tool calls against the JSON stores and returns results."""

    def __init__(
        self,
        inbox_store: InboxStore,
        task_store: TaskStore,
        agent_name: str,
        team_agents: List[str],
        on_message_sent: Optional[Callable] = None,
        on_task_assigned: Optional[Callable] = None,
        on_task_completed: Optional[Callable] = None,
        on_task_changed: Optional[Callable] = None,
    ):
        self.inbox = inbox_store
        self.tasks = task_store
        self.agent_name = agent_name
        self.team_agents = team_agents
        self.on_message_sent = on_message_sent  # callback(to_agent, message_dict)
        self.on_task_assigned = on_task_assigned  # callback(owner, task_dict)
        self.on_task_completed = on_task_completed  # callback(task_dict)
        self.on_task_changed = on_task_changed  # callback(task_dict)

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call and return the result as a string."""
        try:
            handler = getattr(self, f"_handle_{tool_name}", None)
            if handler is None:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            result = await handler(arguments)
            return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return json.dumps({"error": str(e)})

    async def _handle_SendMessage(self, args: Dict[str, Any]) -> Dict[str, Any]:
        msg_type = args.get("type", "message")
        content = args.get("content", "")
        summary = args.get("summary", content[:80] if content else "")

        # --- Protocol message types ---
        if msg_type == "shutdown_request":
            recipient = args.get("recipient", "")
            if not recipient:
                return {"error": "recipient is required for shutdown_request"}
            proto = ProtocolService.create_shutdown_request(self.agent_name, content or "requested by agent", target=recipient)
            await self.inbox.append_message(agent=recipient, from_agent=self.agent_name, text=proto["text"], summary=proto["summary"])
            if self.on_message_sent:
                await self.on_message_sent(recipient, proto)
            return {"status": "shutdown_request_sent", "to": recipient}

        if msg_type == "shutdown_response":
            recipient = args.get("recipient", "")
            if not recipient:
                return {"error": "recipient is required for shutdown_response"}
            proto = ProtocolService.create_shutdown_approved(self.agent_name)
            await self.inbox.append_message(agent=recipient, from_agent=self.agent_name, text=proto["text"], summary=proto["summary"])
            if self.on_message_sent:
                await self.on_message_sent(recipient, proto)
            return {"status": "shutdown_approved_sent", "to": recipient}

        if msg_type == "plan_approval_request":
            recipient = args.get("recipient", "")
            request_id = args.get("request_id", "")
            if not recipient or not request_id:
                return {"error": "recipient and request_id are required for plan_approval_request"}
            proto = ProtocolService.create_plan_approval_request(self.agent_name, request_id, content)
            await self.inbox.append_message(agent=recipient, from_agent=self.agent_name, text=proto["text"], summary=proto["summary"])
            if self.on_message_sent:
                await self.on_message_sent(recipient, proto)
            return {"status": "plan_approval_request_sent", "to": recipient, "request_id": request_id}

        if msg_type == "plan_approval_response":
            recipient = args.get("recipient", "")
            request_id = args.get("request_id", "")
            approve = args.get("approve", False)
            if not recipient or not request_id:
                return {"error": "recipient and request_id are required for plan_approval_response"}
            proto = ProtocolService.create_plan_approval_response(self.agent_name, request_id, approve, content)
            await self.inbox.append_message(agent=recipient, from_agent=self.agent_name, text=proto["text"], summary=proto["summary"])
            if self.on_message_sent:
                await self.on_message_sent(recipient, proto)
            return {"status": "plan_approval_response_sent", "to": recipient, "request_id": request_id, "approve": approve}

        # --- Standard message types ---
        if msg_type == "broadcast":
            sent_to = []
            for agent in self.team_agents:
                if agent != self.agent_name:
                    msg = await self.inbox.append_message(
                        agent=agent,
                        from_agent=self.agent_name,
                        text=content,
                        summary=summary,
                    )
                    sent_to.append(agent)
                    if self.on_message_sent:
                        await self.on_message_sent(agent, msg)
            return {"status": "broadcast_sent", "sent_to": sent_to}
        else:
            # type == "message" (default)
            recipient = args.get("recipient", "")
            if not recipient:
                return {"error": "recipient is required for type=message"}
            msg = await self.inbox.append_message(
                agent=recipient,
                from_agent=self.agent_name,
                text=content,
                summary=summary,
            )
            if self.on_message_sent:
                await self.on_message_sent(recipient, msg)
            return {"status": "message_sent", "to": recipient}

    async def _handle_TaskCreate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        task = await self.tasks.create_task(
            subject=args["subject"],
            description=args.get("description", ""),
            active_form=args.get("activeForm"),
            metadata=args.get("metadata"),
        )
        if self.on_task_changed:
            await self.on_task_changed(task)
        return task

    async def _handle_TaskUpdate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        task_id = args.pop("taskId")
        owner = args.get("owner")
        new_status = args.get("status")
        task = await self.tasks.update_task(task_id, args)
        if task is None:
            return {"error": f"Task {task_id} not found"}
        if self.on_task_changed and not task.get("deleted"):
            await self.on_task_changed(task)
        if self.on_task_assigned and owner and not task.get("deleted"):
            await self.on_task_assigned(task["owner"], task)
        if self.on_task_completed and new_status == "completed" and not task.get("deleted"):
            await self.on_task_completed(task)
        return task

    async def _handle_TaskList(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        return await self.tasks.list_tasks()

    async def _handle_TaskGet(self, args: Dict[str, Any]) -> Dict[str, Any]:
        task_id = args["taskId"]
        task = await self.tasks.get_task(task_id)
        if task is None:
            return {"error": f"Task {task_id} not found"}
        return task
