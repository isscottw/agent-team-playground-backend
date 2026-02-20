from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in an agent's inbox — mirrors Claude Code's inbox format."""
    from_agent: str = Field(..., alias="from", description="Sender agent name")
    to_agent: Optional[str] = Field(None, description="Recipient agent name")
    text: str = Field(..., description="Message body (may contain serialized JSON)")
    summary: Optional[str] = Field(None, description="Short preview for UI")
    msg_type: str = Field(default="message", description="message | broadcast | protocol")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
    )
    color: Optional[str] = None
    read: bool = False

    class Config:
        populate_by_name = True


class Task(BaseModel):
    """A task in the shared task list."""
    id: str
    subject: str
    description: str = ""
    status: str = Field(default="pending", description="pending | in_progress | completed")
    owner: Optional[str] = None
    blocked_by: List[str] = Field(default_factory=list, alias="blockedBy")
    blocks: List[str] = Field(default_factory=list)
    active_form: Optional[str] = Field(None, alias="activeForm")

    class Config:
        populate_by_name = True


class ProtocolEvent(BaseModel):
    """A protocol-level event (JSON-in-JSON wrapper)."""
    type: str = Field(..., description="idle_notification | task_assignment | shutdown_request | shutdown_response")
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
    )
    from_agent: Optional[str] = Field(None, alias="from")

    class Config:
        populate_by_name = True


class SSEEvent(BaseModel):
    """Server-Sent Event payload sent to the frontend."""
    type: str = Field(..., description="Event type: agent_message, tool_call, tool_result, thinking, error, session_start, session_end, turn_start, turn_end")
    session_id: str
    agent: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
    )
