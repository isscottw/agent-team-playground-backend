from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for a single agent in the team."""
    name: str = Field(..., description="Agent display name, e.g. 'researcher'")
    provider: str = Field(..., description="LLM provider: anthropic, openai, kimi, ollama")
    model: str = Field(..., description="Model ID, e.g. claude-sonnet-4-6")
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="System prompt that defines this agent's role",
    )
    connections: List[str] = Field(
        default_factory=list,
        description="Names of agents this agent can message",
    )


class SessionRequest(BaseModel):
    """Request body for POST /api/sessions."""
    agents: List[AgentConfig] = Field(..., min_length=1, description="Agent configurations")
    connections: List[List[str]] = Field(
        default_factory=list,
        description="Adjacency list of [agentA, agentB] pairs for messaging",
    )
    api_keys: Dict[str, str] = Field(
        default_factory=dict,
        description="Provider -> API key mapping, e.g. {'anthropic': 'sk-...'}",
    )


class SessionResponse(BaseModel):
    """Response body for POST /api/sessions."""
    session_id: str
    agents: List[str]
    status: str = "running"
