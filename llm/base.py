"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """A tool call returned by an LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    stop_reason: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class that all LLM providers implement."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        api_key: str = "",
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Send a chat request and return a standardized LLMResponse."""
        ...
