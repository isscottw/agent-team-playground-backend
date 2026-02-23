"""Ollama provider — uses httpx to talk to local Ollama server."""

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from llm.base import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.2:3b"
SUPPORTED_MODELS = ["llama3.2:3b", "deepseek-r1:8b", "gemma3:1b"]
BASE_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        api_key: str = "",
        model: Optional[str] = None,
    ) -> LLMResponse:
        model = model or DEFAULT_MODEL

        # Convert tools to Ollama format (OpenAI-compatible)
        ollama_tools = None
        if tools:
            ollama_tools = []
            for t in tools:
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", t.get("input_schema", {"type": "object", "properties": {}})),
                    },
                })

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if ollama_tools:
            payload["tools"] = ollama_tools

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        message = data.get("message", {})
        content = message.get("content", "")

        tool_calls: List[ToolCall] = []
        for i, tc in enumerate(message.get("tool_calls", [])):
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCall(
                id=f"ollama_{i}",
                name=func.get("name", ""),
                arguments=args,
            ))

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            stop_reason="stop",
        )
