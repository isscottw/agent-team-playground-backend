"""OpenAI provider using the openai SDK."""

import json
import logging
from typing import Any, Dict, List, Optional

import openai

from llm.base import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"
SUPPORTED_MODELS = ["gpt-4o", "gpt-5"]


class OpenAIProvider(LLMProvider):

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        api_key: str = "",
        model: Optional[str] = None,
    ) -> LLMResponse:
        model = model or DEFAULT_MODEL
        client = openai.AsyncOpenAI(api_key=api_key)

        # Convert tools to OpenAI function calling format
        openai_tools = None
        if tools:
            openai_tools = []
            for t in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", t.get("input_schema", {"type": "object", "properties": {}})),
                    },
                })

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        # Parse tool calls
        tool_calls: List[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                ))

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            stop_reason=choice.finish_reason,
        )
