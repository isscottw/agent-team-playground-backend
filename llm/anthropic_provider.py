"""Anthropic Claude provider using the native Anthropic SDK."""

import json
import logging
from typing import Any, Dict, List, Optional

import anthropic

from llm.base import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6-20250514"
SUPPORTED_MODELS = ["claude-sonnet-4-6-20250514", "claude-haiku-4-5-20251001"]


class AnthropicProvider(LLMProvider):

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        api_key: str = "",
        model: Optional[str] = None,
    ) -> LLMResponse:
        model = model or DEFAULT_MODEL
        client = anthropic.AsyncAnthropic(api_key=api_key)

        # Separate system prompt from messages
        system_prompt = ""
        chat_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        # Ensure messages alternate user/assistant and start with user
        if not chat_messages or chat_messages[0]["role"] != "user":
            chat_messages.insert(0, {"role": "user", "content": "Begin."})

        # Convert tools to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = []
            for t in tools:
                anthropic_tools.append({
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", t.get("input_schema", {"type": "object", "properties": {}})),
                })

        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": chat_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = await client.messages.create(**kwargs)

        # Parse response
        content_text = ""
        tool_calls: List[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else json.loads(block.input),
                ))

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            stop_reason=response.stop_reason,
        )
