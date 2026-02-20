"""Factory function to get the right LLM provider by name."""

from llm.base import LLMProvider
from llm.anthropic_provider import AnthropicProvider
from llm.openai_provider import OpenAIProvider
from llm.kimi_provider import KimiProvider
from llm.ollama_provider import OllamaProvider

_PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "kimi": KimiProvider,
    "ollama": OllamaProvider,
}


def get_provider(provider_name: str) -> LLMProvider:
    """Return an LLMProvider instance for the given provider name."""
    cls = _PROVIDERS.get(provider_name.lower())
    if cls is None:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(_PROVIDERS.keys())}")
    return cls()
