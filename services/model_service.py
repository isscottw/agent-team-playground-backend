"""Model registry — knows every model across every provider."""

from typing import Any, Dict, List, Optional


MODELS: Dict[str, List[Dict[str, Any]]] = {
    "anthropic": [
        {
            "id": "claude-sonnet-4-6",
            "name": "Claude Sonnet 4.6",
            "provider": "anthropic",
            "description": "Fast, intelligent model for everyday tasks",
            "context_window": 200000,
            "supports_tools": True,
        },
        {
            "id": "claude-haiku-4-5",
            "name": "Claude Haiku 4.5",
            "provider": "anthropic",
            "description": "Fastest, most compact model for quick responses",
            "context_window": 200000,
            "supports_tools": True,
        },
    ],
    "openai": [
        {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "provider": "openai",
            "description": "OpenAI's flagship multimodal model",
            "context_window": 128000,
            "supports_tools": True,
        },
        {
            "id": "gpt-5",
            "name": "GPT-5",
            "provider": "openai",
            "description": "OpenAI's latest generation model",
            "context_window": 128000,
            "supports_tools": True,
        },
    ],
    "kimi": [
        {
            "id": "kimi-k2-0905-preview",
            "name": "Kimi K2",
            "provider": "kimi",
            "description": "Moonshot AI's K2 model via Anthropic-compatible API",
            "context_window": 128000,
            "supports_tools": True,
        },
    ],
    "ollama": [
        {
            "id": "llama3.2",
            "name": "Llama 3.2",
            "provider": "ollama",
            "description": "Meta's Llama 3.2 running locally via Ollama",
            "context_window": 128000,
            "supports_tools": True,
        },
        {
            "id": "qwen2.5",
            "name": "Qwen 2.5",
            "provider": "ollama",
            "description": "Alibaba's Qwen 2.5 running locally via Ollama",
            "context_window": 128000,
            "supports_tools": True,
        },
    ],
}

# Flat lookup: model_id -> model info
_MODEL_INDEX: Dict[str, Dict[str, Any]] = {}
for _provider, _model_list in MODELS.items():
    for _m in _model_list:
        _MODEL_INDEX[_m["id"]] = _m


class ModelService:

    def get_available_models(self) -> Dict[str, Any]:
        """Return all models grouped by provider."""
        return {"providers": MODELS}

    def get_models_flat(self) -> List[Dict[str, Any]]:
        """Return a flat list of all models."""
        return list(_MODEL_INDEX.values())

    def validate_model(self, model_id: str) -> bool:
        """Check whether a model ID is known."""
        return model_id in _MODEL_INDEX

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return model info dict or None."""
        return _MODEL_INDEX.get(model_id)

    def get_provider_for_model(self, model_id: str) -> Optional[str]:
        """Return the provider name for a model ID."""
        info = _MODEL_INDEX.get(model_id)
        return info["provider"] if info else None
