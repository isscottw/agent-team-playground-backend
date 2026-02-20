"""Token usage tracking across agents and sessions."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TokenTracker:
    """In-memory token usage tracker."""

    def __init__(self):
        # session_id -> agent_name -> {prompt_tokens, completion_tokens}
        self._usage: Dict[str, Dict[str, Dict[str, int]]] = {}

    def record(self, session_id: str, agent: str, prompt_tokens: int, completion_tokens: int) -> None:
        if session_id not in self._usage:
            self._usage[session_id] = {}
        if agent not in self._usage[session_id]:
            self._usage[session_id][agent] = {"prompt_tokens": 0, "completion_tokens": 0}
        self._usage[session_id][agent]["prompt_tokens"] += prompt_tokens
        self._usage[session_id][agent]["completion_tokens"] += completion_tokens

    def get_session_usage(self, session_id: str) -> Dict[str, Dict[str, int]]:
        return self._usage.get(session_id, {})

    def get_totals(self, session_id: str) -> Dict[str, int]:
        session = self._usage.get(session_id, {})
        total_prompt = sum(a["prompt_tokens"] for a in session.values())
        total_completion = sum(a["completion_tokens"] for a in session.values())
        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        }

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        result = []
        for sid, agents in self._usage.items():
            total = self.get_totals(sid)
            result.append({
                "session_id": sid,
                "agents": agents,
                **total,
            })
        return result

    def clear_session(self, session_id: str) -> None:
        self._usage.pop(session_id, None)
