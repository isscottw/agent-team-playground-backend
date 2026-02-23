"""
Filesystem-based inbox store — mirrors Claude Code's inbox JSON format.

Each agent gets an inbox file at:
  data/sessions/{session_id}/inboxes/{agent_name}.json

The file is a JSON array of message objects.
"""

import json
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class InboxStore:
    """Manages per-agent inbox JSON files for a session."""

    def __init__(self, session_id: str, base_dir: str = "data"):
        self.session_id = session_id
        self.inbox_dir = Path(base_dir) / "sessions" / session_id / "inboxes"
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, agent: str) -> asyncio.Lock:
        if agent not in self._locks:
            self._locks[agent] = asyncio.Lock()
        return self._locks[agent]

    def _inbox_path(self, agent: str) -> Path:
        return self.inbox_dir / f"{agent}.json"

    def _read_raw(self, agent: str) -> List[Dict[str, Any]]:
        if not self.inbox_dir.exists():
            return []
        path = self._inbox_path(agent)
        if not path.exists():
            return []
        with open(path, "r") as f:
            return json.load(f)

    def _write_raw(self, agent: str, messages: List[Dict[str, Any]]) -> None:
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        path = self._inbox_path(agent)
        with open(path, "w") as f:
            json.dump(messages, f, indent=2)

    async def read_all(self, agent: str) -> List[Dict[str, Any]]:
        """Read all messages for an agent."""
        async with self._get_lock(agent):
            return self._read_raw(agent)

    async def read_unread(self, agent: str) -> List[Dict[str, Any]]:
        """Read unread messages and mark them as read."""
        async with self._get_lock(agent):
            messages = self._read_raw(agent)
            unread = [m for m in messages if not m.get("read", False)]
            if unread:
                for m in messages:
                    m["read"] = True
                self._write_raw(agent, messages)
            return unread

    async def append_message(
        self,
        agent: str,
        from_agent: str,
        text: str,
        summary: Optional[str] = None,
        color: Optional[str] = None,
        msg_type: str = "message",
    ) -> Dict[str, Any]:
        """Append a message to an agent's inbox. Returns the message dict."""
        message = {
            "from": from_agent,
            "text": text,
            "summary": summary or text[:80],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "color": color,
            "read": False,
        }
        async with self._get_lock(agent):
            messages = self._read_raw(agent)
            messages.append(message)
            self._write_raw(agent, messages)
        return message

    async def mark_read(self, agent: str, indices: Optional[List[int]] = None) -> int:
        """Mark messages as read. If indices is None, mark all as read. Returns count marked."""
        async with self._get_lock(agent):
            messages = self._read_raw(agent)
            count = 0
            for i, m in enumerate(messages):
                if (indices is None or i in indices) and not m.get("read", False):
                    m["read"] = True
                    count += 1
            if count > 0:
                self._write_raw(agent, messages)
            return count

    async def clear(self, agent: str) -> None:
        """Clear all messages for an agent."""
        async with self._get_lock(agent):
            self._write_raw(agent, [])

    def cleanup(self) -> None:
        """Remove the entire inbox directory for this session."""
        import shutil
        session_dir = self.inbox_dir.parent
        if session_dir.exists():
            shutil.rmtree(session_dir)
