"""
SSEBroadcaster — manages Server-Sent Event connections per session.

Each session can have multiple SSE listeners (e.g. multiple browser tabs).
Events are broadcast to all listeners for a given session.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Set

logger = logging.getLogger(__name__)


class SSEBroadcaster:
    """Manages SSE event queues per session."""

    def __init__(self):
        # session_id -> set of asyncio.Queue
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """Create and return a new event queue for a session."""
        if session_id not in self._subscribers:
            self._subscribers[session_id] = set()
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[session_id].add(queue)
        logger.info(f"SSE subscriber added for session {session_id} (total: {len(self._subscribers[session_id])})")
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        subs = self._subscribers.get(session_id)
        if subs:
            subs.discard(queue)
            if not subs:
                del self._subscribers[session_id]

    async def broadcast(self, session_id: str, event: Dict[str, Any]) -> None:
        """Send an event to all subscribers of a session."""
        subs = self._subscribers.get(session_id)
        if not subs:
            return
        for queue in list(subs):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"SSE queue full for session {session_id}, dropping event")

    async def event_generator(self, session_id: str, queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted strings from a subscriber queue.

        Used directly as the body of a StreamingResponse.
        """
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"

                    # Stop streaming on session_end
                    if event.get("type") == "session_end":
                        break
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield f": keepalive {datetime.utcnow().isoformat()}\n\n"
        finally:
            self.unsubscribe(session_id, queue)

    def cleanup(self, session_id: str) -> None:
        """Remove all subscribers for a session."""
        self._subscribers.pop(session_id, None)
