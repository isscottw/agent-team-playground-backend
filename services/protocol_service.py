"""
Protocol message helpers — handles the JSON-in-JSON pattern
used by Claude Code agent teams.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional


class ProtocolService:

    @staticmethod
    def create_message(
        from_agent: str,
        text: str,
        summary: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a plain inbox message dict."""
        return {
            "from": from_agent,
            "text": text,
            "summary": summary or text[:80],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "color": color,
            "read": False,
        }

    @staticmethod
    def create_dm(
        from_agent: str,
        to_agent: str,
        content: str,
        summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a direct message."""
        return ProtocolService.create_message(
            from_agent=from_agent,
            text=content,
            summary=summary or content[:80],
        )

    @staticmethod
    def create_broadcast(
        from_agent: str,
        content: str,
        summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a broadcast message (same format, sent to all inboxes)."""
        return ProtocolService.create_message(
            from_agent=from_agent,
            text=content,
            summary=summary or content[:80],
        )

    @staticmethod
    def create_protocol_event(
        event_type: str,
        from_agent: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a JSON-in-JSON protocol event message.

        The resulting message has `text` set to a serialized JSON string
        containing {type, from, timestamp, ...data}.
        """
        payload = {
            "type": event_type,
            "from": from_agent,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **(data or {}),
        }
        return ProtocolService.create_message(
            from_agent=from_agent,
            text=json.dumps(payload),
            summary=f"{event_type} from {from_agent}",
        )

    @staticmethod
    def create_shutdown_request(from_agent: str, reason: str = "") -> Dict[str, Any]:
        return ProtocolService.create_protocol_event(
            "shutdown_request",
            from_agent,
            {"reason": reason},
        )

    @staticmethod
    def create_idle_notification(from_agent: str, reason: str = "available") -> Dict[str, Any]:
        return ProtocolService.create_protocol_event(
            "idle_notification",
            from_agent,
            {"idleReason": reason},
        )

    @staticmethod
    def parse_protocol_message(text: str) -> Optional[Dict[str, Any]]:
        """Try to parse JSON-in-JSON from a message text field.

        Returns the parsed dict if the text is valid JSON with a `type` field,
        otherwise returns None (meaning it's a plain text message).
        """
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "type" in parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return None
