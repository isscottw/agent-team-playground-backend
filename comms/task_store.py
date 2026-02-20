"""
Filesystem-based task store — mirrors Claude Code's task JSON format.

Each task is stored at:
  data/sessions/{session_id}/tasks/{id}.json

A .highwatermark file tracks the next task ID.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional


class TaskStore:
    """Manages per-task JSON files for a session."""

    def __init__(self, session_id: str, base_dir: str = "data"):
        self.session_id = session_id
        self.task_dir = Path(base_dir) / "sessions" / session_id / "tasks"
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._hwm_path = self.task_dir / ".highwatermark"
        if not self._hwm_path.exists():
            self._hwm_path.write_text("0")

    def _next_id(self) -> str:
        current = int(self._hwm_path.read_text().strip())
        next_val = current + 1
        self._hwm_path.write_text(str(next_val))
        return str(next_val)

    def _task_path(self, task_id: str) -> Path:
        return self.task_dir / f"{task_id}.json"

    def _read_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        path = self._task_path(task_id)
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def _write_task(self, task_id: str, task: Dict[str, Any]) -> None:
        with open(self._task_path(task_id), "w") as f:
            json.dump(task, f, indent=2)

    async def create_task(
        self,
        subject: str,
        description: str = "",
        owner: Optional[str] = None,
        active_form: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new task and return it."""
        async with self._lock:
            task_id = self._next_id()
            task = {
                "id": task_id,
                "subject": subject,
                "description": description,
                "status": "pending",
                "owner": owner,
                "blockedBy": [],
                "blocks": [],
                "activeForm": active_form,
            }
            self._write_task(task_id, task)
            return task

    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a task by ID. Returns the updated task or None if not found."""
        async with self._lock:
            task = self._read_task(task_id)
            if task is None:
                return None

            allowed_fields = {"subject", "description", "status", "owner", "blockedBy", "blocks", "activeForm"}
            for key, value in updates.items():
                if key == "addBlockedBy" and isinstance(value, list):
                    existing = task.get("blockedBy", [])
                    task["blockedBy"] = list(set(existing + value))
                elif key == "addBlocks" and isinstance(value, list):
                    existing = task.get("blocks", [])
                    task["blocks"] = list(set(existing + value))
                elif key in allowed_fields:
                    task[key] = value

            self._write_task(task_id, task)
            return task

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a single task by ID."""
        async with self._lock:
            return self._read_task(task_id)

    async def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks, sorted by ID."""
        async with self._lock:
            tasks = []
            for path in sorted(self.task_dir.glob("*.json")):
                with open(path, "r") as f:
                    tasks.append(json.load(f))
            return tasks

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task file. Returns True if it existed."""
        async with self._lock:
            path = self._task_path(task_id)
            if path.exists():
                path.unlink()
                return True
            return False

    def cleanup(self) -> None:
        """Remove the entire task directory for this session."""
        import shutil
        if self.task_dir.exists():
            shutil.rmtree(self.task_dir)
