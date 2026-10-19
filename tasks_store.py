"""
Task persistence layer: each task is stored as an individual JSON file under
<project dir>/sessions/<session_id>/tasks/<id>.json.
A .highwatermark file tracks the highest allocated task ID to prevent ID reuse after deletions.

All state is loaded into memory once during __init__; mutations synchronously flush to disk
and update the in-memory cache, keeping list() and get() free of disk I/O.
The TUI panel invokes list() on every spinner tick (~100ms); without caching, this would incur per-frame disk access.

Designed for single-process use without file locking; concurrent writes across multiple processes
to the same session_id directory will cause race conditions.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterator, Literal

from utils import get_session_dir


# Only these three statuses are persisted to disk; 'deleted' is a pseudo-status signal during update that triggers deletion instead of serialization.
TaskStatus = Literal["pending", "in_progress", "completed"]


@dataclass
class Task:
    id: str
    subject: str
    description: str
    active_form: str | None = None
    status: TaskStatus = "pending"
    blocks: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)


_HIGHWATERMARK = ".highwatermark"


class TasksStore:
    """
    Session-isolated, file-backed task store providing create, list, get, update, and delete operations.
    Every mutation writes synchronously to disk, enabling state restoration on subsequent /resume calls.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        # Lives inside the session bundle; the directory is created on first write so
        # a session that never creates a task leaves nothing behind
        self.dir = get_session_dir(session_id) / "tasks"
        # Populate in-memory cache from disk once during initialization; subsequent read/write operations update memory and flush to disk synchronously.
        self._tasks: dict[str, Task] = {}
        for path in self._json_paths():
            try:
                task = Task(**json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            self._tasks[task.id] = task
        # Load highwatermark once; subsequent create/delete operations maintain an in-memory integer.
        self._highwatermark = self._read_highwatermark()

    # ---------- Public API ----------

    def create(self, subject: str, description: str, active_form: str | None = None) -> str:
        # id = max(current IDs, highwatermark) + 1; deleted IDs are never reused.
        current_max = max((int(tid) for tid in self._tasks), default=0)
        next_id = max(current_max, self._highwatermark) + 1
        task = Task(
            id=str(next_id),
            subject=subject,
            description=description,
            active_form=active_form,
        )
        self._tasks[task.id] = task
        self._write(task)
        return task.id

    def list(self) -> list[Task]:
        # Return tasks sorted by ascending numeric ID to match UI and task_list output; served from memory without disk I/O.
        return sorted(self._tasks.values(), key=lambda t: int(t.id))

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def update(self, task_id: str, **fields) -> Task:
        # add_blocks and add_blocked_by have append semantics and are handled separately; remaining fields overwrite via dataclasses.replace for type safety.
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"task #{task_id} not found")
        add_blocks = fields.pop("add_blocks", None)
        add_blocked_by = fields.pop("add_blocked_by", None)
        new_blocks = list(dict.fromkeys(task.blocks + list(add_blocks))) if add_blocks else task.blocks
        new_blocked_by = list(dict.fromkeys(task.blocked_by + list(add_blocked_by))) if add_blocked_by else task.blocked_by
        updated = replace(task, **fields, blocks=new_blocks, blocked_by=new_blocked_by)
        self._tasks[task_id] = updated
        self._write(updated)
        return updated

    def delete(self, task_id: str) -> None:
        if task_id not in self._tasks:
            raise KeyError(f"task #{task_id} not found")
        # Update highwatermark before deletion to prevent subsequent allocations from reusing deleted IDs.
        current_max = max((int(tid) for tid in self._tasks), default=0)
        if current_max > self._highwatermark:
            self._highwatermark = current_max
            self._write_highwatermark(self._highwatermark)
        del self._tasks[task_id]
        self._task_path(task_id).unlink()

    def snapshot(self) -> dict:
        """
        Serializable snapshot of the whole task state, used by /rewind checkpoints.
        """
        return {
            "tasks": [asdict(t) for t in self.list()],
            "highwatermark": self._highwatermark,
        }

    def restore(self, snapshot: dict) -> None:
        """
        Roll the store back to a snapshot taken earlier by snapshot().
        Both the in-memory cache and the on-disk directory are rewritten, so tasks
        created after the snapshot disappear and tasks deleted since come back.
        """
        self._tasks = {t["id"]: Task(**t) for t in snapshot.get("tasks", [])}
        self._highwatermark = int(snapshot.get("highwatermark", 0))
        for path in list(self._json_paths()):
            path.unlink(missing_ok=True)
        for task in self._tasks.values():
            self._write(task)
        self._write_highwatermark(self._highwatermark)

    # ---------- Internal Utilities ----------

    def _json_paths(self) -> Iterator[Path]:
        # Helper for __init__; yields only *.json files, skipping hidden files like .highwatermark.
        if not self.dir.is_dir():
            return
        for path in self.dir.iterdir():
            if path.suffix == ".json" and not path.name.startswith("."):
                yield path

    def _task_path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.json"

    def _highwatermark_path(self) -> Path:
        return self.dir / _HIGHWATERMARK

    def _read_highwatermark(self) -> int:
        path = self._highwatermark_path()
        if not path.exists():
            return 0
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0

    def _ensure_dir(self) -> None:
        # Created on demand so an unused session leaves no empty task directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def _write_highwatermark(self, value: int) -> None:
        self._ensure_dir()
        self._highwatermark_path().write_text(str(value), encoding="utf-8")

    def _write(self, task: Task) -> None:
        self._ensure_dir()
        self._task_path(task.id).write_text(
            json.dumps(asdict(task), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
