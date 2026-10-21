"""
ReadFileState: Session-level state shared across read_file, edit_file, and write_file.

Each session maintains an independent instance injected as a tool dependency,
which is replaced with a fresh instance during /new and /resume transitions.
It tracks whether a file was read, its cached content, and its mtime at the time of reading.
"""

import os


class ReadFileState:
    """
    Tracks read state for a session. Queried by edit_file and write_file before modifying files:
    prevents modifications if unread or modified externally since last read.
    """

    def __init__(self):
        # Maps absolute path to {"content": str, "timestamp": float, "offset": int | None, "limit": int | None}
        self._state: dict = {}

    def record(
        self, path: str, content: str, offset: int | None = None, limit: int | None = None
    ) -> None:
        """
        Records a file: caches its read content, current mtime, and read range (offset/limit).
        read_file passes non-None offsets; edit_file and write_file leave them as None to indicate
        a full-content write, bypassing deduplication.
        """
        self._state[os.path.abspath(path)] = {
            "content": content,
            "timestamp": os.path.getmtime(path),
            "offset": offset,
            "limit": limit,
        }

    def get(self, path: str):
        """
        Retrieves the record for a given path; returns None if never read.
        """
        return self._state.get(os.path.abspath(path))

    def stale_paths(self) -> list[str]:
        """
        Returns absolute paths of files whose disk mtime is newer than the recorded read time—
        indicating modifications outside this process after they were read.
        Internal edits via edit_file and write_file refresh the timestamp using record(),
        excluding the model's own changes.
        """
        stale = []
        for path, record in self._state.items():
            # Skip files that may have been deleted without flagging them as modified
            try:
                if os.path.getmtime(path) > record["timestamp"]:
                    stale.append(path)
            except OSError:
                continue
        return stale

    def paths(self) -> list[str]:
        """
        All recorded absolute paths, in first-read order.
        """
        return list(self._state)
