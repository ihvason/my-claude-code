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

    def record(self, path: str, content: str, offset: int | None = None, limit: int | None = None) -> None:
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
