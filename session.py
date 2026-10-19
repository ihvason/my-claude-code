"""
Session persistence: serialize conversation history into JSONL files,
supporting session scanning and restoration.
"""
import json
import uuid
from datetime import datetime
import shutil
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python

from utils import get_project_dir


# Per-project storage directory, isolated by working directory
STORAGE_DIR = get_project_dir() / "sessions"


def new_session_id() -> str:
    return str(uuid.uuid4())


def session_file(session_id: str) -> Path:
    return STORAGE_DIR / f"{session_id}.jsonl"


def append_messages(session_id: str, messages) -> None:
    """
    Append new messages from the current turn to the session file (one per line).
    """
    path = session_file(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(to_jsonable_python(msg), ensure_ascii=False) + "\n")


def load_history(session_id: str) -> list:
    """
    Read the entire session file and deserialize each JSON line into SDK message objects.
    """
    path = session_file(session_id)
    if not path.exists():
        return []
    
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
        
    return ModelMessagesTypeAdapter.validate_python(
        [json.loads(line) for line in lines]
    )


def first_prompt(path: Path) -> str:
    """
    Finds the first genuine user prompt from the beginning of the file to serve as the session summary.
    Recalled memories and @ mentions may precede it, so skip system-reminder content and read 
    additional lines as needed.
    """
    with open(path, encoding="utf-8") as f:
        for _ in range(20):
            line = f.readline()
            if not line:
                break
            msg = json.loads(line)
            for part in msg.get("parts", []):
                if part.get("part_kind") != "user-prompt":
                    continue
                content = str(part.get("content", ""))
                if content.startswith(("<system-reminder>", "<task-notification>")):
                    continue
                return content
    return "(empty session)"


def list_sessions() -> list:
    """
    Scan all session files in the current project, returning a list of
    (session_id, mtime, initial_user_prompt) sorted by modification time descending.
    """
    if not STORAGE_DIR.exists():
        return []
    files = sorted(
        STORAGE_DIR.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [
        (p.stem, datetime.fromtimestamp(p.stat().st_mtime), first_prompt(p))
        for p in files
    ]


def rewrite_messages(session_id: str, messages) -> None:
    """
    Rewrite the session file wholesale from the in-memory conversation history; used
    to persist after /rewind truncates the conversation.
    The full rewrite treats memory as the source of truth and does not rely on disk
    lines lining up one-to-one with in-memory entries.
    """
    path = session_file(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(to_jsonable_python(msg), ensure_ascii=False) + "\n")


def archive_session(session_id: str) -> Path:
    """
    Archive before compaction rewrites the session file: copy the current session
    file into the compact-history/ subdirectory and return the archive path.
    The archive keeps the full pre-compaction transcript so the model can read it
    back when the summary lacks detail.
    The subdirectory keeps archives out of list_sessions()'s *.jsonl scan, so they
    never appear in the /resume list.
    """
    archive_dir = get_project_dir() / "compact-history"
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"{session_id}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    shutil.copy2(session_file(session_id), path)
    return path
