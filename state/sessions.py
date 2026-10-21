"""
Session persistence.

Each session is a self-contained directory:

    <project dir>/sessions/<session-id>/
        meta.json              session metadata for the /resume picker
        transcript.jsonl       append-only conversation log
        tasks/                 task store
        jobs/                  background job logs
        file-history/          /rewind checkpoints and file backups
        archives/              transcripts archived before /compact

Keeping every per-session artifact in one bundle means a session can be deleted,
exported or inspected with a single directory operation.
"""
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python

from core.paths import get_session_dir, get_sessions_dir

# Canonical file names inside a session bundle
TRANSCRIPT_NAME = "transcript.jsonl"
META_NAME = "meta.json"

# Container holding every session bundle for the current project
SESSIONS_DIR = get_sessions_dir()


def session_dir(session_id: str) -> Path:
    """
    Bundle directory for one session.
    """
    return get_session_dir(session_id)


def session_file(session_id: str) -> Path:
    return session_dir(session_id) / TRANSCRIPT_NAME


def meta_file(session_id: str) -> Path:
    return session_dir(session_id) / META_NAME


def new_session_id() -> str:
    return str(uuid.uuid4())


def read_meta(session_id: str) -> dict:
    """
    Read the session metadata; returns an empty dict when it has not been written
    yet (for example a session created but never used).
    """
    try:
        return json.loads(meta_file(session_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_meta(session_id: str, meta: dict) -> None:
    """
    Write the session metadata, creating the bundle directory on demand.
    """
    path = meta_file(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def note_prompt(session_id: str, text: str, model_name: str = "", cwd: str | None = None) -> None:
    """
    Record a user prompt in meta.json: the first prompt becomes the session title,
    and every prompt refreshes last_prompt.

    Writing here (rather than at session creation) keeps a bundle from being
    created for sessions the user never actually uses, and lets /resume show a
    title without opening the transcript.
    """
    meta = read_meta(session_id)
    if not meta:
        meta = {
            "id": session_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "cwd": cwd or os.getcwd(),
            "model": model_name,
            "title": "",
            "last_prompt": "",
        }
    text = " ".join(str(text).split())
    if not meta.get("title"):
        meta["title"] = text[:120]
    meta["last_prompt"] = text[:200]
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_meta(session_id, meta)


def append_messages(session_id: str, messages) -> None:
    """
    Append new messages from the current turn to the transcript (one per line).
    """
    path = session_file(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(to_jsonable_python(msg), ensure_ascii=False) + "\n")


def load_history(session_id: str) -> list:
    """
    Read the entire transcript and deserialize each JSON line into SDK message
    objects.
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
    Find the first genuine user prompt at the start of the file, used as the session summary.
    Recalled memories, @ mentions and job notifications may precede it, so system-reminder,
    task-notification and monitor-event content is skipped and extra lines are read as needed.
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
                if content.startswith(("<system-reminder>", "<task-notification>", "<monitor-event>")):
                    continue
                return content
    return "(empty session)"


def list_sessions() -> list:
    """
    Scan session bundles, returning (session_id, mtime, summary) sorted by
    modification time, newest first.

    The summary comes from meta.json; bundles without metadata fall back to the
    first user prompt in the transcript.
    """
    if not SESSIONS_DIR.is_dir():
        return []
    rows = []
    for path in SESSIONS_DIR.glob(f"*/{TRANSCRIPT_NAME}"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        session_id = path.parent.name
        meta = read_meta(session_id)
        summary = meta.get("title") or meta.get("last_prompt") or first_prompt(path)
        rows.append((session_id, datetime.fromtimestamp(mtime), summary))
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows


def rewrite_messages(session_id: str, messages) -> None:
    """
    Rewrite the transcript wholesale from the in-memory conversation history; used
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
    Archive the transcript before /compact rewrites it.

    The copy goes into the session's own archives/ directory, so it stays with the
    session it belongs to and never shows up in the /resume scan.
    """
    archive_dir = session_dir(session_id) / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    shutil.copy2(session_file(session_id), path)
    return path
