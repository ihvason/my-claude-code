"""
File-based long-term memory storage: each memory is a markdown file carrying
frontmatter, and MEMORY.md is their flat index.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.paths import get_project_dir

# The MEMORY.md index is injected into context in full every turn, so it must be
# capped: truncate once either limit is exceeded
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000

# Truncation limits applied when a single memory is injected on recall
MAX_MEMORY_LINES = 200
MAX_MEMORY_BYTES = 4096

# Frontmatter parsing only reads this many lines from the file head; the body is
# never read
FRONTMATTER_MAX_LINES = 30


def memory_dir() -> Path:
    """
    Memory directory for the current project, placed next to the session files so
    memory is isolated per project.
    """
    return get_project_dir() / "memory"


def index_path() -> Path:
    """Path of the MEMORY.md index."""
    return memory_dir() / "MEMORY.md"


def ensure_memory_dir() -> None:
    """Create the memory directory if it does not exist yet."""
    # Create the directory at startup so the system prompt can tell the model it
    # already exists and must not mkdir it
    memory_dir().mkdir(parents=True, exist_ok=True)


@dataclass
class MemoryHeader:
    """
    Metadata for a single memory file, parsed from its frontmatter; recall judges
    relevance from this alone without reading the body.
    """

    filename: str
    path: Path
    mtime: float
    name: str
    description: str
    type: str


def _parse_frontmatter(path: Path) -> dict:
    """
    Parse the name / description / type fields from the memory file's frontmatter.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = [f.readline() for _ in range(FRONTMATTER_MAX_LINES)]
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    fields = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def scan_memory_files() -> list[MemoryHeader]:
    """
    Scan all memory files (excluding the MEMORY.md index and empty files) and
    sort them by modification time, newest first.
    """
    directory = memory_dir()
    if not directory.exists():
        return []
    headers = []
    for path in directory.glob("*.md"):
        if path.name == "MEMORY.md":
            continue
        # Empty files are old memories cleared during consolidation; skip them
        if path.stat().st_size == 0:
            continue
        fields = _parse_frontmatter(path)
        headers.append(
            MemoryHeader(
                filename=path.name,
                path=path,
                mtime=path.stat().st_mtime,
                name=fields.get("name", path.stem),
                description=fields.get("description", ""),
                type=fields.get("type", ""),
            )
        )
    headers.sort(key=lambda h: h.mtime, reverse=True)
    return headers


def format_manifest(headers: list[MemoryHeader]) -> str:
    """
    Render memory metadata as a one-line-per-memory manifest for the recall and
    extraction prompts.
    """
    lines = []
    for h in headers:
        day = datetime.fromtimestamp(h.mtime).strftime("%Y-%m-%d")
        lines.append(f"- [{h.type}] {h.filename} ({day}): {h.description}")
    return "\n".join(lines)


def truncate_content(content: str, max_lines: int, max_bytes: int, notice: str) -> str:
    """
    Truncate by line count first, then by byte count (falling back to the last
    complete line); append a notice when anything was dropped.
    """
    lines = content.splitlines()
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    text = "\n".join(lines)
    if len(text.encode("utf-8")) > max_bytes:
        text = text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        if "\n" in text:
            text = text[: text.rfind("\n")]
        truncated = True
    if truncated:
        text += f"\n\n> {notice}"
    return text


def read_index() -> str | None:
    """
    Read the MEMORY.md index, truncating it once either limit is exceeded and
    appending a warning that reminds the model to keep the index concise.
    """
    path = index_path()
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not content.strip():
        return None
    return truncate_content(
        content,
        MAX_INDEX_LINES,
        MAX_INDEX_BYTES,
        "Warning: MEMORY.md exceeds the limit (200 lines / 25KB), so only part of it was loaded. Keep the index to one line per memory and move details into memory files.",
    )


def age_days(mtime: float) -> int:
    """Whole days elapsed since the given mtime, never negative."""
    return max(0, int((time.time() - mtime) // 86400))


def age_text(days: int) -> str:
    """Describe freshness in words: today / yesterday / N days ago."""
    # Describe freshness in natural language ("today" / "yesterday" / "N days ago");
    # the model is far more sensitive to relative days than to date arithmetic
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def staleness_text(days: int) -> str:
    """
    Freshness warning prepended when a memory older than one day is injected.
    """
    return (
        f"This memory was stored {days} days ago. Memories are snapshots taken when they were written, not live state — "
        "descriptions of code behavior and file locations may be stale; verify against the current code before treating them as fact."
    )


def is_memory_path(path: str) -> bool:
    """
    Whether a path falls inside the memory directory; shared by the permission
    allowlist and the background agent's write gate.
    """
    if not path:
        return False
    try:
        return Path(path).resolve().is_relative_to(memory_dir().resolve())
    except (OSError, ValueError):
        return False


def has_memory_writes(messages) -> bool:
    """
    Whether this turn's messages already contain a file write inside the memory
    directory; if the main conversation already recorded it, background extraction
    can be skipped.
    """
    for message in messages:
        for part in message.parts:
            if getattr(part, "part_kind", "") != "tool-call":
                continue
            if part.tool_name not in ("write_file", "edit_file"):
                continue
            if is_memory_path(str(part.args_as_dict().get("path", ""))):
                return True
    return False
