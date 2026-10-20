"""
Shared path utilities.

Dependency-free and kept at the project root so that both the top-level modules
(session.py, tasks_store.py) and the agent/ package can import it without
creating an import cycle.
"""
import re
from functools import cache
from pathlib import Path

# Files or directories that identify a project root; the first match wins
ROOT_MARKERS = (".git", "pyproject.toml", "uv.lock")


@cache
def get_project_root(start: str | Path | None = None) -> Path:
    """
    Return the absolute path of the project root.

    Walks upward from `start` and returns the first directory that contains any
    of ROOT_MARKERS. The default anchors on this file's own directory rather than
    on the current working directory, so the result stays the same no matter
    where the process was launched from.

    The result is cached: the project root does not change while the process runs,
    and this keeps repeated calls free.

    Args:
        start: file or directory to begin the search from; defaults to this
            file's directory.

    Returns:
        The project root, or the starting directory if no marker was found.
    """
    base = Path(start).resolve() if start is not None else Path(__file__).resolve().parent
    if base.is_file():
        base = base.parent
    for candidate in (base, *base.parents):
        if any((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate
    return base


def encode_project_path(path: str | Path) -> str:
    """
    Encode an absolute path into a single filesystem-safe directory name.

    Flattens the path with dashes, mirroring the Claude Code convention:
    /Users/quentin/work/foo -> -Users-quentin-work-foo
    """
    return re.sub(r"[^A-Za-z0-9_-]", "-", str(Path(path).resolve()))


@cache
def get_data_dir() -> Path:
    """
    Root of this application's on-disk data: <project root>/.my-claude-code.
    """
    return get_project_root() / ".my-claude-code"


@cache
def get_project_dir() -> Path:
    """
    Data directory for the project currently being worked on:
    <data dir>/projects/<encoded project root>.

    The project root is discovered by walking up from the working directory, so
    starting the app from a subdirectory still maps to the same project and does
    not fork a separate data namespace.
    """
    return get_data_dir() / "projects" / encode_project_path(get_project_root(Path.cwd()))


@cache
def get_sessions_dir() -> Path:
    """
    Container for per-session bundles: <project dir>/sessions.
    """
    return get_project_dir() / "sessions"


def get_session_dir(session_id: str) -> Path:
    """
    Bundle directory for one session. Every per-session artifact (transcript,
    metadata, tasks, jobs, file history, archives) lives inside it, so a session
    can be deleted, backed up or moved as a single directory.
    """
    return get_sessions_dir() / session_id


def find_agents_md(start: str | Path | None = None) -> Path | None:
    """
    Return the project-level AGENTS.md, or None when the project has none.

    The project root is discovered by walking upward from `start` (default: the
    current working directory), so launching the app from a subdirectory still finds
    the file kept at the repository root. When no root marker exists above `start`,
    the search keeps walking upward, which covers projects without .git,
    pyproject.toml or uv.lock.

    Args:
        start: file or directory to begin the search from; defaults to the current
            working directory.

    Returns:
        Absolute path of the AGENTS.md file, or None if it does not exist.
    """
    base = Path(start).resolve() if start is not None else Path.cwd().resolve()
    if base.is_file():
        base = base.parent
    root = get_project_root(base)
    # With a marker-based root the project-level file can only live there; without one,
    # the real root may still sit further up the tree
    marked = any((root / marker).exists() for marker in ROOT_MARKERS)
    candidates = (root,) if marked else (root, *base.parents)
    for candidate in candidates:
        path = candidate / "AGENTS.md"
        if path.is_file():
            return path
    return None
