"""
Shared path utilities.

Dependency-free and kept at the project root so that both the top-level modules
(session.py, tasks_store.py) and the agent/ package can import it without
creating an import cycle.
"""
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
