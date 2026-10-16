"""
Constructs system-reminder text. Actual injection is hooked in agent/hooks.py.
"""
from .file_state import ReadFileState


def build_reminder_text(state: ReadFileState) -> str | None:
    """
    Constructs the reminder text based on current session state; returns None if there are no updates to report.
    """
    stale = state.stale_paths()
    if not stale:
        return None
    lines = [
        "The following files were modified externally after you read them. Context contents may be outdated; call read_file before editing:",
    ]
    lines += [f"- {path}" for path in stale]
    body = "\n".join(lines)
    return f"<system-reminder>\n{body}\n</system-reminder>"
