"""
Constructs system-reminder text. Actual injection is hooked in agent/hooks.py.
"""
from tasks_store import TasksStore

from .file_state import ReadFileState


def _wrap(lines: list[str]) -> str:
    return f"<system-reminder>\n{chr(10).join(lines)}\n</system-reminder>"


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
    return _wrap(lines)


def build_task_reminder_text(store: TasksStore) -> str:
    """
    Constructs the task reminder text: a gentle nudge followed by the current task list.
    Sourced directly from the store, this reminder syncs disk state back into the prompt,
    sparing the model from manually invoking task_list.

    The opening sentence is also used by hooks.py to detect whether the previous message
    was already a reminder; keep it consistent and do not modify.
    """
    tasks = store.list()
    lines = [
        "The task tools haven't been used recently. If your current work involves multi-step tasks, consider using task_create to track progress, and task_update to maintain task state (in_progress when starting, completed when done); you can also clean up outdated tasks if appropriate. Only use these tools if relevant to your current work. This is just a gentle reminder—feel free to ignore it if unrelated.",
    ]
    if tasks:
        lines.append("")
        lines.append("Current tasks:")
        lines.append("")
        for t in tasks:
            lines.append(f"#{t.id}. [{t.status}] {t.subject}")
    return _wrap(lines)
