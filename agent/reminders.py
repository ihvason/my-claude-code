"""
Constructs system-reminder text. Actual injection is hooked in agent/hooks.py.
"""

from html import escape as escape_xml

from rich.markup import escape

from core.console import console
from state.jobs import JobRegistry
from state.tasks import TasksStore

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


def build_job_reminder_text(registry: JobRegistry | None) -> str | None:
    """
    Render finished-but-unnotified background jobs as <task-notification> blocks;
    returns None when there are no new completions.
    """
    if registry is None:
        return None
    jobs = registry.pop_unnotified()
    if not jobs:
        return None
    blocks = []
    for job in jobs:
        # Print a completion line for the user while also notifying the model
        color = "green" if job.status == "completed" else "red"
        console.print(f"[{color}]●[/] {escape(job.summary())}\n")
        summary = f"{job.summary()}, read the output file with read_file"
        # Agent jobs carry a final report in result; include it so the main agent does
        # not have to read the log
        result = f"<result>{job.result}</result>\n" if job.result else ""
        blocks.append(
            "<task-notification>\n"
            f"<task-id>{job.id}</task-id>\n"
            f"<task-type>{job.kind}</task-type>\n"
            f"<output-file>{job.log_path}</output-file>\n"
            f"<status>{job.status}</status>\n"
            f"<summary>{summary}</summary>\n"
            f"{result}"
            "</task-notification>"
        )
    return "\n\n".join(blocks)


def build_monitor_event_text(registry: JobRegistry | None) -> str | None:
    """
    Render the event lines each monitor has buffered since the last call as
    <monitor-event> notices; returns None when there is nothing new.
    """
    if registry is None:
        return None
    popped = registry.pop_events()
    if not popped:
        return None
    blocks = []
    for job, events, dropped in popped:
        notice = f"monitor {escape(job.description)!r}: {len(events)} new event(s)"
        if dropped:
            notice += f" ({dropped} more omitted)"
        console.print(f"[cyan]●[/] {notice}\n")
        omitted = (
            (
                f"<dropped>{dropped} event(s) were omitted because output came too fast; "
                "remount with a tighter filter if you need them</dropped>\n"
            )
            if dropped
            else ""
        )
        description = escape_xml(job.description, quote=False)
        event_text = escape_xml(chr(10).join(events), quote=False)
        blocks.append(
            "<monitor-event>\n"
            f"<task-id>{job.id}</task-id>\n"
            f"<description>{description}</description>\n"
            f"<events>\n{event_text}\n</events>\n"
            f"{omitted}"
            "</monitor-event>"
        )
    return "\n\n".join(blocks)


def build_job_notifications(registry: JobRegistry | None) -> str | None:
    """
    The registry's single notification entry point: finished jobs plus monitor events, so
    both delivery paths ask one function.
    """
    parts = [build_job_reminder_text(registry), build_monitor_event_text(registry)]
    return "\n\n".join(p for p in parts if p) or None
