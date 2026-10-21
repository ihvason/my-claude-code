"""
Task tools: task_create, task_list, task_get, and task_update.
All four interact with the underlying file-backed TasksStore via ctx.deps.tasks_store.

The model uses these tools to break down multi-step tasks into pending tasks, mark the
active one as in_progress, and transition it to completed upon completion.
"""

from dataclasses import asdict
from typing import Literal

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from ..deps import AgentDeps

# The tool layer accepts 4 status literals: the first 3 persist to disk, while 'deleted'
# acts as a pseudo-status signal that routes to store.delete() to remove the file directly.
TaskUpdateStatus = Literal["pending", "in_progress", "completed", "deleted"]

# task_list returns only these summary fields to pick the next task, omitting descriptions
# to prevent polluting the prompt context with repetitive large text blocks.
_SUMMARY_FIELDS = ("id", "subject", "status", "blocked_by")


def task_create(
    ctx: RunContext[AgentDeps],
    subject: str,
    description: str,
    active_form: str | None = None,
) -> dict:
    """
    Creates a structured task in the task list. Use this proactively for complex tasks
    (3+ steps), when the user provides a list of todos, or to clearly demonstrate progress.
    Avoid creating tasks for trivial single-step actions or conversational replies.

    New tasks default to status='pending'. Switch to 'in_progress' via task_update before starting,
    and mark as 'completed' immediately upon finishing.

    Args:
        subject: Brief imperative title (e.g., "Fix login bug").
        description: Detailed requirements for the task.
        active_form: Continuous tense phrase shown in the status bar while in_progress
                     (e.g., "Fixing login bug"); defaults to subject if omitted.
    """
    store = ctx.deps.tasks_store
    task_id = store.create(subject=subject, description=description, active_form=active_form)
    return {"id": task_id, "subject": subject}


def task_list(ctx: RunContext[AgentDeps]) -> list[dict]:
    """
    Returns a concise summary of all tasks to track pending, active, and completed items.
    Typically called after completing a task to determine what to work on next.

    Prefer addressing tasks in ascending ID order, as earlier tasks generally establish
    context for subsequent ones.
    """
    return [{k: asdict(t)[k] for k in _SUMMARY_FIELDS} for t in ctx.deps.tasks_store.list()]


def task_get(ctx: RunContext[AgentDeps], task_id: str) -> dict:
    """
    Retrieves the full details of a single task: subject, description, status, blocks, and blocked_by.
    Invoke before starting a task to inspect full requirements, or to resolve dependencies.

    Args:
        task_id: The ID of the task to retrieve.
    """
    task = ctx.deps.tasks_store.get(task_id)
    if task is None:
        raise ModelRetry(f"Task #{task_id} does not exist. Call task_list to see available tasks.")
    return asdict(task)


def task_update(
    ctx: RunContext[AgentDeps],
    task_id: str,
    status: TaskUpdateStatus | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
) -> dict:
    """
    Updates fields on an existing task. Most commonly used to transition status to 'in_progress'
    before starting work, and 'completed' immediately upon finishing—do not leave finished tasks
    in 'in_progress'.

    Passing status='deleted' permanently removes the task file from disk (no record retained).
    Useful for tasks created in error or no longer needed.

    If implementation is incomplete, tests are failing, or unhandled errors occur, keep the task
    as 'in_progress' and notify the user about the blocker.

    Args:
        task_id: The ID of the task to update.
        status: 'pending', 'in_progress', 'completed', or 'deleted'. 'deleted' triggers immediate removal.
        subject: New title; omit if unchanged.
        description: New description.
        active_form: New continuous tense phrase.
        add_blocks: Task IDs blocked by this task (appended, not overwritten).
        add_blocked_by: Task IDs this task depends on (appended, not overwritten).
    """
    store = ctx.deps.tasks_store
    task = store.get(task_id)
    if task is None:
        raise ModelRetry(f"Task #{task_id} does not exist. Call task_list to see available tasks.")

    # 'deleted' is a pseudo-status; remove the file immediately and skip remaining updates.
    if status == "deleted":
        store.delete(task_id)
        return {"success": True, "task_id": task_id, "deleted": True}

    # Collect explicitly provided fields (None indicates no change) to persist to store
    candidates = {
        "status": status,
        "subject": subject,
        "description": description,
        "active_form": active_form,
        "add_blocks": add_blocks,
        "add_blocked_by": add_blocked_by,
    }
    fields = {k: v for k, v in candidates.items() if v is not None}

    if not fields:
        raise ModelRetry(
            "task_update received no fields to modify. Provide at least one of: "
            "status, subject, description, active_form, add_blocks, or add_blocked_by."
        )

    updated = store.update(task_id, **fields)
    return {
        "success": True,
        "task_id": task_id,
        "updated_fields": list(fields.keys()),
        "task": asdict(updated),
    }
