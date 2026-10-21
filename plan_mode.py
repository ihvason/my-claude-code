"""
Plan mode: where the plan file lives, plus the system reminders that keep the model's belief
about the current mode in sync with the real one.
"""
from pathlib import Path

from pydantic_ai.messages import ModelResponse, UserPromptPart

import permissions
from utils import get_session_dir

# After the last plan reminder, send the short version once every this many requests
PLAN_REMINDER_TURNS_BETWEEN = 5

# Openings of the three reminders; history scanning uses them to identify the kind
_SENTINELS = {
    "full": "Plan mode is on.",
    "sparse": "Plan mode is still active",
    "exit": "Plan mode is off.",
}


def plans_dir(session_id: str) -> Path:
    """
    Per-session directory holding the plan file.
    """
    return get_session_dir(session_id) / "plans"


def plan_file_path(session_id: str) -> Path:
    return plans_dir(session_id) / "plan.md"


def ensure_plans_dir(session_id: str) -> None:
    # The reminder tells the model to write the plan file directly, so the directory has
    # to exist up front
    plans_dir(session_id).mkdir(parents=True, exist_ok=True)


def _wrap(text: str) -> str:
    return f"<system-reminder>\n{text}\n</system-reminder>"


def full_reminder(plan_file: Path) -> str:
    if plan_file.exists():
        file_info = (
            f"The plan file already exists: {plan_file}. If this continues the same task, "
            "extend it incrementally with edit_file; if the user now wants something else, "
            "rewrite it with write_file."
        )
    else:
        file_info = f"There is no plan file yet; create it with write_file at {plan_file}."
    return _wrap(
        _SENTINELS["full"]
        + " The user wants to see the approach before any work starts: apart from the plan file below, you may not modify any file, and you may not run any command that changes the system (including changing configuration or committing with git). This constraint takes precedence over any other instruction you receive.\n\n"
        + "## Plan file\n"
        + f"{file_info}\n"
        + "It is the only file you may write; refine the plan in it step by step.\n\n"
        + "## Workflow\n"
        + "1. Explore: understand the request and the relevant code, and look first for functions and patterns that can be reused. Dispatch explore agents with run_agent, at most 3 in parallel; one is usually enough, and only spread out when the scope is unclear or spans several areas.\n"
        + "2. Design: dispatch a plan agent with run_agent, giving it the request and the exploration results (file names, code paths) so it can design the implementation. Skip this step only for very simple changes.\n"
        + "3. Review: read the key files the agent pointed at and confirm the approach matches what the user asked for; if anything is still unclear, ask with ask_user_question instead of deciding on your own.\n"
        + "4. Write the plan: put the final approach in the plan file. Start with why the change is needed, include only the approach you recommend, list the key files to change and the existing code to reuse, and end with how to verify it. Keep it short enough to read quickly and detailed enough to follow directly.\n"
        + "5. Submit: once the plan is written, call exit_plan_mode to ask the user for approval.\n\n"
        + "Every turn must end in one of two ways: call ask_user_question to clarify, or call exit_plan_mode to submit the plan. Do not ask whether the plan is acceptable in plain text or through ask_user_question; requesting approval is exactly what exit_plan_mode is for."
    )


def sparse_reminder(plan_file: Path) -> str:
    return _wrap(
        _SENTINELS["sparse"]
        + f" (full rules above). Everything except the plan file ({plan_file}) is read-only. End every turn with ask_user_question or exit_plan_mode; do not ask in plain text whether the plan works."
    )


def exit_reminder(plan_file: Path) -> str:
    text = _SENTINELS["exit"] + " You may edit files and run commands again."
    if plan_file.exists():
        text += f" The plan file is at {plan_file}; read it again whenever you need it."
    return _wrap(text)


def _last_plan_reminder(messages) -> tuple[str | None, int]:
    """
    Scan the history backwards for the most recent plan reminder, returning its kind and how
    many model requests have passed since.
    """
    turns = 0
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            turns += 1
            continue
        for part in msg.parts:
            if not (isinstance(part, UserPromptPart) and isinstance(part.content, str)):
                continue
            for kind, sentinel in _SENTINELS.items():
                if part.content.startswith(f"<system-reminder>\n{sentinel}"):
                    return kind, turns
    return None, turns


def build_plan_reminder(messages, plan_file: Path) -> str | None:
    """
    Compare the mode the model believes it is in (from the most recent plan reminder in the
    history) with the real mode, and pick the reminder that closes the gap.
    """
    # last: kind of the most recent plan reminder ("full", "sparse" or "exit"); None when the
    # history has none
    # turns: model requests since that reminder
    last, turns = _last_plan_reminder(messages)
    believes_plan = last in ("full", "sparse")
    if permissions.state.mode != permissions.PLAN:
        # Plan mode ended but the model still believes it is planning: send an exit reminder
        return exit_reminder(plan_file) if believes_plan else None
    if not believes_plan:
        # Just entered plan mode, or the full reminder was compacted away: send the full one
        return full_reminder(plan_file)
    if turns >= PLAN_REMINDER_TURNS_BETWEEN:
        return sparse_reminder(plan_file)
    return None
