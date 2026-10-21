"""
Plan mode tools: enter_plan_mode lets the model enter plan mode on its own, exit_plan_mode
submits the written plan for user approval.
"""
from pydantic_ai import RunContext

from security import permissions

from ..deps import AgentDeps


def enter_plan_mode(ctx: RunContext[AgentDeps]) -> str:
    """
    Enter plan mode: investigate read-only, design the approach, and get the written plan
    approved by the user before touching code.
    Suited to non-trivial implementation work: adding a feature, choosing between several
    viable approaches, changing multiple files, or starting from unclear requirements that
    need investigation first.
    Do not use it for simple work: fixing a typo, changing a few lines, a task the user has
    already broken down, or a plain information lookup.
    """
    if permissions.state.mode == permissions.PLAN:
        return "Already in plan mode."
    # The reminder tells the model to write the plan file directly, so its directory has to
    # exist beforehand; doing it here also covers sessions switched with /new
    if ctx.deps.plan_file is not None:
        ctx.deps.plan_file.parent.mkdir(parents=True, exist_ok=True)
    permissions.set_mode(permissions.PLAN)
    return "Entered plan mode. Focus on exploring the code and designing the approach; do not modify project files until the user approves the plan."


async def exit_plan_mode(ctx: RunContext[AgentDeps]) -> str:
    """
    Call after writing the plan to the plan file to request the user's approval.
    It takes no plan text; it reads the plan file and shows it to the user. Use it only when
    planning work that requires code changes, not for pure investigation tasks.
    """
    if permissions.state.mode != permissions.PLAN:
        return "Not in plan mode. If the plan is already approved, continue implementing."
    plan_file = ctx.deps.plan_file
    plan = plan_file.read_text(encoding="utf-8").strip() if plan_file.exists() else ""
    if not plan:
        return f"The plan file {plan_file} is still empty; write the plan first, then call exit_plan_mode."

    choice, feedback = await permissions.prompt_plan_approval(plan, plan_file)
    if choice == "approve":
        # Approved: restore the mode from before plan mode and hand the full plan back as the
        # tool result
        permissions.set_mode(permissions.state.pre_plan_mode)
        return (
            "The user approved your plan; you may start writing code now. If there are several steps, break the plan into tasks with task_create first.\n\n"
            f"The plan is saved at {plan_file}; read it again at any point while implementing.\n\n"
            f"## Approved plan\n{plan}"
        )
    if feedback:
        return f"The user did not approve this plan; you are still in plan mode. Their feedback:\n{feedback}\nRevise the plan file accordingly, then call exit_plan_mode again."
    return "The user did not approve this plan; you are still in plan mode. Stop here and wait for the user to tell you how to proceed."
