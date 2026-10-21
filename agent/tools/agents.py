"""
run_agent tool: delegates a task to a sub agent, which runs as an agent-type background
job in the same registry as background commands.
"""
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from .. import subagents
from ..deps import AgentDeps
from .shell import _background_notice


async def run_agent(
    ctx: RunContext[AgentDeps],
    description: str,
    prompt: str,
    agent_type: str = "general",
) -> str:
    """
    Delegate a task to a sub agent. It works in a fresh, isolated context and hands
    back only its final report.

    Sub agents always run in the background: this call returns a job id immediately, and
    you receive a <task-notification> when it finishes, with the final report in its
    <result> field. Do not poll for it. A sub agent knows nothing about this
    conversation, so write the prompt like a briefing for a new colleague: the goal, the
    necessary background, and what the report must cover. The user cannot see the report,
    so relay the key points. Available agent types are listed in the system prompt.

    Args:
        description: Short summary of the task (3-5 words), shown in the job list
        prompt: Complete task description for the sub agent
        agent_type: Which agent type to use
    """
    atype = subagents.get_agent_type(agent_type)
    if atype is None:
        available = ", ".join(t.name for t in subagents.list_agent_types())
        raise ModelRetry(f"Unknown agent type {agent_type!r}; available types: {available}")

    registry = ctx.deps.job_registry
    job = await registry.spawn_agent(
        description,
        lambda job: subagents.run_subagent(atype, prompt, job, ctx.deps),
    )
    return _background_notice(job, f"Sub agent started in the background, job id: {job.id}; the completion notice carries its final report in the <result> field")
