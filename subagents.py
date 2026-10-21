"""
Sub agent module: agent type definitions (built-in and custom), sub agent specific hooks,
and the executor that runs a sub agent inside an agent-type job.

A sub agent is a second Agent instance: a fresh context (no message_history), its own
system prompt, and a trimmed tool set. The tool set excludes run_agent, so a sub agent
cannot dispatch another one.
"""
import asyncio
import dataclasses
import os
import platform
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.usage import UsageLimits
from pydantic_graph import End

import classifier
import permissions
import skills
from agent.deps import AgentDeps
from agent.file_state import ReadFileState
from agent.hooks import _handle_tool_error, _retry_on_error
from agent.model import model
from agent.reminders import build_job_reminder_text
from agent.tools.file import edit_file, read_file, write_file
from agent.tools.shell import run_command
from background_jobs import Job, JobRegistry
from memory.store import _parse_frontmatter
from utils import find_agents_md, get_data_dir

# Maximum number of model requests per sub agent run, so a runaway loop cannot spin forever
MAX_SUBAGENT_REQUESTS = 40


# ---------- Sub agent hooks ----------

sub_hooks = Hooks()


@sub_hooks.on.tool_execute
async def _check_sub_permission(ctx, *, call, tool_def, args, handler):
    """
    Permission gate for sub agents. Calls the rules allow run as usual; in auto mode the
    classifier reviews them first. Calls that need human approval cannot open a prompt
    directly (the sub agent runs in the background), so the request is queued and bubbled
    up to the user, and the sub agent waits for the answer.
    """
    if permissions.compute_decision(call.tool_name, args) == "allow":
        return await handler(args)
    # In auto mode the classifier is already the unattended approver, so sub agents use it
    # too; if the classifier itself errors, fall back to human approval below
    if permissions.state.mode == permissions.AUTO:
        verdict = await classifier.classify(ctx.messages, call.tool_name, args)
        if not verdict.get("error"):
            if not verdict["should_block"]:
                return await handler(args)
            return (
                f"The safety check blocked this {call.tool_name} call, so it did not run. "
                f"Reason: {verdict['reason']}. Do not try to bypass the block; find another "
                "way to finish the task, or mention in your final report that this step was skipped."
            )
    choice = await _request_user_approval(ctx.deps.subagent_job, call.tool_name, args)
    if choice == "always":
        # Record it in the session allowlist, shared with the main agent's "don't ask again"
        permissions.state.session_allowed.add(call.tool_name)
    if choice in ("once", "always"):
        return await handler(args)
    return (
        f"The user denied this {call.tool_name} call. "
        "Find a way to finish the task without approval, or mention in your final report "
        "that this step was skipped."
    )


@dataclass
class PendingApproval:
    """
    An approval request bubbled up from a sub agent, waiting for the user's answer.
    """
    job: Job
    tool_name: str
    args: dict
    # The answer is delivered through this future; the waiting sub agent resumes when it resolves
    future: asyncio.Future


PENDING_APPROVALS: list[PendingApproval] = []


async def _request_user_approval(job: Job, tool_name: str, args: dict) -> str:
    fut = asyncio.get_running_loop().create_future()
    PENDING_APPROVALS.append(PendingApproval(job, tool_name, args, fut))
    return await fut


def pop_pending_approval() -> PendingApproval | None:
    """
    Pop the next still-valid approval request; requests whose job is no longer running are
    dropped.
    """
    while PENDING_APPROVALS:
        req = PENDING_APPROVALS.pop(0)
        if req.job.status == "running" and not req.future.done():
            return req
    return None


# Network retries and tool error fallbacks match the main agent, so its hook functions are reused
sub_hooks.on.model_request(_retry_on_error)
sub_hooks.on.tool_execute_error(_handle_tool_error)


@sub_hooks.on.before_model_request
async def _sub_job_notifications(ctx, request_context):
    """
    A sub agent can also background long commands; their completion notices ride along
    with its next model request.
    """
    text = build_job_reminder_text(ctx.deps.job_registry)
    if text is None:
        return request_context
    messages = list(request_context.messages)
    messages.append(ModelRequest(parts=[UserPromptPart(content=text)]))
    return dataclasses.replace(request_context, messages=messages)


# ---------- Agent type definitions ----------

# Every tool a sub agent may use; run_agent, ask_user_question, task_*, job_kill and monitor are excluded
_TOOL_FUNCS = {
    "read_file": read_file,
    "load_skill": skills.load_skill,
    "edit_file": edit_file,
    "write_file": write_file,
    "run_command": run_command,
}


def _build_tools(tool_names: list[str]) -> list:
    # Each Agent instance gets fresh Tool wrappers; file-modifying tools stay sequential=True
    return [
        Tool(_TOOL_FUNCS[name], sequential=True) if name in ("edit_file", "write_file") else _TOOL_FUNCS[name]
        for name in tool_names
    ]


@dataclass
class AgentType:
    name: str
    # How the main agent decides when to use this type; injected into its dynamic instructions
    description: str
    tool_names: list[str]
    # The sub agent's system prompt, replacing the main agent's rather than appending to it
    instructions: str
    # "built-in", or the path of the definition file for a custom agent
    source: str
    # The Agent instance, wired up in _register
    agent: Agent = field(init=False, repr=False)


def _sub_env_context() -> str:
    # Sub agent environment context: working directory, OS, date, the project's AGENTS.md and
    # the skill listing only; no memory index and no task guidance
    cwd = os.getcwd()
    parts = [
        "Environment context:",
        f"- Working directory: {cwd}",
        f"- Operating system: {platform.system()}",
        f"- Today's date: {date.today().isoformat()}",
    ]
    agents_md = find_agents_md(cwd)
    if agents_md is not None:
        try:
            content = agents_md.read_text(encoding="utf-8")
        except OSError:
            content = ""
        if content.strip():
            parts.append("")
            parts.append("Project AGENTS.md guidelines (adhere to these conventions):")
            parts.append(content)

    # Same progressive skill listing as the main agent: name + description now, full body
    # through load_skill once a task matches
    skill_listing = skills.format_skill_listing(skills.discover_skills())
    if skill_listing:
        parts.append("")
        parts.append(skill_listing)
    return "\n".join(parts)


_TYPES: dict[str, AgentType] = {}


def _register(atype: AgentType) -> None:
    atype.agent = Agent(
        model,
        instructions=[atype.instructions, _sub_env_context],
        tools=_build_tools(atype.tool_names),
        deps_type=AgentDeps,
        capabilities=[sub_hooks],
    )
    _TYPES[atype.name] = atype


EXPLORE_INSTRUCTIONS = (
    "You are a read-only code investigation agent. Search and read code in the project to answer the questions you are given.\n"
    "Never modify anything: do not create or change files, and use run_command only for inspection and search (ls, grep, find, cat and the like), never for commands with side effects.\n"
    "When the investigation is done, output a concise report: conclusions first, with the key file paths as evidence. The caller only sees the report you output last, not your intermediate steps, so the report must stand on its own."
)

GENERAL_INSTRUCTIONS = (
    "You are a sub agent dispatched by the main agent. Complete the task you are given independently, and you may read and write files and run commands.\n"
    "Always read an existing file with read_file before modifying it. Prefer edit_file for localized changes; use write_file only to create files or fully rewrite them.\n"
    "Nobody will answer your questions. When in doubt, take the most reasonable approach and explain your trade-off in the report.\n"
    "When done, output a concise final report: what you did, which files you changed, and the key findings. The caller only sees this report, not your intermediate steps."
)

_register(AgentType(
    name="explore",
    description="Read-only code investigation agent: searches and reads code to answer questions such as \"where is X\" or \"how is Y implemented\", making no changes",
    tool_names=["read_file", "load_skill", "run_command"],
    instructions=EXPLORE_INSTRUCTIONS,
    source="built-in",
))

_register(AgentType(
    name="general",
    description="General purpose agent that can read and write files and run commands; suited to multi-step subtasks such as writing tests or fixing an isolated bug",
    tool_names=list(_TOOL_FUNCS),
    instructions=GENERAL_INSTRUCTIONS,
    source="built-in",
))


# ---------- Custom agent loading ----------

def load_agent_types() -> list[AgentType]:
    """
    Load custom agent definitions from <repo>/.my-claude-code/agents/*.md.
    The frontmatter supplies name / description / tools; the markdown body is the
    agent's system prompt.

    Definitions are configuration, not runtime data: they sit next to the projects/
    data directory and belong in version control, unlike the gitignored projects/.
    """
    loaded = []
    agents_dir = get_data_dir() / "agents"
    if not agents_dir.is_dir():
        return loaded
    for path in sorted(agents_dir.glob("*.md")):
        fields = _parse_frontmatter(path)
        # A markdown file without frontmatter is not an agent definition (it may be a note
        # in the directory), so skip it silently
        if not fields:
            continue
        name = fields.get("name", "")
        description = fields.get("description", "")
        if not name or not description:
            raise RuntimeError(f"Custom agent {path} is missing the required name or description frontmatter field")
        if name in _TYPES:
            raise RuntimeError(f"Custom agent {path} uses the name {name!r}, which collides with an existing agent type")
        body = _read_body(path)
        if not body:
            raise RuntimeError(f"Custom agent {path} has an empty body; the body is the agent's system prompt")
        atype = AgentType(
            name=name,
            description=description,
            tool_names=_parse_tools_field(fields.get("tools"), path),
            instructions=body,
            source=str(path),
        )
        _register(atype)
        loaded.append(atype)
    return loaded


def _read_body(path: Path) -> str:
    # Drop the frontmatter header; the remaining body is the system prompt
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        pieces = text.split("---", 2)
        if len(pieces) == 3:
            return pieces[2].strip()
    return text.strip()


def _parse_tools_field(value: str | None, path: Path) -> list[str]:
    # Accept the full-width comma too, since definition files are hand-edited
    names = [n.strip() for n in (value or "").replace("\uff0c", ",").split(",") if n.strip()]
    # An omitted or empty tools field means every tool; otherwise it is an allowlist
    if not names:
        return list(_TOOL_FUNCS)
    unknown = [n for n in names if n not in _TOOL_FUNCS]
    if unknown:
        raise RuntimeError(
            f"Custom agent {path} has unknown tools in its tools field: {', '.join(unknown)}; available tools: {', '.join(_TOOL_FUNCS)}"
        )
    return names


def get_agent_type(name: str) -> AgentType | None:
    return _TYPES.get(name)


def list_agent_types() -> list[AgentType]:
    return list(_TYPES.values())


def format_agent_listing() -> str:
    """
    One line per agent type, injected into the main agent's dynamic instructions.
    """
    return "\n".join(
        f"- {atype.name}: {atype.description} (tools: {', '.join(atype.tool_names)})"
        for atype in _TYPES.values()
    )


# ---------- Executor ----------

async def run_subagent(atype: AgentType, prompt: str, job: Job, parent_deps: AgentDeps) -> None:
    """
    Run a sub agent inside an agent-type job: fresh context, progress streamed live to the
    job's log file, final report stored in job.result.
    """
    # The sub agent gets its own registry so the processes it starts die with it. Its
    # logs nest inside the spawning job's directory under the parent session's jobs/
    # directory, so a whole sub-agent run stays a single subtree on disk.
    sub_registry = JobRegistry(
        session_id=parent_deps.job_registry.session_id,
        jobs_dir=parent_deps.job_registry.jobs_dir / job.id,
    )
    # Derive from the main session deps: isolate the read-file state and swap in this sub
    # agent's own job registry, inheriting every other field as-is (file_history is
    # inherited too, so files a sub agent changed can still be rewound by /rewind)
    deps = dataclasses.replace(
        parent_deps,
        read_file_state=ReadFileState(),
        tasks_store=None,
        job_registry=sub_registry,
        subagent_job=job,
    )
    log = open(job.log_path, "a", encoding="utf-8")
    log.write(f"=== sub agent {job.id} ({atype.name}) ===\nprompt: {prompt}\n\n")
    log.flush()
    try:
        async with atype.agent.iter(
            prompt, deps=deps,
            usage_limits=UsageLimits(request_limit=MAX_SUBAGENT_REQUESTS),
        ) as run:
            node = run.next_node
            while not isinstance(node, End):
                node = await run.next(node)
                if Agent.is_call_tools_node(node):
                    for part in node.model_response.parts:
                        _log_part(log, part)
                elif Agent.is_model_request_node(node):
                    for part in node.request.parts:
                        if part.part_kind in ("tool-return", "retry-prompt"):
                            _log_part(log, part)
        job.result = run.result.output or "(sub agent produced no report)"
        log.write(f"=== Final report ===\n{job.result}\n")
    finally:
        if job.status == "killed":
            log.write("=== Terminated ===\n")
        log.close()
        sub_registry.shutdown()


def _squash(text, limit: int) -> str:
    # Slice before collapsing: tool output can be several MB, and a full split/join is not
    # worth it just to keep 500 characters
    text = " ".join(str(text)[: limit * 4].split())
    return text if len(text) <= limit else text[:limit] + "..."


def _format_part(part) -> str | None:
    kind = part.part_kind
    if kind == "text":
        return f"● {part.content}"
    if kind == "thinking":
        return f"✻ {part.content}"
    if kind == "tool-call":
        return f"⏺ {part.tool_name}({_squash(part.args, 200)})"
    if kind == "tool-return":
        return f"✔ {part.tool_name} -> {_squash(part.content, 500)}"
    if kind == "retry-prompt":
        return f"✘ {part.tool_name}: {_squash(part.content, 200)}"
    return None


def _log_part(log, part) -> None:
    line = _format_part(part)
    if line is None:
        return
    log.write(line + "\n\n")
    log.flush()
