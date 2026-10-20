"""
Agent instantiation: wires model / instructions / tools / hooks together.
"""
import os
import platform
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

import subagents

from memory import store
from memory.instructions import MEMORY_INSTRUCTIONS
from utils import find_agents_md

from .deps import AgentDeps
from .hooks import hooks
from .tools import TOOLS
from .model import model


INSTRUCTIONS = (
    "You are a programming assistant capable of reading/writing files and executing commands to help users complete software development tasks.\n"
    "Always inspect an existing file using read_file before modifying it. "
    "Prefer edit_file for localized changes (sending only the modified snippets to save tokens); reserve write_file for creating new files or complete overwrites.\n"
    "Workflow: Understand requirements, implement code, and run tests/commands to verify. "
    "Fix any errors and re-verify until confirmed correct.\n"
    "Run persistent or long-running commands (e.g., dev servers, long test suites) in the background using run_command with run_in_background=True—use this only when immediate results are not required, without appending '&'. "
    "You will receive a <task-notification> when a background job finishes, so do not actively poll. You can inspect partial output using read_file on its log file, or terminate it early with job_kill.\n"
    "A background job notifies you once when it finishes, while a monitor notifies you for every event while it runs; use a monitor when you want to know each time something happens. Its events arrive on their own schedule, and even when one shows up while you are waiting for the user's reply, it is not the user replying.\n"
    "Delegate conclusion-only work to a sub agent via run_agent: exploratory code search, well-isolated subtasks, and code reviews that need a second opinion."
    "A sub agent starts from an empty context, so spell out the background in the prompt; only you can see its report, so relay the key points to the user."
    "Sub agents always run in the background: do not poll after dispatching them, and end your turn if there is nothing else to do, since the completion notice carries the report. Independent tasks can be dispatched to several sub agents at once. Do not delegate tasks that one or two tool calls could finish.\n"
    "When handling tasks requiring more than 3 steps or multiple tool calls, "
    "first use task_create to break down steps into pending tasks. Before starting work on a task, use task_update to switch its status to in_progress, and mark it completed immediately upon finishing. "
    "Do not create tasks for trivial requests (1–2 steps, pure conversation, simple queries), as this adds unnecessary overhead. Never leave completed tasks lingering in in_progress.\n"
    "If requirements are ambiguous, have multiple viable approaches, or leave you uncertain, "
    "use the ask_user_question tool to present multiple-choice questions for clarification rather than making unilateral assumptions.\n"
    "Conversations may include <system-reminder>...</system-reminder> tags containing automatically injected system notices. Treat them strictly as system messages, not as part of the surrounding user input or tool output."
) + MEMORY_INSTRUCTIONS

agent = Agent(
    model,
    # System Prompt.
    instructions=INSTRUCTIONS,
    tools=TOOLS,
    deps_type=AgentDeps,
    capabilities=[hooks],
)


# Dynamic instructions: re-evaluated on each model request to inject environment context and project-level AGENTS.md.
# Like static instructions, these are excluded from conversation history while delivering up-to-date values each turn,
# remaining invisible to users without polluting context history.
@agent.instructions
def project_context() -> str:
    cwd = os.getcwd()
    parts = [
        "Environment context:",
        f"- Working directory: {cwd}",
        f"- Operating system: {platform.system()}",
        f"- Today's date: {date.today().isoformat()}",
    ]

    # Read project-level guidelines from AGENTS.md at the project root; append them so
    # the model follows the project's conventions
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

    # Inject the available sub agent types: custom agents are loaded from disk at
    # startup, so dynamic injection avoids hardcoding the list in the tool description
    parts.append("")
    parts.append("Agent types available to run_agent:")
    parts.append(subagents.format_agent_listing())

    # Inject memory index: MEMORY.md is fully injected each turn so the model always knows what memory topics are available.
    # The benefit of using dynamic instructions is retrieving fresh values each turn—memories written in the current turn immediately appear in the next turn's index.
    index = store.read_index()
    if index:
        parts.append("")
        parts.append(f"Below is your memory index MEMORY.md (located at {store.index_path()}, preserved across sessions). Use read_file on the corresponding file in the memory directory whenever you need the full content of a memory:")
        parts.append(index)

    return "\n".join(parts)
