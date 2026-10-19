"""
Agent instantiation: wires model / instructions / tools / hooks together.
"""
import os
import platform
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from memory import store
from memory.instructions import MEMORY_INSTRUCTIONS

from .deps import AgentDeps
from .hooks import hooks
from .tools import TOOLS

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set")

MODEL_NAME = "deepseek-flash"

model = OpenAIChatModel(
    MODEL_NAME,
    provider=DeepSeekProvider(api_key=API_KEY),
)

INSTRUCTIONS = (
    "You are a programming assistant capable of reading/writing files and executing commands to help users complete software development tasks.\n"
    "Always inspect an existing file using read_file before modifying it. "
    "Prefer edit_file for localized changes (sending only the modified snippets to save tokens); reserve write_file for creating new files or complete overwrites.\n"
    "Workflow: Understand requirements, implement code, and run tests/commands to verify. "
    "Fix any errors and re-verify until confirmed correct.\n"
    "Run persistent or long-running commands (e.g., dev servers, long test suites) in the background using run_command with run_in_background=True—use this only when immediate results are not required, without appending '&'. "
    "You will receive a <task-notification> when a background job finishes, so do not actively poll. You can inspect partial output using read_file on its log file, or terminate it early with job_kill.\n"
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

    # Read project-level guidelines from AGENTS.md in the root directory; append if present to enforce project conventions
    agents_md = os.path.join(cwd, "AGENTS.md")
    if os.path.isfile(agents_md):
        try:
            content = open(agents_md, encoding="utf-8").read()
        except OSError:
            content = ""
        if content.strip():
            parts.append("")
            parts.append("Project AGENTS.md guidelines (adhere to these conventions):")
            parts.append(content)

    # Inject memory index: MEMORY.md is fully injected each turn so the model always knows what memory topics are available.
    # The benefit of using dynamic instructions is retrieving fresh values each turn—memories written in the current turn immediately appear in the next turn's index.
    index = store.read_index()
    if index:
        parts.append("")
        parts.append(f"Below is your memory index MEMORY.md (located at {store.index_path()}, preserved across sessions). Use read_file on the corresponding file in the memory directory whenever you need the full content of a memory:")
        parts.append(index)

    return "\n".join(parts)
