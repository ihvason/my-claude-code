"""
Agent instantiation: wires model / instructions / tools / hooks together.
"""
import os
import platform
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

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

agent = Agent(
    model,
    # System Prompt.
    instructions=(
    "You are a programming assistant capable of reading/writing files and executing commands to help users complete development tasks.\n"
    "Always inspect an existing file using read_file before modifying it. "
    "Prefer edit_file for partial updates (passing only changed snippets to save tokens); reserve write_file for creating new files or complete overwrites.\n"
    "Workflow: Understand requirements, implement code, and run tests/commands to verify. "
    "Fix any errors and re-verify until confirmed correct.\n"
    "If requirements are ambiguous, have multiple viable approaches, or leave you uncertain, "
    "use ask_user_question to provide multiple-choice options for clarification rather than making unilateral assumptions.\n"
    "Conversations may include <system-reminder>...</system-reminder> tags containing automatically injected system notices. Treat them strictly as system messages, not as part of the surrounding user input or tool output."
    ),
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

    return "\n".join(parts)
