"""
Agent instantiation: wires model / instructions / tools / hooks together.
"""
import os
import platform
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from .file_state import ReadFileState
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
    instructions = (
    "You are a coding assistant. You can read/write files and execute commands to help users accomplish programming tasks.\n"
    "Before modifying an existing file, you must first read it using `read_file`. "
    "Prefer `edit_file` for targeted changes (providing only the modified snippets to save tokens); use `write_file` only when creating new files or performing a complete rewrite.\n"
    "Workflow: Understand the requirements first, write the code, and then run tests/commands to verify. "
    "If errors occur, fix them and rerun until verified correct."
    ),
    tools=TOOLS,
    deps_type=ReadFileState,
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

