"""
Agent instantiation: wires model / instructions / tools / hooks together.
"""
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from .hooks import hooks
from .tools import TOOLS
from .file_state import ReadFileState

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
