"""
Agent instantiation: wires model / instructions / tools / hooks together.
"""
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

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
    # Deliberately kept in Chinese: the language of the prompt determines the
    # language the model replies in. Translate this if you want English answers.
    instructions = (
    "You are a coding assistant capable of reading and writing files and executing shell commands to solve programming tasks.\n"
    "Workflow: Understand the requirements first, implement the code, and verify it by running tests or commands. "
    "If any errors occur, diagnose, fix, and re-run until the solution is fully confirmed to work."
    ),
    tools=TOOLS,
    capabilities=[hooks],
)
