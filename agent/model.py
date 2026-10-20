"""
Model configuration shared by the main agent wiring (core.py) and the sub agent module
(subagents.py), so neither has to import the other.
"""
import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

# Read the API key from the environment
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set")

MODEL_NAME = "deepseek-flash"

model = OpenAIChatModel(
    MODEL_NAME,
    provider=DeepSeekProvider(api_key=API_KEY),
)
