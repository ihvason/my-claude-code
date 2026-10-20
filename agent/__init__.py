"""Complete wrapper package for the coding agent.

External code only needs:
- agent: the Agent instance with its tools and hooks wired up
- MODEL_NAME: the model currently in use (needed by the /status command)
- api_call_log: metadata for the API calls made while the main loop runs the
  agent (needed by the /api-detail command)
- ApiCall: the data structure describing a single API call

The submodules (tools / hooks / core) are implementation details and do not need
to be imported directly.
"""
from .core import agent
from .file_state import ReadFileState
from .hooks import api_call_log, ApiCall
from .model import MODEL_NAME

__all__ = ["agent", "MODEL_NAME", "api_call_log", "ApiCall", "ReadFileState"]
