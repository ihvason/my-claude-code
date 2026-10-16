"""
File mentions via @: parses @path references from user input, reads their content,
and injects mock read_file tool invocations into the conversation history.
This convinces the model that the referenced files have already been read, satisfying
read-before-write checks and enabling immediate edits.
"""
import os
import re
import subprocess
import uuid

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from agent.file_state import ReadFileState
from agent.tools import read_and_register

# Matches @ mentions: requires preceding start-of-line or whitespace to avoid false positives (e.g., foo@bar.com).
# Captures standard path characters (letters, digits, underscores, dots, slashes, backslashes, hyphens, tildes),
# terminating at whitespace or punctuation to avoid capturing subsequent sentence fragments.
# Supports only plain @path syntax; quoted paths, line ranges, and @agent directives are omitted for simplicity.
AT_MENTION_RE = re.compile(r"(?:^|\s)@([\w./~\\\-]+)")


def extract_at_mentions(text: str) -> list[str]:
    """
    Extracts all @path references from user input, deduplicated while preserving order.
    """
    seen = []
    for match in AT_MENTION_RE.finditer(text):
        path = match.group(1)
        if path not in seen:
            seen.append(path)
    return seen


def build_mention_messages(paths: list[str], state: ReadFileState) -> list:
    """
    Reads each mentioned file and constructs mock read_file tool_call and tool_return pairs
    to append to the message history.
    `read_and_register` caches the file in ReadFileState identically to a genuine read_file call,
    allowing subsequent edit_file invocations to pass validation.
    """
    messages = []
    for path in paths:
        abs_path = os.path.abspath(os.path.expanduser(path))
        # Skip non-existent paths and directories
        if not os.path.isfile(abs_path):
            continue
        try:
            numbered = read_and_register(state, abs_path)
        except (OSError, UnicodeDecodeError):
            continue

        # tool_call and tool_return must share an identical call_id for pairing
        call_id = "mention_" + uuid.uuid4().hex[:8]
        # Assistant message simulating a model-initiated read_file call
        messages.append(
            ModelResponse(
                parts=[ToolCallPart(tool_name="read_file", args={"path": abs_path}, tool_call_id=call_id)]
            )
        )
        # Tool return message delivering line-numbered content matching genuine read_file output
        messages.append(
            ModelRequest(
                parts=[ToolReturnPart(tool_name="read_file", content=numbered, tool_call_id=call_id)]
            )
        )
    return messages


# Common noise directories to skip during traversal (dependencies, caches, build artifacts)
# Serves as a static fallback blocklist when .gitignore is unavailable
IGNORED_DIRS = {"__pycache__", "node_modules", "venv", "dist", "build", "target"}


def list_candidate_files() -> list[str]:
    """
    Returns candidate files for @ autocompletion.
    Prefers `git ls-files` (fast, respects .gitignore); falls back to directory traversal.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, timeout=2
        )
        tracked = result.stdout.splitlines() if result.returncode == 0 else []
        # Use tracked files if available; fall back if empty (non-git repo or no staged files)
        if tracked:
            return tracked
    except (OSError, subprocess.SubprocessError):
        pass

    # Fall back to directory walk; prune hidden folders (.git, .venv) and ignored directories
    files = []
    for root, dirs, names in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in IGNORED_DIRS]
        for name in names:
            files.append(os.path.relpath(os.path.join(root, name)))
    return files
