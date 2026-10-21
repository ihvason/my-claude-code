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
from media import images

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


def build_mention_messages(paths: list[str], state: ReadFileState) -> tuple[list, list, list[str]]:
    """
    Read every @-mentioned path; text files become a synthetic read_file call/return pair.
    read_and_register records them in readFileState exactly like a real read_file call, so a
    later edit_file is not rejected.

    Returns the history messages, the image attachments, and the original @ paths of the
    images that were turned into attachments.
    """
    messages = []
    attachments = []
    image_paths = []
    for path in paths:
        abs_path = os.path.abspath(os.path.expanduser(path))
        # Missing paths and directories are skipped
        if not os.path.isfile(abs_path):
            continue

        # Images stay out of the history: they are collected as attachments and sent with
        # this turn's input instead
        if images.is_image(abs_path):
            try:
                attachments.append(images.load_image(abs_path))
                image_paths.append(path)
            except (OSError, images.ImageError):
                continue
            continue

        try:
            numbered = read_and_register(state, abs_path)
        except (OSError, UnicodeDecodeError):
            continue

        # tool_call and tool_return must share one id, otherwise the model does not see them
        # as a single complete call
        call_id = "mention_" + uuid.uuid4().hex[:8]
        # An assistant message pretending the model issued the read_file call itself
        messages.append(
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_file", args={"path": abs_path}, tool_call_id=call_id
                    )
                ]
            )
        )
        # A tool return holding the numbered file body, identical to a real read_file result
        messages.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="read_file", content=numbered, tool_call_id=call_id)
                ]
            )
        )
    return messages, attachments, image_paths


def replace_image_mentions(text: str, image_paths: list[str], first_index: int) -> str:
    """
    Replace each image @path in place with its internal image placeholder.
    """
    indexes = {path: first_index + offset for offset, path in enumerate(image_paths)}

    def replace(match):
        path = match.group(1)
        index = indexes.get(path)
        if index is None:
            return match.group(0)
        prefix = match.group(0).removesuffix(f"@{path}")
        return f"{prefix}[Image #{index}]"

    return AT_MENTION_RE.sub(replace, text)


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
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            # Git emits raw bytes; surrogateescape keeps non-UTF-8 filenames from raising
            encoding="utf-8",
            errors="surrogateescape",
            timeout=2,
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
