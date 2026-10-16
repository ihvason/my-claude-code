"""
Core tools for the Coding Agent: read files, write files, and run shell commands.

Each tool handles known errors individually: raise `ModelRetry` for correctable
input issues (e.g., missing path, permission denied); return error messages for
non-recoverable failures (e.g., binary files, command execution errors).
Unexpected exceptions are caught by `on_tool_execute_error` in the hooks.
"""
import re
import os
import shlex
import subprocess

from pydantic_ai import RunContext, Tool
from pydantic_ai.exceptions import ModelRetry

import permissions

from .file_state import ReadFileState

# Default line limit when reading files; content beyond this is truncated with an offset hint
DEFAULT_MAX_LINES = 2000


def _with_line_numbers(content: str, start_line: int = 1) -> str:
    """
    Prepends line numbers to each line (similar to `cat -n` format: right-aligned line number, tab, line content).
    Enables precise line targeting for the model so `old_string` matches accurately in `edit_file`.
    `start_line` designates the initial line number; during chunked reads, it aligns with `offset`.
    """
    lines = content.splitlines()
    if not lines:
        return "(empty file)"
    # Calculate alignment width based on the maximum line number in the current slice
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(f"{i:>{width}}\t{line}" for i, line in enumerate(lines, start_line))


def read_and_register(state: ReadFileState, path: str, offset: int = 1, limit: int | None = None) -> str:
    """
    Reads a file, slices it by offset/limit, registers it in ReadFileState, and returns line-numbered content.
    Invoked by both the read_file tool and @ mention injections, ensuring files introduced via @
    and files read autonomously by the model share identical representations in ReadFileState.
    """
    # 1. Read full file content
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. Slice by offset/limit: default to DEFAULT_MAX_LINES if limit is omitted
    all_lines = content.splitlines()
    total = len(all_lines)
    start = offset - 1
    if start >= total and total > 0:
        return f"Warning: File contains only {total} lines, but offset is {offset}. No content to read."
    end = start + limit if limit is not None else start + DEFAULT_MAX_LINES
    selected = all_lines[start:end]
    selected_content = "\n".join(selected)
    truncated = limit is None and end < total

    # 3. Register in session ReadFileState: cache raw sliced content (without line numbers) and current mtime
    state.record(path, selected_content, offset=offset, limit=limit)

    # 4. Return line-numbered content; append continuation hint if truncated
    result = _with_line_numbers(selected_content, start_line=offset)
    if truncated:
        result += f"\n\n(File has {total} lines total; {total - end} lines remaining. Use offset={end + 1} to continue reading.)"
    return result


def read_file(ctx: RunContext[ReadFileState], path: str, offset: int = 1, limit: int | None = None) -> str:
    """
    Read file contents with line numbers (similar to `cat -n`).
    For large files, read incrementally using offset and limit instead of loading all at once.
    When using edit_file subsequently, do not include line number prefixes in old_string or new_string.

    Args:
        path: Absolute path to the target file.
        offset: Line number to start reading from, 1-indexed (default: 1 = start of file).
        limit: Number of lines to read; defaults to up to DEFAULT_MAX_LINES lines if omitted.
    """
    # 1. Deduplication: skip if recorded by read_file (offset is not None) with identical slice and mtime
    record = ctx.deps.get(path)
    if record is not None and record.get("offset") is not None:
        if record["offset"] == offset and record["limit"] == limit:
            try:
                if os.path.getmtime(path) <= record["timestamp"]:
                    return "File content has not changed. Refer to previous read_file output; duplicate read skipped."
            except FileNotFoundError:
                pass

    # 2. Reading and registration delegated to read_and_register; exceptions are mapped to ModelRetry
    try:
        return read_and_register(ctx.deps, path, offset, limit)
    except FileNotFoundError:
        raise ModelRetry(f"File not found: {path}. Verify the path or specify another file.")
    except PermissionError:
        raise ModelRetry(f"Permission denied: {path}. Specify a readable file.")
    except IsADirectoryError:
        raise ModelRetry(f"{path} is a directory. Specify a specific file inside it.")
    except UnicodeDecodeError:
        return f"Error: {path} is not a text file and cannot be read."


def edit_file(ctx: RunContext[ReadFileState], path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """
    Precisely replace a targeted string snippet within a file to save tokens.
    Target files must be read using read_file before editing; otherwise, an error is raised.
    Prefer edit_file over write_file for modifying existing files.
    old_string must match uniquely in the file; provide surrounding context if ambiguous, or set replace_all=True.
    Do not include line number prefixes from read_file in old_string or new_string.

    Args:
        path: Path to the file to edit.
        old_string: Exact substring to be replaced, matching file content byte-for-byte (without line number prefixes).
        new_string: Replacement content.
        replace_all: Whether to replace all occurrences. Defaults to False (single unique replacement).
    """
    # 1. No-op validation: reject identical replacements
    if old_string == new_string:
        raise ModelRetry("old_string and new_string are identical; no changes made.")

    # 2. Read-before-write enforcement: ensure file was read via read_file before editing
    record = ctx.deps.get(path)
    if record is None:
        raise ModelRetry(f"{path} has not been read yet. Use read_file first to inspect actual content before editing.")

    # 3. Stale-read prevention via mtime: reject edits if file was modified externally after reading
    try:
        current_mtime = os.path.getmtime(path)
    except FileNotFoundError:
        raise ModelRetry(f"{path} no longer exists; cannot edit.")
    if current_mtime > record["timestamp"]:
        raise ModelRetry(f"{path} was modified after it was read. Call read_file again to fetch the latest content.")

    # 4. Uniqueness validation: check occurrence count of old_string in cached snapshot
    snapshot = record["content"]
    count = snapshot.count(old_string)
    if count == 0:
        raise ModelRetry("old_string not found in file. Check output of read_file to match exact content.")
    if count > 1 and not replace_all:
        raise ModelRetry(
            f"old_string matched {count} times. Add surrounding context to make it unique, or set replace_all=True."
        )

    # 5. Read latest content from disk: resolves race conditions during permission approval prompts
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 6. Apply string replacement
    if replace_all:
        updated = content.replace(old_string, new_string)
    else:
        updated = content.replace(old_string, new_string, 1)

    # 7. Write updated content back to disk and update session state cache
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    ctx.deps.record(path, updated)
    return f"Successfully edited {path} (replaced {count if replace_all else 1} occurrence(s))"


def write_file(ctx: RunContext[ReadFileState], path: str, content: str) -> str:
    """
    Write content to a file, overwriting existing content.
    Overwriting an existing file requires reading it first via read_file.
    Prefer edit_file for targeted updates (saves tokens); reserve write_file for creating new files or full rewrites.
    """
    # Overwrite guard: enforce read-before-write on existing files to prevent accidental data loss
    if os.path.exists(path):
        record = ctx.deps.get(path)
        if record is None:
            raise ModelRetry(f"{path} already exists. Read it with read_file before overwriting to verify contents.")
        if os.path.getmtime(path) > record["timestamp"]:
            raise ModelRetry(f"{path} was modified after it was read. Call read_file again before overwriting.")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except FileNotFoundError:
        raise ModelRetry(f"Directory does not exist for {path}. Specify an existing directory.")
    except PermissionError:
        raise ModelRetry(f"Permission denied: {path}. Specify a writable path.")
    except OSError as e:
        return f"Error: Failed to write {path} ({e})"
    # Cache newly written content to allow subsequent edits without re-reading
    ctx.deps.record(path, content)
    return f"Successfully written to {path}"


def run_command(command: str) -> str:
    """
    Execute a shell command and return its output.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, errors="replace", timeout=10
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[Error] {result.stderr}"
        return output or "(No output)"
    except subprocess.TimeoutExpired:
        return "[Error] Command timed out after 10 seconds."
    except OSError as e:
        return f"[Error] Failed to execute command ({e})"


# High-risk patterns: file deletion, privilege escalation, and direct disk writes
DANGEROUS_PATTERNS = [
    r"\brm\b",
    r"\bsudo\b",
    r"\bdd\b",
    r"\bmkfs\w*\b",
]


# Allowlist for safe, read-only commands
READONLY_COMMANDS = {"ls", "cat", "head", "tail", "grep", "wc", "pwd"}
# Chaining, pipes, redirection, and substitutions may introduce side effects
SHELL_META = re.compile(r"[;&|<>`$\n]")


def is_readonly_command(command: str) -> bool:
    """
    Conservatively checks if a command is read-only.
    Must be a single allowlisted binary without shell metacharacters.
    """
    if SHELL_META.search(command):
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    return bool(tokens) and tokens[0] in READONLY_COMMANDS


def run_command_self_check(args: dict):
    """
    Self-check hook for run_command:
    Demands approval if high-risk, allows if confirmed read-only, or defers to general rules.
    """
    command = args.get("command", "")
    if any(re.search(pattern, command) for pattern in DANGEROUS_PATTERNS):
        return "ask"
    if is_readonly_command(command):
        return "allow"
    # Neither high-risk nor provably read-only: delegate to standard rules
    return None


# Register self-check with the permissions module
permissions.register_self_check("run_command", run_command_self_check)

# Pydantic AI automatically infers JSON schemas from function signatures and docstrings via `tools=[plain_function]`.
# `edit_file` and `write_file` are marked with `sequential=True` to serialize file writes within a turn,
# preventing race conditions and overwrites against stale cached snapshots.
TOOLS = [
    read_file,
    Tool(edit_file, sequential=True),
    Tool(write_file, sequential=True),
    run_command,
]
