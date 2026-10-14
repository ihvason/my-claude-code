"""
Core tools for the Coding Agent: read files, write files, and run shell commands.

Each tool handles known errors individually: raise `ModelRetry` for correctable
input issues (e.g., missing path, permission denied); return error messages for
non-recoverable failures (e.g., binary files, command execution errors).
Unexpected exceptions are caught by `on_tool_execute_error` in the hooks.
"""
import subprocess

from pydantic_ai.exceptions import ModelRetry


def read_file(path: str) -> str:
    """
    Read the contents of a specified file.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise ModelRetry(f"File not found: {path}. Check the path or specify another file.")
    except PermissionError:
        raise ModelRetry(f"Permission denied: {path}. Specify a readable file.")
    except IsADirectoryError:
        raise ModelRetry(f"{path} is a directory. Specify a specific file inside it.")
    except UnicodeDecodeError:
        return f"Error: {path} is not a text file and cannot be read."


def write_file(path: str, content: str) -> str:
    """
    Write content to a specified file.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully written to {path}"
    except FileNotFoundError:
        raise ModelRetry(f"Directory does not exist for {path}. Specify an existing directory.")
    except PermissionError:
        raise ModelRetry(f"Permission denied: {path}. Specify a writable path.")
    except OSError as e:
        return f"Error: Failed to write {path} ({e})"


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


# Pydantic AI automatically infers JSON schemas from function signatures and docstrings via `tools=[plain_function]`
TOOLS = [read_file, write_file, run_command]
