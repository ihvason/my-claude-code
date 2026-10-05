"""
The three tools available to the coding agent: read a file, write a file, and
run a shell command.
"""
import subprocess


def read_file(path: str) -> str:
    """
    Read the contents of the specified file.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file {path} does not exist"


def write_file(path: str, content: str) -> str:
    """
    Write the given content to the specified file.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {path}"


def run_command(command: str) -> str:
    """
    Run a shell command and return its output.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, errors="replace", timeout=10
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[error] {result.stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] command timed out after 10 seconds"


# Pydantic AI accepts tools=[plain_function] and generates the JSON Schema from
# the function signature plus its docstring
TOOLS = [read_file, write_file, run_command]
