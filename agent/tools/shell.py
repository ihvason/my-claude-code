"""
shell tools: run_command. Its self-check requires approval for dangerous commands
and auto-approves simple read-only ones.
"""
import re
import shlex
import subprocess

import permissions


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
    except OSError as e:
        return f"[error] failed to execute command ({e})"


# Danger signatures: file deletion, privilege escalation, raw disk writes
DANGEROUS_PATTERNS = [
    r"\brm\b",
    r"\bsudo\b",
    r"\bdd\b",
    r"\bmkfs\w*\b",
]


# Read-only command allowlist
READONLY_COMMANDS = {"ls", "cat", "head", "tail", "grep", "wc", "pwd"}
# Chaining, pipes, redirection and command substitution can all smuggle in writes
SHELL_META = re.compile(r"[;&|<>`$\n]")


def is_readonly_command(command: str) -> bool:
    """
    Conservatively decide whether a command is read-only: it must be a single
    allowlisted command carrying no shell metacharacters.
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
    Permission self-check for run_command: require approval when a danger
    signature matches, auto-approve a confirmed read-only command, and defer
    everything else to the general rules.
    """
    command = args.get("command", "")
    if any(re.search(pattern, command) for pattern in DANGEROUS_PATTERNS):
        return "ask"
    if is_readonly_command(command):
        return "allow"
    # Neither dangerous nor provably read-only: let the general rules decide
    return None


# Register the self-check with the permissions registry
permissions.register_self_check("run_command", run_command_self_check)
