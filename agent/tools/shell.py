"""
shell tools: run_command and job_kill. The run_command self-check requires approval for
dangerous commands and auto-approves simple read-only ones.
"""
import asyncio
import re
import shlex
import time

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

import permissions

from ..deps import AgentDeps

FOREGROUND_TIMEOUT = 10


async def run_command(
    ctx: RunContext[AgentDeps], command: str, run_in_background: bool = False
) -> str:
    """
    Run a shell command. Runs in the foreground by default and returns the output directly.

    For long-running or slow commands (dev servers, long test suites), pass
    run_in_background=True, and only when you do not need the result immediately: the
    call returns a job id at once (do not append "&" to the command), and you receive a
    <task-notification> when the job finishes - do not poll or wait for it. Meanwhile you
    can read the job's log file with read_file at any time to inspect partial output.

    Args:
        command: the shell command to run
        run_in_background: run as a background job instead of waiting for it to finish
    """
    registry = ctx.deps.job_registry
    # Foreground commands also go through the registry, so the user can press ctrl+b to
    # move them to the background at any time
    job = await registry.spawn_shell(command, background=run_in_background)
    if run_in_background:
        return _background_notice(job, f"Moved to the background, job id: {job.id}")

    start = time.monotonic()
    try:
        while job.status == "running" and not job.background:
            if time.monotonic() - start > FOREGROUND_TIMEOUT:
                registry.kill(job.id)
                return f"[error] command timed out after {FOREGROUND_TIMEOUT}s; for long-running or slow commands, re-run with run_in_background=True"
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        # When the user cancels this turn with Esc, terminate the foreground command too
        # so no orphaned process is left behind
        registry.kill(job.id)
        raise

    if job.background:
        return _background_notice(job, f"The user moved this command to the background; it continues as job {job.id}")

    output = job.log_path.read_text(errors="replace")
    if job.returncode != 0:
        output += f"\n[error] exit code {job.returncode}"
    return output or "(no output)"


def _background_notice(job, lead: str) -> str:
    # Reply tail shared by explicit background runs and ctrl+b moves
    return (
        f"{lead}\n"
        f"Log file: {job.log_path} (inspect it any time with read_file)\n"
        "You will receive a <task-notification> when it finishes; do not wait for it."
    )


def job_kill(ctx: RunContext[AgentDeps], job_id: str) -> str:
    """
    Terminate a running background job (its entire process tree is terminated).

    Args:
        job_id: id of the job to terminate (e.g. "b3k9f02qx")
    """
    registry = ctx.deps.job_registry
    job = registry.get(job_id)
    if job is None:
        raise ModelRetry(f"job {job_id} does not exist")
    if not registry.kill(job_id):
        return f"job {job_id} has already finished ({job.status}); nothing to terminate"
    return f"job {job_id} terminated"


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
    Permission self-check for run_command: require approval when a danger signature
    matches, auto-approve a confirmed read-only command, and defer everything else to
    the general rules.
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
