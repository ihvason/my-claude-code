"""
monitor tool: watch a long-running command and feed its non-empty output lines into the
conversation as events.
"""

from pydantic_ai import RunContext

from security import permissions

from ..deps import AgentDeps
from .shell import run_command_self_check


async def monitor(
    ctx: RunContext[AgentDeps],
    command: str,
    description: str,
    timeout: int = 300,
    persistent: bool = False,
) -> str:
    """
    Start a background monitor: every non-empty output line is delivered as a <monitor-event>
    notification. Once it is running, get on with your own work; do not poll and do not sleep
    waiting for events.

    Choose the tool by how many notifications you need. If one notification is enough (for
    example "tell me when the build finishes"), use run_command with run_in_background=True and
    a command that exits once the condition holds, such as
    `until grep -q "Ready" dev.log; do sleep 1; done`.
    Use monitor only when you want a notification every time something happens, for example
    `tail -n 0 -F app.log | grep --line-buffered "ERROR"`, or a `while true` loop polling a
    remote status.

    Every stage of the pipeline must flush its output line by line, otherwise matched lines can
    sit in a buffer; that is why grep needs --line-buffered.

    A pipe only carries stdout, so the command's stderr bypasses the filter and enters the
    event stream unfiltered. Write `cmd 2>&1 | grep ...` to filter stderr as well.

    Silence is not success: a filter that only matches normal output stays silent when the
    process crashes, which looks exactly like "still running". Cover failure markers too: write
    `grep -E --line-buffered "step=|Traceback|Error|FAILED"` rather than only
    `grep --line-buffered "step="`.

    Every event costs context. A monitor that keeps exceeding the rate limit is stopped
    automatically, so filter down to the lines you will actually act on.

    Args:
        command: Shell command to watch
        description: What you are watching, a few words; it appears in every notification
        timeout: Seconds after which the monitor is stopped
        persistent: Run for the whole session, ignoring timeout
    """
    if timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    registry = ctx.deps.job_registry
    job = await registry.spawn_monitor(command, description, None if persistent else timeout)
    return (
        f"monitor started, job id: {job.id}\n"
        f"Output log: {job.log_path} (use read_file at any time to inspect the recorded output)\n"
        "Non-empty output lines are delivered as <monitor-event> notifications. Do not wait on them; job_kill stops the monitor at any time."
    )


# Reuse run_command's dangerous-command self-check; both tools take a parameter named command
permissions.register_self_check("monitor", run_command_self_check)
