"""
Background task layer: runs shell commands as background jobs, streams their output
to log files, and tracks completion state for notification.
"""
from __future__ import annotations

import asyncio
import os
import random
import signal
import string
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from utils import get_project_dir

JOBS_ROOT = get_project_dir() / "jobs"

_ID_ALPHABET = string.digits + string.ascii_lowercase


def _new_job_id() -> str:
    # Random ids instead of a counter: restarting the process or resuming a session
    # can neither collide nor overwrite old logs
    return "b" + "".join(random.choices(_ID_ALPHABET, k=8))


JobStatus = Literal["running", "completed", "failed", "killed"]

_STATUS_TEXT = {
    "completed": "succeeded",
    "failed": "failed",
    "killed": "was terminated",
}


@dataclass
class Job:
    id: str
    # Only "shell" for now; reserved for future subagent job kinds
    kind: str
    description: str
    # Job output is stored in a log file
    log_path: Path
    status: JobStatus = "running"
    returncode: int | None = None
    notified: bool = False
    # False means a foreground command: its result returns with the tool result and no
    # completion notice is sent; ctrl+b flips it to True
    background: bool = True
    # Termination callback injected by the creator
    kill_func: Callable[[], None] | None = None

    def summary(self) -> str:
        """
        One-line description of how this job ended, used in notifications.
        """
        detail = _STATUS_TEXT[self.status]
        if self.returncode is not None:
            detail += f" (exit code {self.returncode})"
        return f"Background command {self.description!r} {detail}"


class JobRegistry:
    """
    Session-scoped job registry: creates, queries and terminates background jobs, and
    hands "finished but not yet notified" jobs to the notifier.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._jobs: dict[str, Job] = {}
        self._jobs_dir = JOBS_ROOT / session_id
        # Hold watcher references so they are not garbage-collected and cancelled early
        self._watchers: set = set()

    async def spawn_shell(self, command: str, background: bool = True) -> Job:
        job_id = _new_job_id()
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._jobs_dir / f"{job_id}.log"
        log_file = open(log_path, "wb")
        # A separate process group is required to terminate descendants as a whole
        if os.name == "nt":
            group_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            group_kwargs = {"start_new_session": True}
        proc = await asyncio.create_subprocess_shell(
            command, stdout=log_file, stderr=subprocess.STDOUT, **group_kwargs
        )
        pid = proc.pid
        job = Job(
            id=job_id,
            kind="shell",
            description=command,
            log_path=log_path,
            background=background,
            kill_func=lambda: _kill_process_tree(pid),
        )
        self._jobs[job_id] = job
        watcher = asyncio.create_task(self._watch(job, proc, log_file))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        return job

    async def _watch(self, job: Job, proc: asyncio.subprocess.Process, log_file) -> None:
        returncode = await proc.wait()
        log_file.close()
        if job.status == "killed":
            return
        job.returncode = returncode
        job.status = "completed" if returncode == 0 else "failed"

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    def running(self, background: bool | None = None) -> list[Job]:
        """
        Jobs still running; pass `background` to keep only foreground commands (False)
        or background jobs (True).
        """
        return [
            j for j in self._jobs.values()
            if j.status == "running" and (background is None or j.background is background)
        ]

    def move_to_background(self) -> None:
        """
        Move every running foreground command to the background (bound to the user's
        ctrl+b).
        """
        for job in self.running(background=False):
            job.background = True

    def kill(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status != "running":
            return False
        # Mark killed and notified before terminating, so the watcher cannot later flip
        # the status to completed and notify a second time
        job.status = "killed"
        job.notified = True
        if job.kill_func:
            job.kill_func()
        return True

    def pop_unnotified(self) -> list[Job]:
        done = [j for j in self._jobs.values() if j.background and j.status != "running" and not j.notified]
        for job in done:
            job.notified = True
        return done

    def shutdown(self) -> int:
        """
        Terminate every running job (processes must not outlive the session) and return
        how many were killed.
        """
        running = self.running()
        for job in running:
            self.kill(job.id)
        return len(running)


def _kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
