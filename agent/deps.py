"""
Composite agent dependencies: encapsulates ReadFileState (for file tools) and
TasksStore (for task tools) into a single container for unified injection via RunContext.
"""
from dataclasses import dataclass
from pathlib import Path

from state.tasks import TasksStore

from .file_state import ReadFileState

from state.file_history import FileHistory

from state.jobs import Job, JobRegistry


@dataclass
class AgentDeps:
    read_file_state: ReadFileState
    tasks_store: TasksStore
    file_history: FileHistory | None = None
    job_registry: JobRegistry | None = None
    # Set only on sub agent runs: points at its own job, used to identify the requester
    # when an approval request bubbles up to the user
    subagent_job: Job | None = None
    # The current session's plan file; in plan mode it is the only file the main agent may write
    plan_file: Path | None = None
