"""
Composite agent dependencies: encapsulates ReadFileState (for file tools) and
TasksStore (for task tools) into a single container for unified injection via RunContext.
"""
from dataclasses import dataclass

from tasks_store import TasksStore

from .file_state import ReadFileState

from file_history import FileHistory

from background_jobs import JobRegistry


@dataclass
class AgentDeps:
    read_file_state: ReadFileState
    tasks_store: TasksStore
    file_history: FileHistory | None = None
    job_registry: JobRegistry | None = None
