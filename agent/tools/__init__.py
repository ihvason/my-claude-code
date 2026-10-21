"""
Tool subpackage, split by responsibility: file (read/write files) / shell (run
commands) / ask_user (ask the user questions).

External code only needs:
- TOOLS: the tool list registered with the Agent
- read_and_register: reused by mentions.py to inject @ references into the
  conversation history disguised as read_file calls
"""
from pydantic_ai import Tool

# Importing shell explicitly triggers its module-level
# permissions.register_self_check side effect: the danger-pattern self-check for
# run_command must be registered at import time, otherwise dangerous commands
# would be auto-approved later in default mode
from . import shell as _shell  # noqa: F401
from .ask_user import ask_user_question
from .agents import run_agent
from .file import read_and_register, read_file, edit_file, write_file
from .shell import job_kill, run_command
from .monitor import monitor
from .tasks import task_create, task_get, task_list, task_update
from skills import load_skill

# edit_file and write_file are marked sequential=True: multiple file edits in one
# turn must run serially, otherwise they would write concurrently from the same
# stale snapshot and overwrite each other (exactly the race that readFileState +
# mtime exists to prevent)
# Mark task_update with sequential=True: concurrent updates in a single turn risk lost updates due to stale snapshots (store lacks locking).
TOOLS = [
    read_file,
    load_skill,
    Tool(edit_file, sequential=True),
    Tool(write_file, sequential=True),
    run_command,
    monitor,
    job_kill,
    run_agent,
    task_create,
    task_list,
    task_get,
    Tool(task_update, sequential=True),
    ask_user_question,
]

__all__ = ["TOOLS", "read_and_register"]
