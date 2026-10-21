"""
Background memory agent: after each turn, fork the conversation to extract
memories worth saving, and periodically consolidate the memory files.

Both jobs share one fork agent: same model and instructions as the main agent, but
only file tools are attached and writes are gated to the memory directory.
"""

import asyncio
import time
from pathlib import Path

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import Hooks
from pydantic_ai.usage import UsageLimits

from agent.deps import AgentDeps
from agent.file_state import ReadFileState
from agent.instructions import INSTRUCTIONS
from agent.model import model
from agent.tools.file import edit_file, read_file, write_file
from core.console import print_step
from state import sessions as session

from . import store

# Maximum model requests allowed per background run, to stop a runaway loop
MAX_FORK_REQUESTS = 10

# Two gates for automatic consolidation: at least this many hours since the last
# run, and at least this many sessions touched in the meantime
DREAM_MIN_HOURS = 24
DREAM_MIN_SESSIONS = 5

fork_hooks = Hooks()


@fork_hooks.on.tool_execute
async def _memory_write_gate(ctx, *, call, tool_def, args, handler):
    """
    The background agent must not touch the user's project: reads are allowed,
    writes only inside the memory directory, and everything else is denied.
    It runs silently in the background, so there is no approval prompt to fall
    back on; denial is the only answer available.
    """
    if call.tool_name == "read_file":
        return await handler(args)
    if call.tool_name in ("write_file", "edit_file") and store.is_memory_path(
        str(args.get("path", ""))
    ):
        return await handler(args)
    return f"The background memory agent may only write files inside the memory directory; this {call.tool_name} call was denied."


# Fork agent: same instructions as the main agent (including the memory
# conventions), with only the three file tools attached
fork_agent = Agent(
    model,
    instructions=INSTRUCTIONS,
    tools=[read_file, Tool(edit_file, sequential=True), Tool(write_file, sequential=True)],
    deps_type=AgentDeps,
    capabilities=[fork_hooks],
)


def _fork_deps() -> AgentDeps:
    # The fork runs once with a fresh readFileState; no task tools are attached,
    # so tasks_store stays empty
    return AgentDeps(read_file_state=ReadFileState(), tasks_store=None)


def _manifest_section() -> str:
    """
    Pre-inject the existing memory manifest into the prompt, saving the fork a
    turn spent listing the directory.
    """
    headers = store.scan_memory_files()
    if not headers:
        return ""
    return (
        "\n## Existing memory files\n\n"
        + store.format_manifest(headers)
        + "\n\nCheck this manifest before writing - update an existing file instead of creating a duplicate."
    )


def _written_memory_files(messages) -> list[str]:
    """
    Collect the memory filenames written during one run (excluding the MEMORY.md
    index), for the completion notice.
    """
    written = []
    for message in messages:
        for part in message.parts:
            if getattr(part, "part_kind", "") != "tool-call":
                continue
            if part.tool_name not in ("write_file", "edit_file"):
                continue
            path = str(part.args_as_dict().get("path", ""))
            if store.is_memory_path(path):
                name = Path(path).name
                if name != "MEMORY.md" and name not in written:
                    written.append(name)
    return written


# ---------- Extraction: after each turn, check for anything not yet recorded ----------

_EXTRACT_PROMPT = """You are now the memory extraction subagent. Analyze the conversation above and use its content to update your persistent memory.

You may only use content that appears in the conversation to update memory. Do not spend any turn on further investigation - do not read source code to confirm that a pattern exists, and do not run commands.
Your turns are limited. edit_file requires reading the same file with read_file first, so the efficient strategy is: read every file you might update in parallel in the first turn, write in parallel in the second, and do not interleave reads and writes across extra turns.
Follow the memory system conventions in the system prompt for what to record, what not to record, and how to save it; if nothing is worth saving, just reply "nothing to save" - do not save for the sake of saving.
{manifest}"""


async def _extract(history: list) -> None:
    """
    Fork the conversation to let the agent extract memories, then echo which
    memory files were written.
    """
    result = await fork_agent.run(
        _EXTRACT_PROMPT.format(manifest=_manifest_section()),
        message_history=history,
        deps=_fork_deps(),
        usage_limits=UsageLimits(request_limit=MAX_FORK_REQUESTS),
    )
    written = _written_memory_files(result.new_messages())
    if written:
        print_step(
            "[cyan]✦ memory[/]",
            f"[cyan dim]Background extraction saved memories: {', '.join(written)}[/]",
        )


# ---------- Dream: periodic memory consolidation ----------

_DREAM_PROMPT = """# Dream: memory consolidation

Do a round of dream now - review and reorganize your memory files, settling what has accumulated recently into durable, well-organized memories so future sessions can get up to speed quickly.

Memory directory: {memory_dir}
The directory already exists; write directly, do not mkdir.

---

## Phase 1: Take stock

- Read MEMORY.md to learn the current index
- Against the manifest below, skim memory files that may overlap or be stale - the goal later is to improve them, not to write duplicates

Current memory file manifest:
{manifest}

## Phase 2: Gather

Spot the signals worth handling:
- Memories on the same topic spread across multiple files
- Memories that contradict each other
- Memories that clearly conflict with current facts or are obviously stale

## Phase 3: Consolidate

For each topic worth keeping, write or update a memory file. Priorities:
- Merge new signals into an existing topic file instead of creating an approximate duplicate; several memories on the same topic that differ only by scenario should also go into one file as separate entries (e.g. three commit message conventions become a single "commit conventions" memory)
- Convert relative dates like "yesterday" or "next week" into absolute dates so they stay readable as time passes
- Delete facts that have been disproven - if current observation overturns an old memory, fix it at the source
- Clear files that were merged away with write_file (empty files no longer participate in recall)

## Phase 4: Prune

Update MEMORY.md and keep it under 200 lines and 25KB. It is an index, not a content dump - one line per entry, at most 150 characters: `- [title](filename.md) - one-line hook`. Never write memory content directly into it.
- Remove lines pointing at stale, emptied or superseded memories
- An index line longer than 200 characters means content landed in the wrong place: shorten the line and move details into the memory file
- Add pointer lines for important new memories
- When two files disagree, fix the wrong one

---

Finish with a brief summary of what you merged, updated and pruned. If memory was already compact and unchanged, just say so.
"""


def _lock_path() -> Path:
    return store.memory_dir() / ".consolidate-lock"


def _dream_due(current_session_id: str) -> bool:
    """
    Check the automatic consolidation gates; the lock file's mtime records the
    time of the last consolidation.
    """
    lock = _lock_path()
    last = lock.stat().st_mtime if lock.exists() else 0.0
    if time.time() - last < DREAM_MIN_HOURS * 3600:
        return False
    if not session.SESSIONS_DIR.exists():
        return False
    # Count transcripts touched since the last consolidation, excluding the current
    # session - it is still in progress
    touched = [
        p
        for p in session.SESSIONS_DIR.glob(f"*/{session.TRANSCRIPT_NAME}")
        if p.parent.name != current_session_id and p.stat().st_mtime > last
    ]
    return len(touched) >= DREAM_MIN_SESSIONS


async def dream(history: list, *, force: bool = False, session_id: str = "") -> None:
    """
    Consolidation entry point: automatic runs must pass the gates, while the
    /dream command uses force to run immediately.
    """
    if not force and not _dream_due(session_id):
        return
    headers = store.scan_memory_files()
    if not headers:
        if force:
            print_step("[cyan]✦ memory[/]", "[cyan dim]No memories yet; nothing to consolidate[/]")
        return
    # The same fork agent does the work; only the task passed to run() changes to
    # the dream consolidation prompt
    result = await fork_agent.run(
        _DREAM_PROMPT.format(
            memory_dir=store.memory_dir(), manifest=store.format_manifest(headers)
        ),
        message_history=history,
        deps=_fork_deps(),
        usage_limits=UsageLimits(request_limit=MAX_FORK_REQUESTS),
    )
    # Touch the lock file so its mtime records the time of this consolidation
    _lock_path().touch()
    print_step("[cyan]✦ memory[/]", f"[cyan dim]{result.output}[/]")


# ---------- Background scheduling after each turn ----------

# Hold references to background tasks so they are not garbage-collected and
# cancelled early
_background_tasks: set = set()


def schedule(state, new_messages) -> None:
    """
    Fire-and-forget after each turn: extract memories for this turn, then check
    whether consolidation is due. The user can keep talking without being blocked.
    """
    # Skip this turn if the previous background task has not finished: two
    # concurrent extractions read the same manifest and write duplicate memories
    if any(not t.done() for t in _background_tasks):
        return
    task = asyncio.create_task(
        _run_background(list(state.history), list(new_messages), state.session_id)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def drain(timeout: float = 60) -> None:
    """
    Called before exit: give in-flight background memory tasks up to `timeout`
    seconds to finish, so extraction is not killed halfway through writing.
    """
    tasks = [t for t in _background_tasks if not t.done()]
    if not tasks:
        return
    print_step(
        "[dim]◇ memory[/]",
        "[dim]Waiting for background memory tasks to finish (Ctrl+C to force quit)...[/]",
    )
    await asyncio.wait(tasks, timeout=timeout)


async def _run_background(history: list, new_messages: list, session_id: str) -> None:
    try:
        # If the main conversation already wrote memory this turn, skip extraction
        if not store.has_memory_writes(new_messages):
            await _extract(history)
        await dream(history, session_id=session_id)
    except Exception as e:
        # Background failures must not disturb the main flow; one line is enough
        print_step(
            "[yellow]⚠ memory[/]",
            f"[yellow]Background memory task failed: {type(e).__name__}: {e}[/]",
        )
