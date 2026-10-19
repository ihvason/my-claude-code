"""
Main application entry point and event loop for the CLI agent.

Initializes session state, coordinates the persistent REPL interface,
dispatches slash commands, and steps through Pydantic AI agent graph nodes
to stream thoughts, tool calls, and responses in real time.
"""
import asyncio
import os

from pydantic_ai import Agent
from pydantic_graph import End

import session
import compact
import mcp_servers
from agent import agent, MODEL_NAME, api_call_log
from agent.deps import AgentDeps
from agent.reminders import build_job_reminder_text
from tasks_store import TasksStore
from file_history import FileHistory
from background_jobs import JobRegistry
from memory import background, recall, store
from ui.commands import (
    COMMANDS,
    SessionState,
    console,
    print_part,
    print_welcome_banner,
)
from ui.input_ui import Repl
from mentions import build_mention_messages, extract_at_mentions


os.environ["PYDANTIC_AI_NO_BANNER"] = "1"


async def handle_command(user_input, state):
    """
    Handle input that starts with "/".
    Returns 'pass' when this is not a command, so the main loop falls through and
    hands the input to the agent;
    'continue' when the command was handled and the loop should start over;
    'break' when the command asks to leave the loop.
    """
    if not user_input.startswith("/"):
        return "pass"
    parts = user_input[1:].split()
    if not parts:
        # Bare "/" carries no command name; report it instead of falling through
        console.print("Type a command after '/'. Use /help to list commands.\n")
        return "continue"
    cmd_name = parts[0]
    command = COMMANDS.get(cmd_name)
    if command is None:
        console.print(f"Unknown command: /{cmd_name}. Type /help to list commands.\n")
        return "continue"
    result = command.handler(state)
    if asyncio.iscoroutine(result):
        result = await result
    return "continue" if result else "break"


def apply_result(state, result):
    """
    After one agent turn, sync the result into SessionState and render the new
    intermediate steps.
    """
    state.history = result.all_messages()
    usage = result.usage
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens
    state.last_api_calls = list(api_call_log)

    # Append this turn's new messages to the session file.
    session.append_messages(state.session_id, result.new_messages())


async def run_agent_loop(user_input, state):
    """
    Expand agent.run_sync() to step through the agent loop node by node with real-time logging.
    """
    api_call_log.clear()

    # deps bundles this session's readFileState, tasksStore, file checkpoints and job
    # registry into AgentDeps for the tool layer, where tools reach them via ctx.deps.*
    deps = AgentDeps(
        read_file_state=state.read_file_state,
        tasks_store=state.tasks_store,
        file_history=state.file_history,
        job_registry=state.job_registry,
    )
    async with agent.iter(
        user_input, message_history=state.history, deps=deps,
        toolsets=mcp_servers.active_toolsets(),
    ) as run:
        node = run.next_node

        while not isinstance(node, End):
            node = await run.next(node)

            if Agent.is_call_tools_node(node):
                for part in node.model_response.parts:
                    print_part(part)

            elif Agent.is_model_request_node(node):
                for part in node.request.parts:
                    if part.part_kind == "tool-return":
                        print_part(part)

    apply_result(state, run.result)
    console.print()
    return run.result


def inject_at_mentions(user_input, state):
    """
    Parses @path references in user input and injects mock read_file tool invocations
    into the conversation history before running the agent.
    """
    paths = extract_at_mentions(user_input)
    if not paths:
        return
    mention_messages = build_mention_messages(paths, state.read_file_state)
    if not mention_messages:
        return
    # Inject into history so the model perceives the files as already read
    state.history += mention_messages
    # Persist injected files to allow restoration via /resume
    session.append_messages(state.session_id, mention_messages)
    # Echo injected files to the terminal to confirm the @ reference took effect
    for msg in mention_messages:
        for part in msg.parts:
            print_part(part)


async def main():
    session_id = session.new_session_id()
    # Create the memory directory at startup: the system prompt promises it already
    # exists, so the model need not waste a turn confirming it.
    store.ensure_memory_dir()
    tasks_store = TasksStore(session_id=session_id)
    state = SessionState(
        model_name=MODEL_NAME,
        session_id=session_id,
        tasks_store=tasks_store,
        file_history=FileHistory(session_id=session_id),
        job_registry=JobRegistry(session_id=session_id),
    )
    print_welcome_banner("my-claude-code")

    # Spinner while connecting; otherwise a cold start that fetches packages looks like a hang
    with console.status("Connecting MCP server..."):
        mcp_summary = await mcp_servers.startup()
    if mcp_summary:
        console.print(mcp_summary + "\n")

    # Persistent input bar: remains visible at the bottom throughout the session
    repl = Repl(state)

    async def on_submit(user_input):
        # Triggered on each Enter submission
        # Process slash commands first
        action = await handle_command(user_input, state)
        if action == "break":
            # Command requested exit: shut down the REPL
            repl.exit()
            return
        # A command may hand back text to edit and resend (e.g. /rewind); put it in
        # the input box instead of running the agent
        if state.pending_input:
            repl.prefill(state.pending_input)
            state.pending_input = ""
            return
        if action == "continue":
            return

        # Record the prompt in the session metadata so /resume can list the session
        # without opening the transcript
        session.note_prompt(state.session_id, user_input, MODEL_NAME)

        # Check the context watermark before sending; compact first if it is over the threshold
        await compact.auto_compact_if_needed(state)

        # Create the checkpoint: @ mentions and memory recall have not been injected
        # yet, so len(history) is the clean rewind index. The task snapshot is taken
        # at the same moment so both channels roll back to the same point.
        state.file_history.make_checkpoint(
            len(state.history),
            user_input,
            tasks=state.tasks_store.snapshot() if state.tasks_store else None,
        )

        # Parse @ mentions and inject referenced files as mock read_file calls into history
        inject_at_mentions(user_input, state)

        # Core agent loop: show 'Working...' on start, cleared on completion or interrupt;
        # ESC / Ctrl+C interrupts execution midway
        repl.start_working()

        # Recall relevant memories into the history so the model builds on what it
        # has learned before handling this input
        await recall.inject_memories(user_input, state)
        result = await run_agent_loop(user_input, state)
        # After the turn, extract memories in the background and check whether
        # periodic consolidation is due
        background.schedule(state, result.new_messages())

    async def on_system(text):
        # System-triggered turn (background job completion notice): skip checkpointing,
        # @ parsing and memory recall, and run the agent directly
        await compact.auto_compact_if_needed(state)
        repl.start_working()
        result = await run_agent_loop(text, state)
        background.schedule(state, result.new_messages())

    async def watch_jobs():
        # Idle wake-up: if a job finishes while the REPL is idle, submit the notice as a
        # system turn so the model reports it
        try:
            while True:
                await asyncio.sleep(1)
                if not repl.is_idle:
                    continue
                text = build_job_reminder_text(state.job_registry)
                if text:
                    repl.submit_system(text)
        except asyncio.CancelledError:
            pass

    job_watcher = asyncio.ensure_future(watch_jobs())
    try:
        await repl.run(on_submit, on_system)
    finally:
        job_watcher.cancel()

    # Before exiting, give in-flight background memory tasks time to finish;
    # killing extraction midway leaves half-written memories behind
    await background.drain()

    # Background jobs are real processes: terminate them instead of waiting, so none
    # outlive this session
    killed = state.job_registry.shutdown()
    if killed:
        console.print(f"[dim]Terminated {killed} still-running background job(s) on exit[/]")

    await mcp_servers.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        pass
