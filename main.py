"""
Main application entry point and event loop for the CLI agent.

Initializes session state, coordinates the persistent REPL interface,
dispatches slash commands, and steps through Pydantic AI agent graph nodes
to stream thoughts, tool calls, and responses in real time.
"""
import asyncio
import os

from prompt_toolkit import PromptSession
from pydantic_ai import Agent
from pydantic_graph import End

import session

from ui.input_ui import Repl

from agent import agent, MODEL_NAME, api_call_log
from ui.commands import (
    COMMANDS,
    SessionState,
    console,
    print_divider,
    print_part,
    print_welcome_banner,
)

os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

# PromptSession beats the builtin input(): it supports cursor movement for
# editing, and remembers the input history of this run so you can scroll back
# through it with the arrow keys.
prompt_session = PromptSession()


def read_user_input():
    """
    Print a top rule, read one line of input, then print a bottom rule so the
    input keeps clear boundaries in the scrollback. Returns None when the user
    wants to quit (Ctrl-C / Ctrl-D).
    """
    print_divider()
    try:
        user_input = prompt_session.prompt("❯ ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    print_divider()
    return user_input


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
    cmd_name = user_input[1:].split()[0]
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

    async with agent.iter(user_input, message_history=state.history) as run:
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


async def main():
    state = SessionState(
        model_name=MODEL_NAME,
        session_id=session.new_session_id(),
    )
    print_welcome_banner("my-claude-code")

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
        if action == "continue":
            return

        # Core agent loop: show 'Working...' on start, cleared on completion or interrupt;
        # ESC / Ctrl+C interrupts execution midway
        repl.start_working()
        await run_agent_loop(user_input, state)

    await repl.run(on_submit)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        pass
