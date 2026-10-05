from prompt_toolkit import PromptSession

from agent import agent, MODEL_NAME, api_call_log
from ui.commands import (
    COMMANDS,
    SessionState,
    console,
    print_agent_steps,
    print_divider,
    print_welcome_banner,
)

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


def handle_command(user_input, state):
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
    return "continue" if command.handler(state) else "break"


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
    # result.new_messages() already yields only this turn's messages, so there is
    # no offset to track by hand
    print_agent_steps(result.new_messages())


def main():
    state = SessionState(model_name=MODEL_NAME)
    print_welcome_banner("Coding Agent")

    while True:
        user_input = read_user_input()
        if user_input is None:
            break
        if not user_input:
            continue

        action = handle_command(user_input, state)
        if action == "break":
            break
        if action == "continue":
            continue

        # Clear the collection buffer first so that /api-detail reports the calls
        # of this turn only
        api_call_log.clear()
        result = agent.run_sync(user_input, message_history=state.history)
        apply_result(state, result)


if __name__ == "__main__":
    main()
