from dataclasses import dataclass, field
from typing import Callable, Optional

import questionary
import session

from rich.markdown import Heading, Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.rule import Rule

from .render import console, print_step, print_welcome_banner


class LeftAlignedHeading(Heading):
    """
    rich renders Markdown headings centered by default, which looks misaligned in
    a wide terminal; override it to align left instead.
    """
    def __rich_console__(self, console, options):
        text = self.text
        text.justify = "left"
        yield text


# Replace the Markdown heading renderer globally
Markdown.elements["heading_open"] = LeftAlignedHeading


@dataclass
class SessionState:
    """
    Session state shared across commands; the main loop passes it to every
    command handler.
    """
    history: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model_name: str = ""
    session_id: str = ""
    # every model API call triggered by the most recent round of user input
    last_api_calls: list = field(default_factory=list)


@dataclass
class Command:
    name: str
    description: str
    # a handler returning False means the main loop should exit
    handler: Callable[["SessionState"], bool]


def print_divider() -> None:
    """
    Print a divider before every round of interaction to separate the input
    areas. Rule adapts itself to the terminal width.
    """
    console.print(Rule(style="grey50"))


def _truncate(text, limit: int = 120) -> str:
    """
    Truncate and escape, for content that can grow long such as tool arguments,
    return values and user input.
    """
    text = str(text).strip()
    text = text if len(text) <= limit else text[:limit] + "..."
    return escape(text)


def _full(text) -> str:
    """
    Show in full, escaping only and never truncating; used for thinking and
    assistant text, which is what the user cares about.
    """
    return escape(str(text).strip())


def _format_part_line(part) -> Optional[str]:
    """
    Format a single part of a message into a string carrying Rich markup.
    Layout: the icon and the role label take one line, the content moves to the
    next line, with no "|" separator.
    """
    # content lines are indented 2 columns to line up with the role name that
    # follows the icon (the icon plus its trailing space take 2 columns)
    kind = part.part_kind
    if kind == "user-prompt":
        return f"[cyan]❯ user[/]\n  {_truncate(part.content)}"
    if kind == "thinking":
        # dim the whole thinking block to lower its visual weight; never
        # truncate, keep the full reasoning
        return f"[dim]✻ thinking[/]\n  [dim]{_full(part.content)}[/]"
    if kind == "text":
        content = (part.content or "").strip()
        if not content:
            return None
        # the assistant reply is the final answer the user cares about most, so
        # show it in full
        return f"[green]● assistant[/]\n  {_full(content)}"
    if kind == "tool-call":
        # commands and paths easily run to hundreds of characters, so relax the
        # argument limit to 500 before truncating
        return f"[yellow]⏺ tool_call[/]\n  [yellow dim]{part.tool_name}({_truncate(part.args, 500)})[/]"
    if kind == "tool-return":
        return f"[magenta]✔ tool_return[/]\n  [magenta dim]{part.tool_name} -> {_truncate(part.content)}[/]"
    if kind == "retry-prompt":
        # after a tool raises ModelRetry, the SDK generates a retry-prompt that
        # feeds the error back to the model
        return f"[yellow]✘ tool_retry[/]\n  [yellow dim]{part.tool_name} -> {_truncate(part.content)}[/]"
    return None


def print_assistant_markdown(content: str) -> None:
    """
    Model replies are naturally Markdown, so render them as a whole block rather
    than printing the raw text.
    """
    console.print("[green]● assistant[/]")
    # Markdown is a block-level renderable and cannot follow an inline prefix, so
    # render it on its own line; indent 2 columns to line up with the role name
    console.print(Padding(Markdown(content), (0, 0, 0, 2)))


def print_part(part) -> None:
    """
    Render a single message part: assistant text goes through Markdown block
    rendering, every other part is a single line of text.
    """
    if part.part_kind == "text":
        content = (part.content or "").strip()
        if content:
            print_assistant_markdown(content)
            # leave a blank line at the end of each role block so blocks are not
            # crammed together
            console.print()
        return
    line = _format_part_line(part)
    if line:
        label, _, body = line.partition("\n")
        # body looks like "  [markup]..."; strip the literal leading 2 spaces and
        # let print_step indent with Padding, so wrapped continuation lines stay
        # indented too
        print_step(label, body[2:])


def print_agent_steps(new_messages) -> None:
    """
    Called from the main loop: show the intermediate steps the agent produced
    during this turn (thinking, text, tool calls, tool returns).
    """
    for msg in new_messages:
        for part in msg.parts:
            # the main loop already echoed what the user just typed
            if part.part_kind == "user-prompt":
                continue
            print_part(part)


def cmd_exit(state: SessionState) -> bool:
    console.print("Bye 👋")
    return False


def cmd_help(state: SessionState) -> bool:
    console.print("Available commands:")
    for cmd in COMMANDS.values():
        console.print(f"  /{cmd.name:<10} {cmd.description}")
    console.print()
    return True


def cmd_new(state: SessionState) -> bool:
    """
    Start a new session: clear the history, the token counters and the API call
    records.
    """
    state.history.clear()
    state.input_tokens = 0
    state.output_tokens = 0
    state.last_api_calls.clear()
    state.session_id = session.new_session_id()
    console.print("Started a new session\n")
    return True


def _summary_line(mtime, prompt: str) -> str:
    """
    Format a session display line: modification time + initial user prompt snippet.
    """
    prompt = " ".join(str(prompt).split())
    if len(prompt) > 50:
        prompt = prompt[:50] + "..."
    return f"{mtime:%m-%d %H:%M}  {prompt}"


def cmd_resume(state: SessionState) -> bool:
    sessions = session.list_sessions()
    if not sessions:
        console.print("(No session history found for the current project)\n")
        return True

    choices = [
        questionary.Choice(title=_summary_line(mtime, prompt), value=sid)
        for sid, mtime, prompt in sessions
    ]
    # Append a cancellation option
    choices.append(questionary.Choice(title="[Cancel and return]", value=None))

    selected = questionary.select(
        "Select a session to resume (Up/Down to navigate, Enter to confirm, Ctrl+C to cancel):",
        choices=choices,
        instruction=" ",
    ).ask()

    # User selected cancel or pressed Ctrl+C
    if selected is None:
        console.print("[dim](Session restoration canceled)[/]\n")
        return True

    # Restore session history
    state.history = session.load_history(selected)
    state.session_id = selected

    # Aggregate token usage across restored responses
    state.input_tokens = sum(
        m.usage.input_tokens for m in state.history if m.kind == "response"
    )
    state.output_tokens = sum(
        m.usage.output_tokens for m in state.history if m.kind == "response"
    )
    # Clear in-memory API call logs as they are ephemeral to the previous process
    state.last_api_calls.clear()

    # Replay conversation history to console
    console.print(f"\nResumed session {selected[:8]} with {len(state.history)} message(s):\n")
    for msg in state.history:
        for part in msg.parts:
            # Reuse the part rendering logic for replay
            print_part(part)
    console.print()
    return True


def cmd_status(state: SessionState) -> bool:
    console.print(f"Model:            {state.model_name}")
    console.print(f"History messages: {len(state.history)}")
    console.print(f"Input tokens:     {state.input_tokens}")
    console.print(f"Output tokens:    {state.output_tokens}\n")
    return True


def cmd_api_detail(state: SessionState) -> bool:
    """
    Show the metadata of every model API call triggered by the most recent round
    of user input.
    """
    if not state.last_api_calls:
        console.print("(No model calls recorded yet - send a message first)\n")
        return True

    console.print(f"{len(state.last_api_calls)} model API call(s) in the last round\n")

    for i, call in enumerate(state.last_api_calls, 1):
        console.print(f"[bold]Call #{i}[/]")
        console.print("  Request:")
        console.print(f"    model:        {call.model}")
        console.print(f"    messages:     {call.messages_count}")
        if call.last_part is not None:
            preview = _format_part_line(call.last_part)
            if preview:
                console.print(f"    last_message: {preview}")
        console.print(f"    tools:        {', '.join(call.tools)}")
        console.print("  Response:")
        console.print(f"    finish_reason: {call.finish_reason}")
        console.print(f"    parts:         {', '.join(call.parts_kinds)}")
        console.print(f"    usage:         input={call.input_tokens}, output={call.output_tokens}")
        console.print()
    return True


COMMANDS = {
    "new": Command("new", "start a new session", cmd_new),
    "resume": Command("resume", "resume a history session", cmd_resume),
    "status": Command("status", "show the current session state", cmd_status),
    "api-detail": Command("api-detail", "show model API call details for the last round", cmd_api_detail),
    "help": Command("help", "list available commands", cmd_help),
    "exit": Command("exit", "quit the program", cmd_exit),
}
