"""
CLI command handlers and terminal rendering utilities for the Coding Agent.

Manages interactive session lifecycle (new, resume, exit, status), formatted
console output with Rich, Markdown rendering overrides, and message step replay.
"""
import difflib
from dataclasses import dataclass, field
from typing import Callable, Optional

import questionary
from prompt_toolkit.application import in_terminal
from rich.markdown import Heading, Markdown
from rich.markup import escape
from rich.padding import Padding

import permissions
import session
from agent import ReadFileState
from tasks_store import TasksStore
from memory import background, store
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
    # Session-scoped ReadFileState injected into file tools; replaced with a fresh instance on session switch
    read_file_state: ReadFileState = field(default_factory=ReadFileState)
    # Session TasksStore injected into task tools
    tasks_store: TasksStore | None = None
    # Memory filenames already recalled and injected in this session; prevents duplicate injections
    surfaced_memories: set = field(default_factory=set)


@dataclass
class Command:
    name: str
    description: str
    # a handler returning False means the main loop should exit
    handler: Callable[["SessionState"], bool]


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


# Maximum lines to display in read/write previews; excess lines are collapsed into "… +N lines"
_PREVIEW_MAX_LINES = 8


def _preview_numbered(content: str) -> Optional[str]:
    """
    Formats content with green line numbers and truncates for write_file previews.
    """
    lines = content.splitlines()
    if not lines:
        return None
    width = len(str(len(lines)))
    shown = lines[:_PREVIEW_MAX_LINES]
    out = [f"  [green]{i:>{width}}[/] {escape(ln)}" for i, ln in enumerate(shown, 1)]
    if len(lines) > _PREVIEW_MAX_LINES:
        out.append(f"  [dim]… +{len(lines) - _PREVIEW_MAX_LINES} lines[/]")
    return "\n".join(out)


def _preview_diff(old: str, new: str) -> str:
    """
    Renders old -> new diff using difflib: red for deletions, green for additions, dim for unchanged lines.
    """
    out = []
    for line in difflib.ndiff(old.splitlines(), new.splitlines()):
        tag, body = line[:2], escape(line[2:])
        if tag == "- ":
            out.append(f"  [red]- {body}[/]")
        elif tag == "+ ":
            out.append(f"  [green]+ {body}[/]")
        elif tag == "  ":
            out.append(f"  [dim]  {body}[/]")
        # Skip '? ' intra-line hints generated by difflib
    return "\n".join(out)


def _print_file_op(part) -> bool:
    """
    Rich rendering for file tools: diffs for edit, line-numbered previews for write,
    and truncated outputs for read. Returns True if handled.
    """
    # Explicitly dispatched by tool name without a global registry to avoid an inverted dependency from agent core onto UI
    kind = part.part_kind
    if kind == "tool-call" and part.tool_name in ("edit_file", "write_file"):
        args = part.args_as_dict()
        path = escape(str(args.get("path", "")))
        # Share consistent header formatting for edit/write; vary only the body preview
        console.print("[yellow]⏺ tool_call[/]")
        console.print(Padding(f"[yellow dim]{part.tool_name}({path})[/]", (0, 0, 0, 2)))
        if part.tool_name == "edit_file":
            console.print(_preview_diff(args.get("old_string", ""), args.get("new_string", "")))
        else:
            block = _preview_numbered(args.get("content", ""))
            if block:
                console.print(block)
        return True
    if kind == "tool-return" and part.tool_name == "read_file":
        # read_file returns pre-numbered lines; truncate by line count for preview
        console.print("[magenta]✔ tool_return[/]")
        console.print(Padding("[magenta dim]read_file[/]", (0, 0, 0, 2)))
        lines = str(part.content).splitlines()
        for ln in lines[:_PREVIEW_MAX_LINES]:
            console.print(f"  [dim]{escape(ln)}[/]")
        if len(lines) > _PREVIEW_MAX_LINES:
            console.print(f"  [dim]… +{len(lines) - _PREVIEW_MAX_LINES} lines[/]")
        return True
    return False


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
    # File tools (read/edit/write) use specialized rich rendering
    if _print_file_op(part):
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
    Starts a new session: clears history, token counters, API call logs, and assigns a new session ID.
    """
    state.history.clear()
    state.input_tokens = 0
    state.output_tokens = 0
    state.last_api_calls.clear()
    state.session_id = session.new_session_id()
    # Permission whitelists are session-scoped; 'always allow' choices do not carry over to new sessions
    permissions.state.session_allowed.clear()
    # ReadFileState is session-scoped; resets to empty so cached files do not carry over
    state.read_file_state = ReadFileState()
    # TasksStore is isolated per session (~/.my-claude-code/tasks/<session_id>/); instantiating a new store updates the TUI task panel on the next redraw
    state.tasks_store = TasksStore(session_id=state.session_id)
    # Recall deduplication set is session-scoped; allows past memories to be surfaced again in the new session
    state.surfaced_memories = set()
    console.print("Started new session\n")
    return True


def _summary_line(mtime, prompt: str) -> str:
    """
    Format a session display line: modification time + initial user prompt snippet.
    """
    prompt = " ".join(str(prompt).split())
    if len(prompt) > 50:
        prompt = prompt[:50] + "..."
    return f"{mtime:%m-%d %H:%M}  {prompt}"


async def cmd_resume(state: SessionState) -> bool:
    """
    Lists historical sessions for the current project and restores conversation history upon selection.
    Runs asynchronously within the REPL event loop: in_terminal yields the terminal to questionary,
    restoring the prompt upon completion.
    """
    sessions = session.list_sessions()
    if not sessions:
        console.print("(No session history found for the current project)\n")
        return True

    choices = [
        questionary.Choice(title=_summary_line(mtime, prompt), value=sid)
        for sid, mtime, prompt in sessions
    ]
    async with in_terminal():
        selected = await questionary.select(
            "Select a session to resume (arrow keys to navigate, Enter to confirm):",
            choices=choices,
        ).ask_async()
    # Selection cancelled via Ctrl+C
    if selected is None:
        return True

    # Restore conversation history and point session ID to the chosen past session so subsequent messages append to the same file
    state.history = session.load_history(selected)
    state.session_id = selected
    # Permission whitelist is session-scoped; clear upon switching sessions
    permissions.state.session_allowed.clear()
    # ReadFileState is session-scoped; replace with a clean instance. Although read history exists in transcripts, file mtimes on disk may have changed since exit; requiring a re-read on initial edit ensures consistency
    state.read_file_state = ReadFileState()
    # Point TasksStore to the resumed session directory to reconnect disk-backed tasks (each stored as standalone JSON)
    state.tasks_store = TasksStore(session_id=state.session_id)
    # Reset recall deduplication set; previously surfaced memories may be recalled once more without significant downside
    state.surfaced_memories = set()

    # Recompute cumulative token usage from per-response usage entries in jsonl
    state.input_tokens = sum(
        m.usage.input_tokens for m in state.history if m.kind == "response"
    )
    state.output_tokens = sum(
        m.usage.output_tokens for m in state.history if m.kind == "response"
    )
    # Clear in-memory API call logs since they cannot be reconstructed from disk
    state.last_api_calls.clear()

    # Replay restored conversation transcript to the terminal
    console.print(f"\nResumed session {selected[:8]} ({len(state.history)} messages):\n")
    for msg in state.history:
        for part in msg.parts:
            # Replay shares identical rendering logic with live streaming
            print_part(part)
    console.print()
    return True


def cmd_status(state: SessionState) -> bool:
    console.print(f"Model:            {state.model_name}")
    console.print(f"Permission mode:  {permissions.state.mode}")
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


def cmd_memory(state: SessionState) -> bool:
    """
    Displays the memory directory, all memory files, and the MEMORY.md index.
    """
    console.print(f"Memory directory: {store.memory_dir()}\n")
    headers = store.scan_memory_files()
    if not headers:
        console.print("(No memories accumulated yet; memories build up across conversations)\n")
        return True
    for h in headers:
        age = store.age_text(store.age_days(h.mtime))
        console.print(escape(f"  [{h.type or '?'}] {h.filename} ({age})"))
        console.print(f"      [dim]{escape(h.description)}[/]")
    console.print()
    index = store.read_index()
    if index:
        console.print("MEMORY.md Index:")
        console.print(Padding(Markdown(index), (0, 0, 0, 2)))
        console.print()
    return True


async def cmd_dream(state: SessionState) -> bool:
    """
    Manually triggers memory consolidation and sleep phase, bypassing automated gates for immediate testing.
    """
    console.print("Consolidating memories, this may take a moment...\n")
    try:
        await background.dream(list(state.history), force=True, session_id=state.session_id)
    except Exception as e:
        console.print(f"[red]Consolidation failed: {type(e).__name__}: {e}[/]\n")
    console.print()
    return True


COMMANDS = {
    "new": Command("new", "start a new session", cmd_new),
    "resume": Command("resume", "resume a history session", cmd_resume),
    "status": Command("status", "show the current session state", cmd_status),
    "api-detail": Command("api-detail", "show model API call details for the last round", cmd_api_detail),
    "help": Command("help", "list available commands", cmd_help),
    "exit": Command("exit", "quit the program", cmd_exit),
    "memory": Command("memory", "inspect long-term memories", cmd_memory),
    "dream": Command("dream", "consolidate and compact long-term memories now", cmd_dream),
}
