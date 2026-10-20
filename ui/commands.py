import difflib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import questionary
from prompt_toolkit.application import Application, in_terminal
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.markdown import Heading, Markdown
from rich.markup import escape
from rich.padding import Padding

import compact
import mcp_servers
import permissions
import session
import subagents
from agent import ReadFileState
from file_history import FileHistory
from background_jobs import JobRegistry
from memory import background, store
from tasks_store import TasksStore
from .render import console, print_step, print_welcome_banner


class LeftAlignedHeading(Heading):
    """
    rich centers Markdown headings by default, which looks misaligned in a wide
    terminal; override it to align left instead.
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
    Session state shared across commands; the main loop passes it to every command
    handler.
    """
    history: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model_name: str = ""
    # Current session ID; determines which jsonl file the conversation history goes to
    session_id: str = ""
    # Records of every model API call triggered by the most recent round of user input
    last_api_calls: list = field(default_factory=list)
    # This session's readFileState, injected into the file tools; swapped for a fresh
    # instance when switching sessions
    read_file_state: ReadFileState = field(default_factory=ReadFileState)
    # This session's TasksStore, injected into the task tools. Persisted under
    # <project root>/.my-claude-code/projects/<project>/tasks/<session_id>/, so main
    # must construct it and pass it in
    tasks_store: TasksStore | None = None
    # Memory filenames already recalled and injected in this session; the same memory
    # is not injected twice
    surfaced_memories: set = field(default_factory=set)
    # This session's file checkpoints; /rewind uses them to roll back files and
    # conversation. Constructed by main and passed in
    file_history: FileHistory | None = None
    # Original prompt to refill into the input box after /rewind rolls back the
    # conversation; consumed by Repl once the command returns
    pending_input: str = ""
    # Consecutive automatic-compaction failures; retrying stops once this reaches the limit
    compact_failures: int = 0
    # This session's background job registry, injected into the shell tools; constructed
    # by main and passed in
    job_registry: JobRegistry | None = None


@dataclass
class Command:
    name: str
    description: str
    # A handler returning False means the main loop should exit
    handler: Callable[["SessionState"], bool]
    # Whether this command accepts the argument string after its name (e.g. /compact's extra instructions)
    takes_args: bool = False


def _truncate(text, limit: int = 120) -> str:
    """
    Truncate and escape; used for content that can grow long, such as tool
    arguments, return values and user input.
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
    Format a single part of a message as a string carrying Rich markup.
    Layout: icon + role label on their own line, content on the next line, with no
    "|" separator.
    """
    # Content lines are indented 2 columns to line up with the role name after the
    # 2-column icon (icon + space)
    kind = part.part_kind
    if kind == "user-prompt":
        return f"[cyan]❯ user[/]\n  {_truncate(part.content)}"
    if kind == "thinking":
        # Dim the whole thinking block to lower its visual weight; never truncate,
        # keep the full reasoning
        return f"[dim]✻ thinking[/]\n  [dim]{_full(part.content)}[/]"
    if kind == "text":
        content = (part.content or "").strip()
        if not content:
            return None
        # The assistant reply is the final answer the user cares about most; show it
        # in full
        return f"[green]● assistant[/]\n  {_full(content)}"
    if kind == "tool-call":
        # Commands and paths easily run to hundreds of characters, so relax the
        # argument limit to 500 before truncating
        return f"[yellow]⏺ tool_call[/]\n  [yellow dim]{part.tool_name}({_truncate(part.args, 500)})[/]"
    if kind == "tool-return":
        # Tool returned successfully; the label uses ✔
        return f"[magenta]✔ tool_return[/]\n  [magenta dim]{part.tool_name} -> {_truncate(part.content)}[/]"
    if kind == "retry-prompt":
        # After a tool raises ModelRetry, the SDK generates a retry-prompt to feed the
        # error back to the model; the label uses ✘ to mark the failed call
        return f"[yellow]✘ tool_retry[/]\n  [yellow dim]{part.tool_name} -> {_truncate(part.content)}[/]"
    return None


def print_assistant_markdown(content: str) -> None:
    """
    Model replies are naturally Markdown, so render them as a whole block instead of
    printing the raw text.
    """
    console.print("[green]● assistant[/]")
    # Markdown is a block-level renderable and cannot follow an inline prefix, so
    # render it on its own line; indent 2 columns to line up with the role name
    console.print(Padding(Markdown(content), (0, 0, 0, 2)))


# Maximum lines shown in read / write content previews; the rest collapses into "… +N lines"
_PREVIEW_MAX_LINES = 8


def _preview_numbered(content: str) -> Optional[str]:
    """
    Add green line numbers and truncate into a preview; used to display the content
    written by write_file.
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
    Render old -> new as a red/green diff with difflib: removed lines red, added
    lines green, unchanged lines dim.
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
        # "? " is difflib's character-level hint line; skip it
    return "\n".join(out)


def _print_file_op(part) -> bool:
    """
    Rich display for file tools: a red/green diff for edit, a numbered write preview
    for write, and numbered read results for read. Returns True when handled.
    """
    # Special-cased by tool name rather than a registry like permissions: the display
    # logic depends on rich, and having the tool layer register renderers would make
    # the agent layer depend on the UI layer, which is not worth it
    kind = part.part_kind
    if kind == "tool-call" and part.tool_name in ("edit_file", "write_file"):
        args = part.args_as_dict()
        path = escape(str(args.get("path", "")))
        # The label line for edit / write matches the "tool_name(path)" content-line
        # format, so print them uniformly; only the preview below differs
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
        # read_file already returns numbered content, so truncate the preview by line
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
    Render a single message part: assistant text goes through Markdown, file
    operations through the rich display, and every other part is a single line of
    text.
    """
    if part.part_kind == "text":
        content = (part.content or "").strip()
        if content:
            print_assistant_markdown(content)
            # Leave a blank line at the end of each role block so blocks are not cramped
            console.print()
        return
    # File tools (read / edit / write) use the dedicated rich display
    if _print_file_op(part):
        console.print()
        return
    line = _format_part_line(part)
    if line:
        label, _, body = line.partition("\n")
        # body looks like "  [markup]..."; strip the literal leading 2 spaces and let
        # print_step indent with Padding (wrapped continuation lines stay indented too)
        print_step(label, body[2:])


def print_agent_steps(new_messages) -> None:
    """
    Called from the main loop: show the intermediate steps the agent produced during
    this turn (thinking, text, tool calls, tool returns).
    """
    for msg in new_messages:
        for part in msg.parts:
            # The main loop already echoed what the user just typed
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


def _swap_job_registry(state: SessionState) -> JobRegistry:
    """
    Terminate jobs still running in the old session (real processes tied to that
    session) and return a fresh registry.
    """
    killed = state.job_registry.shutdown()
    if killed:
        console.print(f"[dim]Terminated {killed} still-running background job(s)[/]")
    return JobRegistry(session_id=state.session_id)


def cmd_new(state: SessionState) -> bool:
    """
    Start a new session: clear history, token counters and API call records, and
    switch to a new session ID.
    """
    state.history.clear()
    state.input_tokens = 0
    state.output_tokens = 0
    state.last_api_calls.clear()
    state.session_id = session.new_session_id()
    # The permission allowlist is session-scoped; "always allow for this session"
    # must not carry into a new session
    permissions.state.session_allowed.clear()
    # readFileState is session-scoped too; a new session starts blank and does not
    # inherit files read by the old one
    state.read_file_state = ReadFileState()
    # TasksStore persists inside the session bundle
    # (sessions/<session_id>/tasks/); switching sessions swaps in a new one, and the
    # task panel reflects the new store on the next repaint
    state.tasks_store = TasksStore(session_id=state.session_id)
    # The recall de-duplication set is session-scoped; old memories may be recalled
    # again in a new session
    state.surfaced_memories = set()
    # File checkpoints are also session-scoped; a new session records from scratch
    state.file_history = FileHistory(session_id=state.session_id)
    state.compact_failures = 0
    # Background jobs do not cross sessions: terminate the old ones and swap in a new registry
    state.job_registry = _swap_job_registry(state)
    console.print("Started a new session\n")
    return True


def _one_line(text, limit: int = 50) -> str:
    """
    Collapse whitespace onto one line and truncate; plain text for prompt_toolkit,
    with no Rich escaping.
    """
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _summary_line(mtime, prompt: str) -> str:
    """
    Build the display text for one session entry: modification time + a summary of
    the first user input.
    """
    return f"{mtime:%m-%d %H:%M}  {_one_line(prompt)}"


def _recount_tokens(state: SessionState) -> None:
    """
    Recount the session's token usage from the conversation history; every model
    response carries usage. Called wherever the history is replaced wholesale
    (/resume restoring it, /rewind truncating it).
    """
    state.input_tokens = sum(
        m.usage.input_tokens for m in state.history if m.kind == "response"
    )
    state.output_tokens = sum(
        m.usage.output_tokens for m in state.history if m.kind == "response"
    )


async def cmd_resume(state: SessionState) -> bool:
    """
    List this project's past sessions and restore the conversation history of the
    chosen one.
    It runs on the REPL event loop, hence async: in_terminal yields the terminal to
    questionary and restores the input box afterwards.
    """
    sessions = session.list_sessions()
    if not sessions:
        console.print("(No past sessions for this project yet)\n")
        return True

    choices = [
        questionary.Choice(title=_summary_line(mtime, prompt), value=sid)
        for sid, mtime, prompt in sessions
    ]
    async with in_terminal():
        selected = await questionary.select(
            "Select a session to resume (arrow keys to move, Enter to confirm):", choices=choices
        ).ask_async()
    # The user cancelled the selection with Ctrl+C
    if selected is None:
        return True

    # Restore the conversation history and switch the session ID to the chosen old
    # session, so later messages keep appending to the same file
    state.history = session.load_history(selected)
    state.session_id = selected
    # The permission allowlist is session-scoped; clear it when switching sessions
    permissions.state.session_allowed.clear()
    # readFileState is session-scoped too; swap in a fresh instance after switching.
    # The restored history contains read traces, but files may have changed since the
    # process exited and their mtimes are no longer trustworthy, so it is safer to
    # have the model re-read before its first edit
    state.read_file_state = ReadFileState()
    # TasksStore points at the current session_id's directory, so resuming picks the
    # old task list back up directly (each task is its own JSON file, already on disk)
    state.tasks_store = TasksStore(session_id=state.session_id)
    # Restart the recall de-duplication set from empty: memories already injected in
    # the restored history may be recalled once more, and one duplicate is harmless
    state.surfaced_memories = set()
    # File checkpoints are restored along with the session: the constructor loads
    # that session's checkpoints.json, making old checkpoints usable immediately
    state.file_history = FileHistory(session_id=state.session_id)
    state.compact_failures = 0
    # Background jobs do not cross sessions: terminate the old ones; the resumed session
    # starts with an empty registry
    state.job_registry = _swap_job_registry(state)

    # Add up the restored session's token usage
    _recount_tokens(state)
    # The last round's API call records live only in this process and cannot be
    # restored; clear them
    state.last_api_calls.clear()

    # Replay the restored conversation to the screen
    console.print(f"\nResumed session {selected[:8]} with {len(state.history)} messages:\n")
    for msg in state.history:
        for part in msg.parts:
            # Replay and live output share the same part-rendering logic
            print_part(part)
    console.print()
    return True


# Rewind picker colors: current entry highlighted blue, descriptions and footer
# muted, added/removed line counts green/red
_PICKER_STYLE = Style.from_dict({
    "question": "bold",
    "label-current": "#3b82f6 bold",
    "label": "",
    "desc": "#6b7280",
    "plus": "#10b981",
    "minus": "#ef4444",
    "footer": "#6b7280",
})


class _ListPicker:
    """
    Hand-drawn single-select picker; each option may carry one muted description
    line, visually aligned with the permission approval picker.
    """

    def __init__(self, question: str, options: list, header: list | None = None):
        # options is a list of (value, label, desc); desc is a list of FormattedText
        # fragments, None means no description line
        self.question = question
        self.options = options
        # header is a list of FormattedText fragments rendered between the question
        # and the options
        self.header = header
        self.cursor = 0
        # Value of the selected option; stays None when cancelled with Esc / Ctrl+C
        self.result = None
        self.app = self._build_app()

    def _render_question(self):
        return FormattedText([("class:question", self.question)])

    def _render_header(self):
        return FormattedText(self.header)

    def _render_options(self):
        lines: list[tuple[str, str]] = []
        for i, (_value, label, desc) in enumerate(self.options):
            is_cursor = (i == self.cursor)
            pointer = "❯" if is_cursor else " "
            cls_label = "class:label-current" if is_cursor else "class:label"
            lines.append((cls_label, f" {pointer}  {i + 1}. {label}"))
            lines.append(("", "\n"))
            if desc:
                # Indent the description line to align with the label text
                lines.append(("", "       "))
                lines.extend(desc)
                lines.append(("", "\n"))
        return FormattedText(lines)

    def _render_footer(self):
        return FormattedText([("class:footer", "  ↑↓ Select · Enter Confirm · Esc Cancel")])

    def _move(self, delta: int):
        self.cursor = (self.cursor + delta) % len(self.options)

    def _build_app(self) -> Application:
        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def _(event):
            self._move(-1)

        @kb.add("down")
        @kb.add("j")
        def _(event):
            self._move(1)

        @kb.add("enter")
        def _(event):
            self.result = self.options[self.cursor][0]
            self.app.exit()

        @kb.add("escape")
        @kb.add("c-c")
        def _(event):
            self.app.exit()

        windows = [
            Window(FormattedTextControl(self._render_question), height=1, always_hide_cursor=True),
            Window(FormattedTextControl(self._render_options), dont_extend_height=True, always_hide_cursor=True),
            Window(FormattedTextControl(self._render_footer), height=1, always_hide_cursor=True),
        ]
        if self.header:
            windows.insert(1, Window(
                FormattedTextControl(self._render_header),
                dont_extend_height=True, always_hide_cursor=True,
            ))
        layout = Layout(HSplit(windows))
        return Application(
            layout=layout,
            key_bindings=kb,
            style=_PICKER_STYLE,
            full_screen=False,
            mouse_support=False,
            # Erase the whole picker after selection so options are not left in the scrollback
            erase_when_done=True,
        )

    async def run(self):
        await self.app.run_async()
        return self.result


def _age_text(timestamp: float) -> str:
    """
    Turn a timestamp into relative time such as "14 min ago".
    """
    seconds = max(0, time.time() - timestamp)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"


def _changes_headline(changes) -> tuple:
    """
    Compress a list of FileChange into (name, insertions, deletions): a single file
    shows its filename, several show a count.
    """
    plus = sum(c.insertions for c in changes)
    minus = sum(c.deletions for c in changes)
    if len(changes) == 1:
        name = os.path.basename(changes[0].path)
    else:
        name = f"{len(changes)} files"
    return name, plus, minus


def _changes_fragments(changes) -> list:
    """
    Render a list of FileChange as picker description fragments: "main.py +12 -3" or
    "no code changes".
    """
    if not changes:
        return [("class:desc", "no code changes")]
    name, plus, minus = _changes_headline(changes)
    return [
        ("class:desc", f"{name} "),
        ("class:plus", f"+{plus}"),
        ("class:minus", f" -{minus}"),
    ]


async def cmd_rewind(state: SessionState) -> bool:
    """
    Rewind to an earlier checkpoint: first pick the checkpoint, then choose whether to
    roll back the conversation, the code, or both.
    """
    fh = state.file_history
    if fh is None or not fh.checkpoints:
        console.print("(No checkpoints to rewind to; one is created automatically each time a message is sent)\n")
        return True

    # Checkpoints listed newest first, each annotated with the change volume produced
    # by that turn
    cps = list(reversed(fh.checkpoints))
    options = [
        (i, _one_line(s.prompt), _changes_fragments(fh.turn_stats(s)))
        for i, s in enumerate(cps)
    ]
    async with in_terminal():
        picked = await _ListPicker("Rewind to before which message:", options).run()
    if picked is None:
        return True
    cp = cps[picked]
    # Tracked files edited outside the agent's file tools (manual edits, a formatter,
    # another process) are left untouched: the user's work outranks a change the agent
    # can replay, and overwriting it would be unrecoverable
    skipped = sorted(p for p in fh.externally_modified() if p in fh.versions)
    # The confirmation shows the cumulative cost: every change to undo when going from
    # the current state back to the checkpoint
    plan = fh.diff_stats(cp, skip=set(skipped))

    # The confirmation is rendered as the second-level menu's header by the picker
    header = [
        ("", "\n"),
        ("class:desc", f"  │ {_one_line(cp.prompt, 100)}\n"),
        ("class:desc", f"  │ ({_age_text(cp.timestamp)})\n"),
        ("", "\n"),
    ]
    if plan:
        name, plus, minus = _changes_headline(plan)
        header += [
            ("", "Code will restore "),
            ("class:plus", f"+{plus}"),
            ("class:minus", f" -{minus}"),
            ("", f" ({name})\n"),
        ]
    else:
        header.append(("class:desc", "No code changes\n"))
    if skipped:
        header.append(("class:desc", f"  │ {len(skipped)} file(s) changed outside the agent will be left untouched:\n"))
        for path in skipped[:5]:
            header.append(("class:desc", f"  │   {os.path.basename(path)}\n"))
        if len(skipped) > 5:
            header.append(("class:desc", f"  │   … and {len(skipped) - 5} more\n"))
        header.append(("", "\n"))

    channel_options = [
        ("both", "Rewind both conversation and code", None),
        ("conversation", "Rewind conversation only", None),
        ("code", "Rewind code only", None),
        (None, "Cancel", None),
    ]
    async with in_terminal():
        channel = await _ListPicker("Will rewind to before you sent this message:", channel_options, header=header).run()
    if channel is None:
        return True

    # Rewind files before the conversation; the order matters, because rewinding the
    # conversation drops this checkpoint and every later one
    if channel in ("both", "code"):
        fh.rewind_files(cp, skip=set(skipped))
    if channel in ("both", "conversation"):
        # Truncate the in-memory history and rewrite the session file wholesale
        state.history = state.history[: cp.history_index]
        session.rewrite_messages(state.session_id, state.history)
        # Checkpoints for the truncated conversation are meaningless now; drop them
        fh.drop_from(cp)
        # Refill the input box with the prompt that was rolled back, so the user can
        # tweak and resend it
        state.pending_input = cp.prompt
        # Recount tokens from the remaining history, clear the in-process API call
        # records, and allow memories to be recalled again
        _recount_tokens(state)
        state.last_api_calls.clear()
        state.surfaced_memories = set()
        state.compact_failures = 0
        # Tasks are conversation-level state; roll them back to the snapshot taken at
        # this checkpoint. Checkpoints written before task rollback existed have no
        # snapshot, so they are left alone.
        if state.tasks_store is not None and cp.tasks is not None:
            state.tasks_store.restore(cp.tasks)
    # Whichever channel is rewound, readFileState's recorded contents and mtimes are
    # stale and must be cleared; otherwise the read-before-write gate would allow
    # edits based on old content
    state.read_file_state = ReadFileState()

    console.print(f"Rewound to the checkpoint from {datetime.fromtimestamp(cp.timestamp):%H:%M}")
    if skipped:
        names = ", ".join(os.path.basename(p) for p in skipped[:5])
        more = f" and {len(skipped) - 5} more" if len(skipped) > 5 else ""
        console.print(f"[yellow]Kept {len(skipped)} file(s) changed outside the agent: {names}{more}[/]")
    console.print()
    return True


def cmd_status(state: SessionState) -> bool:
    console.print(f"Model:            {state.model_name}")
    console.print(f"Permission mode:  {permissions.state.mode}")
    console.print(f"History messages: {len(state.history)}")
    used = compact.context_tokens(state.history)
    threshold = compact.compact_threshold()
    if used:
        console.print(f"Context (est.):   {used:,} / {threshold:,} tokens ({used * 100 // threshold}%)")
    else:
        console.print(f"Context (est.):   no data yet (auto-compaction threshold {threshold:,} tokens)")
    console.print(f"Input tokens:     {state.input_tokens}")
    console.print(f"Output tokens:    {state.output_tokens}\n")
    return True


def cmd_memory(state: SessionState) -> bool:
    """
    Show the memory directory, all memory files, and the MEMORY.md index.
    """
    console.print(f"Memory directory: {store.memory_dir()}\n")
    headers = store.scan_memory_files()
    if not headers:
        console.print("(No memories yet; memory accumulates gradually as you talk)\n")
        return True
    for h in headers:
        age = store.age_text(store.age_days(h.mtime))
        console.print(escape(f"  [{h.type or '?'}] {h.filename} ({age})"))
        console.print(f"      [dim]{escape(h.description)}[/]")
    console.print()
    index = store.read_index()
    if index:
        console.print("MEMORY.md index:")
        console.print(Padding(Markdown(index), (0, 0, 0, 2)))
        console.print()
    return True


async def cmd_dream(state: SessionState) -> bool:
    """
    Trigger memory consolidation manually, bypassing the automatic gates so the effect
    is visible immediately.
    """
    console.print("Consolidating memory, this may take a moment...\n")
    try:
        await background.dream(list(state.history), force=True, session_id=state.session_id)
    except Exception as e:
        console.print(f"[red]Consolidation failed: {type(e).__name__}: {e}[/]\n")
    console.print()
    return True


def cmd_api_detail(state: SessionState) -> bool:
    """
    Show metadata for every model API call triggered by the most recent round of user
    input.
    """
    if not state.last_api_calls:
        console.print("(No model calls recorded yet; send a message first and come back)\n")
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


async def cmd_compact(state: SessionState, args: str = "") -> bool:
    """
    Compact the context manually; accepts extra instructions, e.g.
    /compact keep the file changes.
    """
    try:
        await compact.run_compact(state, custom_instructions=args)
    except Exception as e:
        console.print(f"[red]Compaction failed: {type(e).__name__}: {e}[/]\n")
    return True


def cmd_mcp(state: SessionState) -> bool:
    """
    Show connection status and tool inventory for every configured MCP server.
    """
    if not mcp_servers.RECORDS:
        console.print(f"No MCP servers configured. Add them to .mcp.json in the project root or to {mcp_servers.USER_CONFIG}.\n")
        return True
    for record in mcp_servers.RECORDS:
        console.print(f"[bold]{record.name}[/]  [dim]{escape(record.transport)}[/]")
        if record.status == "connected":
            console.print(f"  [green]connected[/], {len(record.tool_names)} tool(s)")
            for name in record.tool_names:
                console.print(f"    - {name}")
        else:
            console.print(f"  [red]connection failed[/]: {escape(record.error)}")
        console.print()
    return True


def cmd_jobs(state: SessionState) -> bool:
    """
    List this session's background jobs and their status.
    """
    jobs = [job for job in state.job_registry.list() if job.background]
    if not jobs:
        console.print("(No background jobs in this session yet)\n")
        return True
    for job in jobs:
        console.print(f"  {job.id} {escape(f'[{job.status}]')} {escape(job.description)}")
        console.print(f"      log: {job.log_path}")
    console.print()
    return True


def cmd_agents(state: SessionState) -> bool:
    """
    List the agent types available to run_agent.
    """
    for atype in subagents.list_agent_types():
        source = "built-in" if atype.source == "built-in" else f"custom: {atype.source}"
        console.print(f"  [bold]{atype.name}[/] [bright_black]({source})[/]")
        console.print(f"      {escape(atype.description)}")
        console.print(f"      [bright_black]Tools: {', '.join(atype.tool_names)}[/]")
    console.print()
    return True


COMMANDS = {
    "new": Command("new", "start a new session", cmd_new),
    "resume": Command("resume", "resume a past session", cmd_resume),
    "rewind": Command("rewind", "rewind to an earlier checkpoint", cmd_rewind),
    "compact": Command("compact", "compact the context (accepts extra instructions)", cmd_compact, takes_args=True),
    "status": Command("status", "show the current session state", cmd_status),
    "memory": Command("memory", "view long-term memory", cmd_memory),
    "dream": Command("dream", "consolidate long-term memory now", cmd_dream),
    "mcp": Command("mcp", "show MCP server status and tools", cmd_mcp),
    "jobs": Command("jobs", "list background jobs", cmd_jobs),
    "api-detail": Command("api-detail", "show model API call details for the last round", cmd_api_detail),
    "help": Command("help", "list available commands", cmd_help),
    "agents": Command("agents", "list the sub agent types available to run_agent", cmd_agents),
    "exit": Command("exit", "quit the program", cmd_exit),
}
