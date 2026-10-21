"""
Interactive REPL interface built on prompt_toolkit.

Maintains a persistent bottom input bar while streaming agent output above via
patch_stdout. Supports real-time execution cancellation (Esc / Ctrl+C), command
history, dynamic spinners, and cycling permission modes via Shift+Tab.
"""
import asyncio
import platform
import re
import time
from html import escape as html_escape

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import ConditionalContainer, Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout
from rich.markup import escape

import permissions
import images
from mentions import list_candidate_files
from .render import console, print_system_text

# Persistent input area: remains pinned to the bottom throughout the session.
# Agent output streams above via patch_stdout; requests can be interrupted immediately via ESC / Ctrl+C.

# Animation frames for the working spinner
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Row rendering templates for the task panel; keys correspond to task.status, and {s} represents the escaped subject
_TASK_ROW_TEMPLATES = {
    "completed": "  <ansigreen>✔</ansigreen> <s><ansibrightblack>{s}</ansibrightblack></s>",
    "in_progress": "  <ansiyellow>▶</ansiyellow> <b>{s}</b>",
    "pending": "  <ansibrightblack>☐ {s}</ansibrightblack>",
}

# Placeholder inserted into the input line for a pasted image; backspace deletes it whole
_IMAGE_PLACEHOLDER = re.compile(r"\[Image #(\d+)\]")
# Same placeholder anchored at the end, to tell whether the cursor sits right after one
_IMAGE_PLACEHOLDER_AT_END = re.compile(r"\[Image #(\d+)\]$")


class AtFileCompleter(Completer):
    """
    Autocompletes @ file references: yields matching paths when typing an @token.
    """
    def get_completions(self, document, complete_event):
        # Inspect text before cursor to capture active @token (non-whitespace following @)
        match = re.search(r"@(\S*)$", document.text_before_cursor)
        if match is None:
            return
        token = match.group(1).lower()
        for path in list_candidate_files():
            # Case-insensitive substring match
            if token in path.lower():
                # Negative start_position replaces the entered prefix with the full path upon selection
                yield Completion(path, start_position=-len(match.group(1)))


class Repl:
    # Persistent input area; on_submit callback is injected by main to handle each input line

    def __init__(self, state):
        self.state = state
        # Injected by main: async handler for submitted input (slash commands or agent queries)
        self._on_submit = None
        # Injected by main: async handler for system-triggered turns (e.g. background job notices)
        self._on_system = None
        # Background task handling current input; targets ESC / Ctrl+C cancellation. None when idle.
        self._task = None
        # Tracks active model execution to toggle the working indicator
        self.working = False
        self._work_start = 0.0
        self._frame = 0
        # Images pasted but not sent yet; they leave with the next Enter
        self._pending_images = []
        # Guards against several clipboard reads running at once
        self._pasting = False
        # Configure input buffer with AtFileCompleter; complete_while_typing enables inline popups without Tab
        self._buffer = Buffer(
            multiline=False,
            history=InMemoryHistory(),
            completer=AtFileCompleter(),
            complete_while_typing=True,
        )
        self.app = self._build_app()

    def _prompt_prefix(self, line_number, wrap_count):
        return HTML("<ansicyan>❯ </ansicyan>")

    def _task_lines(self):
        # Invoked on every spinner tick (~100ms); backed by an in-memory list() cache for zero render overhead
        tasks = self.state.tasks_store.list()
        if not tasks:
            return HTML("")
        rows = [_TASK_ROW_TEMPLATES[t.status].format(s=html_escape(t.subject)) for t in tasks]
        return HTML("\n".join(rows))

    def _working_line(self):
        # Header above the input prompt: spinner + elapsed time + cancellation hint (active only during requests)
        frame = _SPINNER[self._frame % len(_SPINNER)]
        elapsed = int(time.monotonic() - self._work_start)
        hint = ""
        if self.state.job_registry.running(background=False):
            hint = " · ctrl+b to background"
        return HTML(f"<ansigreen>{frame}</ansigreen> <b>Working…</b><ansibrightblack>({elapsed}s elapsed · esc to cancel{hint})</ansibrightblack>")

    def _mode_line(self):
        # Line under the input box: current permission mode and switch hint, plus a
        # running background job count per kind
        mode = permissions.state.mode
        line = f"  <ansimagenta><b>▶▶ {mode}</b></ansimagenta><ansibrightblack>(Shift+Tab to switch)</ansibrightblack>"
        running = self.state.job_registry.running(background=True)
        for kind in ("shell", "agent", "monitor"):
            count = sum(1 for job in running if job.kind == kind)
            if count:
                line += f"<ansibrightblack> · </ansibrightblack><ansicyan>{count} {kind}</ansicyan>"
        return HTML(line)

    def _divider(self):
        # Horizontal divider
        return Window(height=1, char="─", style="fg:ansibrightblack")

    def _build_app(self):
        # Top-down layout: working indicator (active during requests), task panel (shown when tasks exist),
        # divider, input bar, divider, permission mode footer
        body = HSplit(
            [
                # Working indicator sits on top; task panel sits immediately below, stacked above the input divider
                ConditionalContainer(
                    Window(FormattedTextControl(self._working_line), height=1),
                    filter=Condition(lambda: self.working),
                ),
                # Task panel adjusts dynamically to content height; collapses to 0 height when empty without manual toggling
                ConditionalContainer(
                    Window(
                        FormattedTextControl(self._task_lines),
                        height=Dimension(min=0),
                        dont_extend_height=True,
                    ),
                    filter=Condition(lambda: len(self.state.tasks_store.list()) > 0),
                ),
                self._divider(),
                # Input window matches content height; soft-wraps multi-line content
                Window(
                    BufferControl(buffer=self._buffer),
                    get_line_prefix=self._prompt_prefix,
                    height=Dimension(min=1),
                    wrap_lines=True,
                    dont_extend_height=True,
                ),
                self._divider(),
                Window(FormattedTextControl(self._mode_line), height=1),
            ]
        )
        # Wrap body in FloatContainer to overlay the cursor-following autocompletion menu
        layout = Layout(
            FloatContainer(
                content=body,
                floats=[
                    Float(
                        xcursor=True,
                        ycursor=True,
                        content=CompletionsMenu(max_height=8, scroll_offset=1),
                    ),
                ],
            )
        )
        return Application(layout=layout, key_bindings=self._build_key_bindings())

    def _build_key_bindings(self):
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            self._on_enter()

        @kb.add("c-c")
        def _(event):
            # Cancel ongoing agent loop if active; exit program when idle
            if self._task is not None:
                self._task.cancel()
            else:
                event.app.exit()

        @kb.add("escape")
        def _(event):
            # Cancel ongoing request if active; clear buffer when idle
            if self._task is not None:
                self._task.cancel()
            else:
                self._buffer.reset()

        @kb.add("c-d")
        def _(event):
            # Exit when idle and input is empty
            if self._task is None and not self._buffer.text:
                event.app.exit()

        @kb.add("s-tab")
        def _(event):
            # Cycle through permission modes
            permissions.cycle_mode()

        @kb.add("c-b")
        def _(event):
            # Move a running foreground command to the background; the tool call returns
            # immediately with a job id
            self.state.job_registry.move_to_background()

        # Windows pastes images with Alt+V so Ctrl+V keeps pasting text
        image_paste_keys = ("escape", "v") if platform.system() == "Windows" else ("c-v",)

        @kb.add(*image_paste_keys)
        def _(event):
            # Clipboard tools are blocking subprocesses; run the paste off the event loop
            # so the REPL does not freeze for seconds
            self.app.create_background_task(self._paste_image())

        @kb.add("backspace")
        def _(event):
            # When the cursor sits right after a placeholder, delete it whole and drop the
            # image it refers to
            match = _IMAGE_PLACEHOLDER_AT_END.search(self._buffer.document.text_before_cursor)
            if match is None:
                self._buffer.delete_before_cursor(1)
                return
            self._buffer.delete_before_cursor(len(match.group()))
            index = int(match.group(1)) - 1
            if 0 <= index < len(self._pending_images):
                self._pending_images.pop(index)
            self._renumber_placeholders()

        return kb

    async def _paste_image(self):
        """
        Read the clipboard image off the event loop and add it as a pending attachment.
        """
        if self._pasting:
            return
        self._pasting = True
        try:
            image = await images.read_clipboard_image_async(self.state.session_id)
        except images.ImageError as e:
            console.print(f"[yellow]{escape(str(e))}[/]\n")
            return
        except Exception as e:
            console.print(f"[yellow]Clipboard image failed: {type(e).__name__}: {e}[/]\n")
            return
        finally:
            self._pasting = False
        if image is None:
            return
        self._pending_images.append(image)
        self._buffer.insert_text(f"[Image #{len(self._pending_images)}]")
        self.app.invalidate()

    def _renumber_placeholders(self):
        # After deleting an image in the middle, renumber the remaining placeholders so the
        # numbers keep matching the indices in _pending_images
        seq = iter(range(1, len(self._pending_images) + 1))
        cursor = self._buffer.cursor_position
        self._buffer.text = _IMAGE_PLACEHOLDER.sub(
            lambda _: f"[Image #{next(seq)}]", self._buffer.text
        )
        self._buffer.cursor_position = min(cursor, len(self._buffer.text))

    def _clear_input(self):
        # Clearing the input line also drops pending images; otherwise the placeholder is
        # gone but the image would still be sent with the next message
        self._buffer.reset()
        self._pending_images.clear()

    def _on_enter(self):
        # With the completion menu open and an item highlighted, Enter accepts the
        # completion instead of submitting
        if self._buffer.complete_state and self._buffer.complete_state.current_completion:
            self._buffer.apply_completion(self._buffer.complete_state.current_completion)
            return
        # No new submission while a request is in flight (the input line stays visible,
        # Enter just does not start another turn)
        if self._task is not None:
            return
        text = self._buffer.text.strip()
        if not text:
            self._clear_input()
            return
        # Record it in the input history and clear the line
        self._buffer.history.append_string(text)
        self._buffer.reset()
        # Echo the line to the scrollback so it is not lost: the input line is persistent
        self._echo_input(text)
        # Hand the pending images to this turn and reset for the next one
        attachments = self._pending_images
        self._pending_images = []
        # Dispatch to a background task so Enter returns immediately and the input line
        # keeps rendering and stays interruptible
        self._task = self.app.create_background_task(self._process(text, attachments=attachments))

    def _echo_input(self, text):
        # Echo submitted input framed by horizontal rules to mirror the prompt styling
        rule = "─" * console.width
        console.print(f"[bright_black]{rule}[/]")
        console.print(f"[cyan]❯[/] {escape(text)}")
        console.print(f"[bright_black]{rule}[/]")
        console.print()

    async def _process(self, text, handler=None, attachments=None):
        # Background task: delegate to the matching handler (on_submit by default), catching
        # interruption and runtime errors in one place
        handler = handler or self._on_submit
        try:
            await handler(text, attachments or [])
        except asyncio.CancelledError:
            console.print("\n[bold yellow]Interrupted[/]\n")
        except Exception as e:
            console.print(f"\n[bold red]✗ {type(e).__name__}: {e}[/]\n")
        finally:
            self._task = None
            self.working = False
            # After /rewind rolls the conversation back, put the original input back into
            # the buffer so the user can edit and resend it
            if self.state.pending_input:
                self.prefill(self.state.pending_input)
                self.state.pending_input = ""
            self.app.invalidate()
            
    def start_working(self):
        # Set working state to display spinner; cleared in _process.finally on completion or cancellation
        self.working = True
        self._work_start = time.monotonic()
        self.app.invalidate()

    def prefill(self, text: str) -> None:
        # Put text back into the input buffer with the cursor at the end, ready to edit and
        # resend; used after /rewind rolls the conversation back. Image placeholders are
        # stripped because the attachments themselves cannot be restored.
        text = re.sub(r" {2,}", " ", _IMAGE_PLACEHOLDER.sub("", text)).strip()
        self._buffer.text = text
        self._buffer.cursor_position = len(text)
        self.app.invalidate()

    def exit(self):
        # Shut down persistent input area (used by /exit)
        self.app.exit()

    @property
    def is_idle(self):
        return self._task is None

    def submit_system(self, text) -> None:
        """
        Submit a system-triggered turn (e.g. a background job completion notice); the
        caller must check is_idle first.
        """
        print_system_text(text)
        self._task = self.app.create_background_task(self._process(text, self._on_system))

    async def run(self, on_submit, on_system):
        # Run REPL event loop until exited; on_submit is the async handler for input lines
        self._on_submit = on_submit
        self._on_system = on_system
        # Heartbeat loop for spinner re-rendering during active requests
        ticker = asyncio.ensure_future(self._tick())
        try:
            # patch_stdout routes Rich logs above the persistent input prompt without overwriting it
            with patch_stdout(raw=True):
                await self.app.run_async()
        finally:
            ticker.cancel()

    async def _tick(self):
        try:
            while True:
                await asyncio.sleep(0.1)
                if self.working:
                    self._frame += 1
                    self.app.invalidate()
        except asyncio.CancelledError:
            pass
