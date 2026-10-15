"""
Interactive REPL interface built on prompt_toolkit.

Maintains a persistent bottom input bar while streaming agent output above via
patch_stdout. Supports real-time execution cancellation (Esc / Ctrl+C), command
history, dynamic spinners, and cycling permission modes via Shift+Tab.
"""
import asyncio
import time

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.patch_stdout import patch_stdout
from rich.markup import escape

import permissions
from .render import console

# Persistent input bar: Anchored to the bottom throughout the session.
# Agent output prints above via patch_stdout; pressing ESC / Ctrl+C interrupts running requests immediately.

# Spinner animation frames for the working indicator
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Repl:
    """
    Persistent REPL bar; main loop injects an on_submit callback to handle input.
    """

    def __init__(self, state):
        self.state = state
        # Injected by main to process each line (slash command or dispatched to Agent)
        self._on_submit = None
        # Background task processing current input; target for ESC / Ctrl+C cancellation
        self._task = None
        # Model execution state; toggles visibility of the working indicator
        self.working = False
        self._work_start = 0.0
        self._frame = 0
        self._buffer = Buffer(multiline=False, history=InMemoryHistory())
        self.app = self._build_app()

    def _prompt_prefix(self, line_number, wrap_count):
        return HTML("<ansicyan>❯ </ansicyan>")

    def _working_line(self):
        # Header line: spinner + elapsed time + cancel hint (shown only during active requests)
        frame = _SPINNER[self._frame % len(_SPINNER)]
        elapsed = int(time.monotonic() - self._work_start)
        return HTML(
            f"<ansigreen>{frame}</ansigreen> <b>Working…</b>"
            f"<ansibrightblack> ({elapsed}s elapsed · press esc to interrupt)</ansibrightblack>"
        )

    def _mode_line(self):
        # Footer line: active permission mode + cycle hint
        mode = permissions.state.mode
        return HTML(
            f"  <ansimagenta><b>▶▶ {mode}</b></ansimagenta>"
            f"<ansibrightblack> (Shift+Tab to cycle)</ansibrightblack>"
        )

    def _divider(self):
        # Horizontal divider rule
        return Window(height=1, char="─", style="fg:ansibrightblack")

    def _build_app(self):
        # Vertical layout: working indicator, divider, input bar, divider, mode status
        layout = Layout(
            HSplit(
                [
                    ConditionalContainer(
                        Window(FormattedTextControl(self._working_line), height=1),
                        filter=Condition(lambda: self.working),
                    ),
                    self._divider(),
                    # Input line auto-wraps and dynamically sizes to content
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
        )
        return Application(layout=layout, key_bindings=self._build_key_bindings())

    def _build_key_bindings(self):
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            self._on_enter()

        @kb.add("c-c")
        def _(event):
            # Interrupt running agent loop if active; exit program if idle
            if self._task is not None:
                self._task.cancel()
            else:
                event.app.exit()

        @kb.add("escape")
        def _(event):
            # Interrupt running agent loop if active; clear buffer if idle
            if self._task is not None:
                self._task.cancel()
            else:
                self._buffer.reset()

        @kb.add("c-d")
        def _(event):
            # Exit if idle and buffer is empty
            if self._task is None and not self._buffer.text:
                event.app.exit()

        @kb.add("s-tab")
        def _(event):
            # Cycle permission mode
            permissions.cycle_mode()

        return kb

    def _on_enter(self):
        # Reject new submissions while a request is in progress
        if self._task is not None:
            return
        text = self._buffer.text.strip()
        if not text:
            self._buffer.reset()
            return
        # Append to history and clear input buffer
        self._buffer.history.append_string(text)
        self._buffer.reset()
        # Echo submitted line to the scrollable output history above
        self._echo_input(text)
        # Dispatch execution to background task to keep UI responsive and interruptible
        self._task = self.app.create_background_task(self._process(text))

    def _echo_input(self, text):
        # Render submitted prompt framed with rules matching the input bar style
        rule = "─" * console.width
        console.print(f"[bright_black]{rule}[/]")
        console.print(f"[cyan]❯[/] {escape(text)}")
        console.print(f"[bright_black]{rule}[/]")
        console.print()

    async def _process(self, text):
        # Background task handling on_submit, exceptions, and cancellations
        try:
            await self._on_submit(text)
        except asyncio.CancelledError:
            console.print("\n[bold yellow]Interrupted[/]\n")
        except Exception as e:
            console.print(f"\n[bold red]✗ {type(e).__name__}: {e}[/]\n")
        finally:
            self._task = None
            self.working = False
            self.app.invalidate()

    def start_working(self):
        # Enter working state, triggering indicator display
        self.working = True
        self._work_start = time.monotonic()
        self.app.invalidate()

    def exit(self):
        # Terminate REPL loop (used by /exit)
        self.app.exit()

    async def run(self, on_submit):
        # Run REPL until exit; on_submit is an async handler for input lines
        self._on_submit = on_submit
        # Spinner tick loop to invalidate and redraw UI during execution
        ticker = asyncio.ensure_future(self._tick())
        try:
            # patch_stdout ensures rich output prints above the persistent input bar
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
