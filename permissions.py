"""
Permission control: Evaluates whether to allow, prompt, or deny a tool call before execution.
"""
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, in_terminal
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich.markdown import Markdown
from rich.panel import Panel

from memory import store
from ui.render import console


# Permission modes
# default: Requires approval for file writes and shell commands; auto-allows read-only operations
DEFAULT = "default"
# acceptEdits: Auto-allows file read/write operations; prompts for other tools
ACCEPT_EDITS = "acceptEdits"
# bypass: Allows all tool calls without prompting
BYPASS = "bypass"
# auto: Delegates pending calls to an LLM classifier; allows safe actions and blocks risky ones
AUTO = "auto"
# plan: explore and plan read-only; only the plan file may be written, and every other
# non-read-only tool asks for approval each time
PLAN = "plan"

# Order in which Shift+Tab cycles the permission modes; plan comes right after acceptEdits
MODES = [DEFAULT, ACCEPT_EDITS, PLAN, AUTO, BYPASS]

READONLY_TOOLS = {"read_file", "ask_user_question", "task_create", "task_list", "task_get", "task_update", "job_kill"}

# File modification tools: Auto-allowed in acceptEdits mode
EDIT_TOOLS = {"write_file", "edit_file"}

# Allowed in plan mode besides read-only tools: the plan workflow itself relies on them
# (dispatching read-only sub agents, loading a skill) and exit_plan_mode opens its own
# approval picker
PLAN_ALLOWED_TOOLS = READONLY_TOOLS | {"run_agent", "load_skill", "enter_plan_mode", "exit_plan_mode"}

# Tool self-check registry: Evaluates arguments for fine-grained semantic risk control
TOOL_SELF_CHECKS = {}


def register_self_check(tool_name: str, check) -> None:
    """
    Registers a custom self-check for a tool.
    Receives args and returns "allow", "ask", or None to delegate to general rules.
    """
    TOOL_SELF_CHECKS[tool_name] = check


@dataclass
class PermissionState:
    """
    Process-level permission state and session-level allowlist.
    - Mode is process-wide: persists across /new and /resume; resets on restart.
    - Allowlist is session-scoped: clears when switching sessions via /new or /resume.
    """
    mode: str = DEFAULT
    # Mode to restore once the plan is approved
    pre_plan_mode: str = DEFAULT
    # Tools marked "always allow" for the current session
    session_allowed: set = field(default_factory=set)


state = PermissionState()


def compute_decision(tool_name: str, args: dict, plan_file: Path | None = None) -> str:
    """
    Evaluates permission rules and returns "allow" or "ask" for the given tool call.
    """
    # Bypass mode: Allow all without evaluation
    if state.mode == BYPASS:
        return "allow"
    # Tool self-check: Takes precedence over session allowlist
    check = TOOL_SELF_CHECKS.get(tool_name)
    verdict = check(args) if check else None

    # Plan mode is decided on its own and bypasses the session allowlist: "don't ask again"
    # must not permit changes during planning
    if state.mode == PLAN:
        # Only the plan file may be written; everything else is denied
        if tool_name in EDIT_TOOLS:
            if plan_file and _is_same_file(args.get("path", ""), plan_file):
                return "allow"
            return "deny"
        # Read-only tools, the plan workflow's own tools, and calls the self-check proved safe
        # (ls, grep and the like) are allowed; everything else asks every time
        return "allow" if tool_name in PLAN_ALLOWED_TOOLS or verdict == "allow" else "ask"
    
    if verdict:
        return verdict
    # Session allowlist check
    if tool_name in state.session_allowed:
        return "allow"
    # Read-only tools are always allowed
    if tool_name in READONLY_TOOLS:
        return "allow"
    # Auto-allow writes into the memory directory: that is the agent's own data
    # directory, it never touches the user's project or system state, and prompting
    # would only interrupt the automatic accumulation of memory
    if tool_name in EDIT_TOOLS and store.is_memory_path(str(args.get("path", ""))):
        return "allow"
    # acceptEdits and auto modes: allow file edits; other tools proceed to subsequent checks.
    # Included in auto mode as a fast path: file edits can be determined deterministically without calling the classifier.
    if state.mode in (ACCEPT_EDITS, AUTO) and tool_name in EDIT_TOOLS:
        return "allow"
    # Default fallback: prompt user
    return "ask"


def _is_same_file(path: str, target: Path) -> bool:
    return bool(path) and Path(path).resolve() == target.resolve()


def set_mode(mode: str) -> None:
    # Remember the previous mode when entering plan mode, so approval can restore it
    if mode == PLAN and state.mode != PLAN:
        state.pre_plan_mode = state.mode
    state.mode = mode


def cycle_mode() -> str:
    """
    Cycles mode through default -> acceptEdits -> plan -> auto -> bypass -> default.
    """
    index = MODES.index(state.mode)
    set_mode(MODES[(index + 1) % len(MODES)])
    return state.mode


# Truncate large payloads in previews while keeping paths and commands intact
BULKY_ARGS = {"content", "old_string", "new_string"}


def _format_call(tool_name: str, args: dict) -> str:
    """
    Formats a tool invocation for approval preview, e.g. run_command(command=npm test).
    """
    def show(key, value):
        text = " ".join(str(value).split())
        if key in BULKY_ARGS and len(text) > 60:
            return text[:60] + "..."
        return text

    inner = ", ".join(f"{key}={show(key, value)}" for key, value in args.items())
    return f"{tool_name}({inner})"


_STYLE = Style.from_dict({
    "question": "bold",
    "label-current": "#3b82f6 bold",
    "label": "",
    "footer": "#6b7280",
})


class _ApprovalPicker:
    """
    Interactive single-choice CLI picker matching the design of ask_user_question.
    """

    def __init__(self, question: str, options: list[tuple[str, str]]):
        self.question = question
        self.options = options  # List of (value, label) tuples
        self.cursor = 0
        self.result = None  # None indicates cancellation (defaults to deny)
        self.app = self._build_app()

    def _render_question(self):
        return FormattedText([("class:question", self.question)])

    def _render_options(self):
        lines: list[tuple[str, str]] = []
        for i, (_value, label) in enumerate(self.options):
            is_cursor = (i == self.cursor)
            pointer = "❯" if is_cursor else " "
            cls_label = "class:label-current" if is_cursor else "class:label"
            lines.append((cls_label, f" {pointer}  {i + 1}. {label}"))
            lines.append(("", "\n"))
        return FormattedText(lines)

    def _render_footer(self):
        return FormattedText([("class:footer", "  ↑↓ Navigate · Enter Confirm · Esc Cancel")])

    def _move(self, delta: int):
        self.cursor = (self.cursor + delta) % len(self.options)

    def _build_app(self) -> Application:
        kb = KeyBindings()

        # Stacked decorators bind multiple key aliases to the same handler
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

        # always_hide_cursor prevents the terminal cursor from overlapping the prompt text
        layout = Layout(HSplit([
            Window(FormattedTextControl(self._render_question), wrap_lines=True, dont_extend_height=True, always_hide_cursor=True),
            Window(FormattedTextControl(self._render_options), dont_extend_height=True, always_hide_cursor=True),
            Window(FormattedTextControl(self._render_footer), height=1, always_hide_cursor=True),
        ]))
        return Application(
            layout=layout,
            key_bindings=kb,
            style=_STYLE,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,  # Clear picker output from terminal on exit
        )

    async def run(self) -> str:
        await self.app.run_async()
        return self.result if self.result is not None else "deny"


async def prompt_approval(tool_name: str, args: dict, requester: str = "") -> str:
    """
    Show the approval picker before a tool runs; returns "once", "always" or "deny".
    A requester prefix identifies approval requests that come from a sub agent.
    """
    options = [
        ("once", "Allow once"),
        ("always", f"Always allow {tool_name} for this session"),
        ("deny", "Deny"),
    ]
    async with in_terminal():
        question = f"{requester}Allow {_format_call(tool_name, args)}?"
        choice = await _ApprovalPicker(question, options).run()
    return choice


async def prompt_plan_approval(plan: str, plan_file: Path) -> tuple[str, str]:
    """
    Show the plan and open the approval picker. Returns (choice, feedback), where choice is
    "approve", "keep" to continue planning, or "deny" when cancelled with Esc.
    """
    options = [
        ("approve", f"Approve and switch to {state.pre_plan_mode} to implement"),
        ("keep", "Not approved; keep refining the plan"),
    ]
    feedback = ""
    async with in_terminal():
        console.print(Panel(Markdown(plan), title=str(plan_file), border_style="magenta"))
        choice = await _ApprovalPicker("Approve this plan and start implementing?", options).run()
        if choice == "keep":
            feedback = await PromptSession().prompt_async("Tell the model what to change (press Enter to skip): ")
    return choice, feedback.strip()
