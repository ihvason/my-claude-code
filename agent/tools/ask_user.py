"""
ask_user_question tool: lets the model ask the user multiple-choice questions
mid-execution to gather preferences, resolve ambiguity, or let the user decide.

The UI is hand-drawn with prompt_toolkit and laid out as: a top chip navigation
bar (← chip chip ✔ Submit →), the full question, then a vertical option list with
a ❯ pointer and dim descriptions.

in_terminal() temporarily yields the persistent Repl input line to this picker
Application and restores it afterwards.
"""
from dataclasses import dataclass, field

from prompt_toolkit.application import Application, in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from pydantic_ai.exceptions import ModelRetry

from ui.render import console

# Color scheme: purple background for the focused chip, green once Submit is
# ready, blue for the pointer and the current label, grey for descriptions.
# Kept at module level so every picker reuses one style object instead of
# rebuilding it from_dict each time
_STYLE = Style.from_dict({
    "nav-arrow": "#9ca3af",
    "nav-arrow-dim": "#4b5563",
    "chip-current": "bg:#5b21b6 #ffffff bold",
    "chip-answered": "#9ca3af",
    "chip-pending": "#6b7280",
    "chip-submit-ready": "#10b981 bold",
    "question": "bold",
    "row-current": "#3b82f6 bold",
    "row": "",
    "label-current": "#3b82f6 bold",
    "label": "",
    "label-dim": "#9ca3af",
    "desc": "#6b7280",
    "footer": "#6b7280",
    "text-input-label": "#9ca3af",
})

# Fixed hint line for text-input mode; also module-level so the FormattedText is
# not rebuilt on every frame
_TEXT_INPUT_LABEL = FormattedText([("class:text-input-label", "  > Custom text: ")])


@dataclass
class QuestionOption:
    """
    A single option within a question.
    """
    # Display text for the option (1-5 words is ideal)
    label: str
    # What the option means / what happens if it is chosen
    description: str = ""


@dataclass
class Question:
    """
    A single multiple-choice question.
    """
    # The full question text; should end with a question mark
    question: str
    # Label of at most 12 characters, rendered in the top chip bar
    header: str
    # 2-4 options; an "Other" entry is appended automatically for free-form input
    options: list[QuestionOption]
    # True = multi-select (Space toggles, Enter confirms this question);
    # False = single-select (Enter picks and advances to the next question)
    multi_select: bool = False


@dataclass
class _QuestionState:
    """
    Runtime state for one question, held by the picker Application.
    Single-select enforces |picks| <= 1; multi-select allows any subset.
    custom_text is the text entered through "Other": additive with picks in
    multi-select, mutually exclusive with them in single-select.
    """
    # Index of the highlighted row in the option list (0..len(options)); the last
    # row is "Other"
    cursor: int = 0
    # Set of picked option labels; at most 1 in single-select, any number in multi-select
    picks: set[str] = field(default_factory=set)
    # Free-form text the user entered through the "Other" entry
    custom_text: str = ""


class _Picker:
    """
    Hand-drawn picker that renders the multiple-choice layout with prompt_toolkit.

    Focus model: a single q_idx field suffices - 0..len(questions)-1 means "on
    that question", and q_idx == len(questions) means "focus is on Submit". No
    separate on_submit flag is maintained.
    in_text_input is an orthogonal sub-mode, true only while editing an "Other" answer.
    """

    def __init__(self, questions: list[Question]):
        self.questions = questions
        # 0..len-1 -> on a question; len -> on Submit
        self.q_idx = 0
        self.states = [_QuestionState() for _ in questions]
        self.in_text_input = False
        self.text_buffer = Buffer(multiline=False)
        self.cancelled = False
        self.app: Application = self._build_app()

    # ------------------------------------------------------------------
    # Internal state queries
    # ------------------------------------------------------------------

    def _on_submit(self) -> bool:
        return self.q_idx == len(self.questions)

    def _current_q(self) -> Question:
        return self.questions[self.q_idx]

    def _current_state(self) -> _QuestionState:
        return self.states[self.q_idx]

    def _num_rows(self) -> int:
        # option rows plus the trailing "Other" row
        return len(self._current_q().options) + 1

    def _is_answered(self, idx: int) -> bool:
        s = self.states[idx]
        return bool(s.picks) or bool(s.custom_text)

    def _all_answered(self) -> bool:
        return all(self._is_answered(i) for i in range(len(self.questions)))

    def _answer_for(self, idx: int) -> str:
        """
        Format one question's final answer as a string: the picked option labels
        (in option order) plus any "Other" custom text.
        """
        q = self.questions[idx]
        s = self.states[idx]
        items = [opt.label for opt in q.options if opt.label in s.picks]
        if s.custom_text:
            items.append(s.custom_text)
        return ", ".join(items) if items else "[empty]"

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_nav(self):
        """
        Top chip navigation bar: ← [chip1] chip2 chip3 ✔ Submit →
        """
        # Compute the answered flags once instead of recomputing them in each of
        # the chip, can_next and submit-ready paths below
        answered_flags = [self._is_answered(i) for i in range(len(self.questions))]
        on_submit = self._on_submit()
        all_answered = all(answered_flags)
        parts: list[tuple[str, str]] = []

        # Left arrow: enabled when not on question 0 (or when on Submit), dim otherwise
        can_prev = on_submit or self.q_idx > 0
        parts.append(("class:nav-arrow" if can_prev else "class:nav-arrow-dim", "← "))

        for i, q in enumerate(self.questions):
            answered = answered_flags[i]
            symbol = "✔" if answered else "□"
            # Focus is on this question (on Submit, q_idx == len, so this never matches)
            if i == self.q_idx:
                cls = "class:chip-current"
            elif answered:
                cls = "class:chip-answered"
            else:
                cls = "class:chip-pending"
            parts.append((cls, f" {symbol} {q.header} "))
            parts.append(("", " "))

        # Submit chip: green once everything is answered (ready to submit),
        # highlighted purple when focused
        if on_submit:
            submit_cls = "class:chip-current"
        elif all_answered:
            submit_cls = "class:chip-submit-ready"
        else:
            submit_cls = "class:chip-pending"
        parts.append((submit_cls, " ✔ Submit "))

        # Right arrow: there is a next question, or we are on the last one and
        # everything is answered so Submit is reachable
        can_next = not on_submit and (self.q_idx < len(self.questions) - 1 or all_answered)
        parts.append(("class:nav-arrow" if can_next else "class:nav-arrow-dim", "  →"))
        return FormattedText(parts)

    def _render_question_text(self):
        if self._on_submit():
            return FormattedText([("class:question", "All questions answered - press Enter to submit")])
        return FormattedText([("class:question", self._current_q().question)])

    def _render_options(self):
        """
        Vertical option list. The current row carries a ❯ pointer; the description
        renders dim on the following line.
        Single-select: ● marks the picked option (picks holds exactly one element).
        Multi-select: each option shows a ☐/☑ checkbox.
        """
        if self._on_submit():
            return FormattedText([])

        q = self._current_q()
        s = self._current_state()
        lines: list[tuple[str, str]] = []

        for i, opt in enumerate(q.options):
            is_cursor = (i == s.cursor)
            is_picked = opt.label in s.picks
            pointer = "❯" if is_cursor else " "
            if q.multi_select:
                mark = "☑ " if is_picked else "☐ "
            else:
                mark = "● " if is_picked else "  "

            cls_row = "class:row-current" if is_cursor else "class:row"
            cls_label = "class:label-current" if is_cursor else "class:label"
            lines.append((cls_row, f" {pointer}  "))
            lines.append((cls_label, f"{i + 1}. {mark}{opt.label}"))
            lines.append(("", "\n"))

            if opt.description:
                lines.append(("class:desc", f"      {opt.description}"))
                lines.append(("", "\n"))

        # Trailing "Other" row: echo back any text already entered
        other_idx = len(q.options)
        is_cursor = (s.cursor == other_idx)
        pointer = "❯" if is_cursor else " "
        cls_row = "class:row-current" if is_cursor else "class:row"
        cls_label = "class:label-current" if is_cursor else "class:label-dim"
        suffix = f": {s.custom_text}" if s.custom_text else ""
        other_text = f"{other_idx + 1}. Other (enter custom text){suffix}"
        lines.append((cls_row, f" {pointer}  "))
        lines.append((cls_label, other_text))
        lines.append(("", "\n"))

        return FormattedText(lines)

    def _render_footer(self):
        """
        Bottom help line; its contents change with the current mode.
        """
        if self.in_text_input:
            return FormattedText([("class:footer", "  Enter to confirm · Esc to discard text input")])
        if self._on_submit():
            return FormattedText([("class:footer", "  Enter to submit · ← back to previous question · Esc to cancel")])
        q = self._current_q()
        hints = ["↑↓ options"]
        if q.multi_select:
            hints += ["Space to toggle", "Enter to confirm this question"]
        else:
            hints.append("Enter to pick and advance")
        hints += ["←→ switch questions", "Esc to cancel"]
        return FormattedText([("class:footer", "  " + " · ".join(hints))])

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _advance(self):
        """
        Move focus forward one step: to the next question, or to Submit when
        already on the last question.
        """
        if not self._on_submit():
            self.q_idx += 1

    def _move(self, delta: int):
        """
        Move the cursor ±1 within the option list, wrapping at the ends. A no-op
        on Submit and in text-input mode.
        """
        if self.in_text_input or self._on_submit():
            return
        s = self._current_state()
        s.cursor = (s.cursor + delta) % self._num_rows()

    def _prev_question(self):
        # q_idx == len -> step back to the last question; otherwise a plain q_idx -= 1
        if self.in_text_input or self.q_idx == 0:
            return
        self.q_idx -= 1

    def _next_question(self):
        if self.in_text_input or self._on_submit():
            return
        # Advance directly unless this is the last question; on the last question,
        # moving to Submit is allowed only once everything is answered
        if self.q_idx < len(self.questions) - 1 or self._all_answered():
            self.q_idx += 1

    def _enter_text_input(self):
        """
        Switch to text-input mode, pre-filling the buffer with any custom text
        already entered for this question.
        """
        self.in_text_input = True
        self.text_buffer.text = self._current_state().custom_text
        # Put the cursor at the end so typing continues naturally
        self.text_buffer.cursor_position = len(self.text_buffer.text)

    def _commit_text(self):
        """
        Write the entered custom text back into the question state.
        Single-select: the custom text clears any previously picked list option
        and advances to the next question.
        Multi-select: the custom text adds to the picked options and focus stays
        put so the user can keep ticking others.
        """
        if not self.in_text_input:
            return
        text = self.text_buffer.text.strip()
        s = self._current_state()
        s.custom_text = text
        self.in_text_input = False
        if text and not self._current_q().multi_select:
            # In single-select, custom text is the sole answer
            s.picks.clear()
            self._advance()

    def _toggle_pick(self, label: str):
        """
        Toggle a label in the current question's picks set. Shared by Enter
        (multi-select) and Space.
        """
        s = self._current_state()
        if label in s.picks:
            s.picks.remove(label)
        else:
            s.picks.add(label)

    def _on_enter(self):
        """
        Enter in selection mode: behaviour varies with the cursor position and
        with single- vs multi-select.
        """
        if self._on_submit():
            # Focus is on Submit: finish the dialog
            self.app.exit(result=True)
            return

        q = self._current_q()
        s = self._current_state()

        # Cursor is on the "Other" row: switch to text-input mode
        if s.cursor == len(q.options):
            self._enter_text_input()
            return

        if q.multi_select:
            # In multi-select, Enter means "confirm this question" - toggling is
            # Space's job, which avoids the counter-intuitive flow of ticking an
            # option with Space and then un-ticking it with Enter
            if self._is_answered(self.q_idx):
                self._advance()
            return

        # Single-select Enter: overwrite the previous pick (including custom_text)
        # with the current option, then advance
        opt = q.options[s.cursor]
        s.picks = {opt.label}
        s.custom_text = ""
        self._advance()

    def _on_space(self):
        """
        Space key: toggles the current checkbox in multi-select, ignored in
        single-select.
        """
        if self._on_submit() or self.in_text_input:
            return
        q = self._current_q()
        if not q.multi_select:
            return
        s = self._current_state()
        if s.cursor == len(q.options):
            self._enter_text_input()
            return
        self._toggle_pick(q.options[s.cursor].label)

    def _on_escape(self):
        """
        Esc: in text-input mode, discard the input and return to selection;
        otherwise cancel the whole dialog.
        """
        if self.in_text_input:
            self.in_text_input = False
            self.text_buffer.text = ""
            return
        self.cancelled = True
        self.app.exit(result=False)

    # ------------------------------------------------------------------
    # Application construction
    # ------------------------------------------------------------------

    def _build_app(self) -> Application:
        kb = KeyBindings()
        sel = Condition(lambda: not self.in_text_input)
        text = Condition(lambda: self.in_text_input)

        # Stacked decorators bind one callback to several keys (multiple arguments
        # to kb.add form a key *sequence*, not alternatives)
        @kb.add("up", filter=sel)
        @kb.add("k", filter=sel)
        def _(event):
            self._move(-1)

        @kb.add("down", filter=sel)
        @kb.add("j", filter=sel)
        def _(event):
            self._move(1)

        @kb.add("enter", filter=sel)
        def _(event):
            self._on_enter()

        @kb.add("space", filter=sel)
        def _(event):
            self._on_space()

        @kb.add("left", filter=sel)
        def _(event):
            self._prev_question()

        @kb.add("right", filter=sel)
        def _(event):
            self._next_question()

        @kb.add("escape", filter=sel)
        @kb.add("c-c", filter=sel)
        def _(event):
            self._on_escape()

        # Key bindings for text-input mode
        @kb.add("enter", filter=text)
        def _(event):
            self._commit_text()

        @kb.add("escape", filter=text)
        def _(event):
            self._on_escape()

        # One Window per region
        nav_win = Window(FormattedTextControl(self._render_nav), height=1)
        question_win = Window(FormattedTextControl(self._render_question_text), wrap_lines=True, dont_extend_height=True)
        options_win = Window(FormattedTextControl(self._render_options), dont_extend_height=True)
        text_win = ConditionalContainer(
            HSplit([
                # Hint line for text-input mode; module-level constant, so every
                # frame returns the same object
                Window(FormattedTextControl(lambda: _TEXT_INPUT_LABEL), height=1),
                Window(BufferControl(buffer=self.text_buffer), height=1),
            ]),
            filter=text,
        )
        footer_win = Window(FormattedTextControl(self._render_footer), height=1)

        layout = Layout(HSplit([
            nav_win,
            Window(height=1, char=" "),
            question_win,
            Window(height=1, char=" "),
            options_win,
            text_win,
            Window(height=1, char=" "),
            footer_win,
        ]))

        return Application(
            layout=layout,
            key_bindings=kb,
            style=_STYLE,
            full_screen=False,
            mouse_support=False,
        )

    async def run(self) -> dict[str, str] | None:
        """
        Run the picker and return the answer dict, or None if it was cancelled.
        """
        submitted = await self.app.run_async()
        if self.cancelled or not submitted:
            return None
        return {
            q.question: self._answer_for(i)
            for i, q in enumerate(self.questions)
        }


async def ask_user_question(questions: list[Question]) -> str:
    """
    Ask the user multiple-choice questions mid-execution to gather preferences,
    resolve ambiguity, or let the user make a decision.

    Use it to:
    1. Collect the user's preferences or requirements
    2. Clarify ambiguous instructions
    3. Get the user's decision on an implementation approach
    4. Let the user choose the next direction

    Usage notes:
    - The user can always pick "Other" to enter custom text, so do not add an
      Other option yourself
    - multi_select=True allows multiple picks, for options that are not mutually
      exclusive
    - If one option is recommended, put it first and append "(Recommended)" to its label

    Args:
        questions: 1-4 questions, each with 2-4 options
    """
    # Argument validation: hard limits of 1-4 questions with 2-4 options each
    if not (1 <= len(questions) <= 4):
        raise ModelRetry("Ask between 1 and 4 questions at a time")
    for q in questions:
        if not (2 <= len(q.options) <= 4):
            raise ModelRetry(f'Question "{q.question}" must have 2-4 options')

    # in_terminal yields the terminal to the picker Application; the persistent
    # Repl input line is restored automatically afterwards
    async with in_terminal():
        # Leave one blank line to visually separate the picker from the tool_call
        # line above it
        console.print()
        picker = _Picker(questions)
        answers = await picker.run()
        console.print()

    if answers is None:
        return "The user cancelled the question and gave no answer. Wait for further instructions; do not act on your own."

    # Join into natural language for the model, matching CC's
    # mapToolResultToToolResultBlockParam format
    formatted = "; ".join(f'"{q}" → "{a}"' for q, a in answers.items())
    return f"The user answered as follows: {formatted}. Continue based on these answers."
