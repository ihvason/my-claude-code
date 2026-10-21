"""
Context compaction: one LLM call summarizes the whole conversation into a
structured summary, then the summary plus the most recently read files rebuild a
much smaller history.
The manual /compact command and automatic compaction share this flow; only the
trigger differs.
"""

import os
import re

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from rich.markdown import Markdown
from rich.padding import Padding

from core.console import console
from media.mentions import build_mention_messages
from state import sessions as session

from .file_state import ReadFileState
from .model import model

# The model's context window size
CONTEXT_WINDOW = 131_072
# Output space reserved for the summary during a compaction call
COMPACT_OUTPUT_RESERVE = 20_000
# Safety margin kept below the watermark
AUTO_COMPACT_BUFFER = 10_000
# Stop attempting automatic compaction for this session after this many consecutive failures
MAX_COMPACT_FAILURES = 3
# Files restored after compaction: maximum count and total character budget
RESTORE_MAX_FILES = 5
RESTORE_MAX_CHARS = 30_000


COMPACT_PROMPT = (
    "IMPORTANT: output plain text only and do not call any tools.\n\n"
    "- Do not use read_file, edit_file, write_file, run_command or any other tool.\n"
    "- The conversation above already contains everything you need.\n"
    "- Tool calls will be rejected and will waste your only turn - you will be unable to complete the task.\n"
    "- Your entire reply must be plain text: an <analysis> block first, then a <summary> block.\n\n"
    "Your task is to write a detailed summary of the conversation so far, focused on the user's explicit requests and what you have already done. Keep enough technical detail that later work can continue seamlessly from it.\n"
    "This compaction instruction is not part of the conversation being summarized - do not mention or quote it in any section, do not count it as a user message, and do not describe writing the summary as the current work.\n\n"
    "Before writing the final summary, draft your thinking inside the <analysis> tag: walk the conversation in order and check the user's requests, your approach, key decisions, file names and code snippets, and the errors and fixes.\n\n"
    "Then output the final summary inside the <summary> tag, with these sections:\n\n"
    "1. Primary requests and intent: record every explicit user request and intent in detail\n"
    "2. Key technical concepts: list the important technical concepts, frameworks and approaches involved\n"
    "3. Files and code: list the specific files inspected, modified or created, with key code snippets\n"
    "4. Errors and fixes: list the errors encountered and how they were fixed, especially feedback asking you to take a different approach\n"
    "5. All user messages: list every user message verbatim (tool results and this compaction instruction do not count); do not paraphrase - they are key to understanding shifts in feedback and intent\n"
    "6. Current work and next step: describe precisely what was being done right before writing the summary and the immediate next step, quoting the most recent messages verbatim so work resumes without drifting\n\n"
    "Reminder: do not call any tools. Output plain text only - an <analysis> block first, then a <summary> block."
)


SUMMARY_WRAPPER = (
    "This session continues from a conversation that was compacted. The summary below covers the earlier conversation:\n\n"
    "{summary}\n\n"
    "If the summary lacks detail you need (specific code snippets, original error text, etc.), use read_file to read the full transcript from before compaction: {transcript}\n"
    "Continue working from this summary. Do not repeat the summary back to the user; pick up where the conversation left off."
)


# Compaction-only agent: no tools are registered, so the model cannot call any
summarizer = Agent(model)


def compact_threshold() -> int:
    """
    Threshold that triggers automatic compaction: step back from the window limit by
    the reserved summary output, then by a safety margin.
    """
    return CONTEXT_WINDOW - COMPACT_OUTPUT_RESERVE - AUTO_COMPACT_BUFFER


def context_tokens(history) -> int:
    """
    Estimate current context usage from the most recent model response's usage: the
    input covers everything sent that turn, plus what the model generated.
    """
    for msg in reversed(history):
        if msg.kind == "response" and msg.usage.input_tokens:
            return msg.usage.input_tokens + msg.usage.output_tokens
    return 0


def extract_summary(text: str) -> str:
    """
    Strip the <analysis> draft as a whole first, then take the contents of the
    <summary> block.
    The order matters: the draft may literally mention the <summary> tag, which would
    skew the match.
    """
    text = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.DOTALL)
    match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    text = match.group(1) if match else text
    # Also remove any stray tag text left in the body
    return re.sub(r"</?(analysis|summary)>", "", text).strip()


def build_summary_message(summary: str, transcript_path) -> ModelRequest:
    """
    Wrap the summary as a single user message, the only starting point of the new
    history.
    """
    content = SUMMARY_WRAPPER.format(summary=summary, transcript=transcript_path)
    return ModelRequest(parts=[UserPromptPart(content=content)])


def restore_file_messages(old_state: ReadFileState, new_state: ReadFileState) -> list:
    """
    Re-inject the most recently read files: the summary cannot reproduce file
    contents, so the program preserves them itself, within a budget.
    """
    picked, used = [], 0
    for path in reversed(old_state.paths()):
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if len(picked) >= RESTORE_MAX_FILES or used + size > RESTORE_MAX_CHARS:
            break
        picked.append(path)
        used += size
    # Restore in the original read order
    picked.reverse()
    # Same mechanism as @ references: inject as fake read_file calls into the history,
    # registering the files in the new readFileState along the way
    return build_mention_messages(picked, new_state)


async def run_compact(state, custom_instructions: str = "") -> None:
    """
    Full compaction flow: one LLM call summarizes the entire history, then the summary
    plus the restored files replace it wholesale with a new history.
    """
    if not state.history:
        console.print("(No conversation to compact yet)\n")
        return

    prompt = COMPACT_PROMPT
    extra = custom_instructions.strip()
    if extra:
        prompt += "\n\nAdditional requirements:\n" + extra

    console.print("Compacting context, this may take a moment...\n")
    result = await summarizer.run(prompt, message_history=state.history)
    summary = extract_summary(result.output)

    # The compaction call's own cost also counts toward the session totals
    usage = result.usage
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens

    # Archive the full transcript before rewriting the session file, so the model can
    # read back details the summary cannot reproduce
    transcript_path = session.archive_session(state.session_id)

    # Swap in a fresh readFileState: old records are invalidated along with the old
    # history, and the restored files register into the new one
    old_state = state.read_file_state
    state.read_file_state = ReadFileState()
    restored = restore_file_messages(old_state, state.read_file_state)
    state.history = [build_summary_message(summary, transcript_path), *restored]
    # Rewrite the session file wholesale with the new history
    session.rewrite_messages(state.session_id, state.history)

    # Old checkpoints' history_index values point at messages that were compacted away;
    # drop them all
    fh = state.file_history
    if fh is not None and fh.checkpoints:
        fh.drop_from(fh.checkpoints[0])
    # Recalled memories disappeared with the old history, so allow them to be recalled
    # again; also clear the in-process API call log
    state.surfaced_memories = set()
    state.last_api_calls.clear()
    # Compaction succeeded, so the auto-compaction failure streak is no longer relevant
    state.compact_failures = 0

    # The compaction call's input_tokens is the pre-compaction context size
    console.print(
        f"[magenta]✻ Compaction complete: context was {usage.input_tokens:,} tokens, rebuilt as one summary + {len(restored) // 2} recently read file(s)[/]"
    )
    console.print(f"[dim]Full transcript archived at: {transcript_path}[/]")
    console.print(Padding(Markdown(summary), (0, 0, 0, 2)), style="dim")
    console.print()


async def auto_compact_if_needed(state) -> None:
    """
    Called before each request: compact automatically once the context crosses the
    watermark; failures only warn and never block, and after several consecutive
    failures it stops retrying.
    """
    if state.compact_failures >= MAX_COMPACT_FAILURES:
        return
    used = context_tokens(state.history)
    threshold = compact_threshold()
    if used < threshold:
        return
    console.print(
        f"[yellow]Context near the limit ({used:,} / {threshold:,} tokens); compacting automatically[/]\n"
    )
    try:
        await run_compact(state)
        state.compact_failures = 0
    except Exception as e:
        state.compact_failures += 1
        console.print(
            f"[yellow]Automatic compaction failed ({type(e).__name__}: {e}); continuing[/]\n"
        )
        if state.compact_failures >= MAX_COMPACT_FAILURES:
            console.print(
                "[yellow]Automatic compaction failed repeatedly; not retrying in this session[/]\n"
            )
