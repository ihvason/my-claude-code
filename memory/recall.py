"""
Memory recall: before a user message enters the agent loop, send a separate LLM
request to pick relevant memories from the manifest, then wrap them as
system-reminder messages and inject them into the conversation history.
"""

import json
import os

from openai import AsyncOpenAI
from pydantic_ai.messages import ModelRequest, UserPromptPart

from core.console import print_step
from state import sessions as session

from . import store

# Like the classifier, this is a separate request outside the main conversation,
# bypassing the agent framework
_client = AsyncOpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url="https://api.deepseek.com",
)

RECALL_MODEL = "deepseek-flash"

# Maximum number of memories recalled per turn
MAX_RECALL = 5
# Very short input (e.g. "continue", "ok") is usually a follow-up and not worth a
# dedicated recall request
MIN_INPUT_CHARS = 4

SYSTEM_PROMPT = """You are selecting memories for a coding agent to help it handle the user's latest input. The user message gives you that input plus a manifest of available memories (one memory file per line, with type, date and description).

Return the list of memory filenames that are clearly useful for handling this input (at most 5). Select only memories you are confident are helpful based on their filename and description.
- If you are unsure whether a memory is useful, leave it out. Be picky.
- If the manifest contains nothing clearly useful, return an empty list.

Output JSON only: {"selected_memories": ["filename", ...]}
"""


async def _select(user_input: str, headers: list) -> list[str]:
    """
    Make a single LLM request to pick relevant filenames. Fail-open by design:
    recall is only an enhancement, so any error means no memories for this turn.
    """
    manifest = store.format_manifest(headers)
    try:
        response = await _client.chat.completions.create(
            model=RECALL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Latest input: {user_input}\n\nMemory manifest:\n{manifest}",
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        selected = json.loads(response.choices[0].message.content).get("selected_memories") or []
    except Exception:
        return []
    # The model may invent filenames; filter down to ones that actually exist
    valid = {h.filename for h in headers}
    return [name for name in selected if isinstance(name, str) and name in valid][:MAX_RECALL]


def _build_reminder(header) -> str:
    """
    Wrap one memory file as system-reminder body: relative save time (a freshness
    warning once older than a day) + file path + content.
    """
    content = header.path.read_text(encoding="utf-8")
    content = store.truncate_content(
        content,
        store.MAX_MEMORY_LINES,
        store.MAX_MEMORY_BYTES,
        f"This memory was truncated; use read_file to view the full file: {header.path}",
    )
    days = store.age_days(header.mtime)
    if days <= 1:
        head = f"Memory (saved {store.age_text(days)}): {header.path}:"
    else:
        # For older memories, replace the header with a freshness warning so the
        # model verifies before trusting
        head = f"{store.staleness_text(days)}\n\nMemory: {header.path}:"
    return f"<system-reminder>\n{head}\n\n{content}\n</system-reminder>"


async def inject_memories(user_input: str, state) -> None:
    """
    Recall entry point triggered by each user message: select memories, wrap them
    as messages, inject into history and persist; memories already surfaced in this
    session are not injected again.
    """
    if len(user_input.strip()) < MIN_INPUT_CHARS:
        return
    headers = [h for h in store.scan_memory_files() if h.filename not in state.surfaced_memories]
    if not headers:
        return
    selected = await _select(user_input, headers)
    if not selected:
        return

    by_name = {h.filename: h for h in headers}
    injected = []
    for filename in selected:
        header = by_name[filename]
        try:
            text = _build_reminder(header)
        except OSError:
            continue
        injected.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        state.surfaced_memories.add(filename)
        # Echo which memory was recalled so the user can see recall actually happened
        print_step(
            "[dim]◇ memory[/]",
            f"[dim]Recalled memory: {filename} ({store.age_text(store.age_days(header.mtime))})[/]",
        )
    if not injected:
        return

    # Inject into history and persist like @ references, so /resume restores
    # recalled memories too
    state.history += injected
    session.append_messages(state.session_id, injected)
