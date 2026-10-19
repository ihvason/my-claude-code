"""
File tools: read_file / edit_file / write_file, plus the shared
read_and_register helper.

read_file registers every file it reads in the session's readFileState; edit_file
and write_file consult that record first and refuse to proceed if the file was
never read, or was modified after being read. This structurally prevents both
"editing from hallucinated content" and "clobbering someone else's changes".

readFileState is dependency-injected through RunContext, so each session gets its
own isolated instance.
"""
import os

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from ..file_state import ReadFileState

# Maximum lines read when no limit is given; anything beyond is truncated with a
# hint to continue using offset
DEFAULT_MAX_LINES = 2000


def _with_line_numbers(content: str, start_line: int = 1) -> str:
    """
    Prefix every line with its number, cat -n style: right-aligned number, a tab,
    then the original line.
    The model locates code by line number, which is what lets the old_string
    passed to edit_file line up exactly.
    start_line is the number of the first line; for ranged reads it equals offset,
    so numbers match the real file.
    """
    lines = content.splitlines()
    if not lines:
        return "(empty file)"
    # Alignment width comes from the largest line number in this range (for ranged
    # reads, numbering starts at start_line rather than necessarily at 1)
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(f"{i:>{width}}\t{line}" for i, line in enumerate(lines, start_line))


def read_and_register(state: ReadFileState, path: str, offset: int = 1, limit: int | None = None) -> str:
    """
    Read a file, slice it by offset/limit, register it in readFileState, and
    return the content with line numbers.
    Both the read_file tool and @ reference injection call this, so files pulled
    in via @ look identical in readFileState to files the model read itself.
    """
    # 1. Read the whole file
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. Slice by offset/limit; without a limit, take at most DEFAULT_MAX_LINES
    all_lines = content.splitlines()
    total = len(all_lines)
    start = offset - 1
    if start >= total and total > 0:
        return f"Warning: the file has only {total} lines but offset is {offset}; nothing to read"
    end = start + limit if limit is not None else start + DEFAULT_MAX_LINES
    selected = all_lines[start:end]
    selected_content = "\n".join(selected)
    truncated = limit is None and end < total

    # 3. Register in the session's readFileState: the raw sliced content (without
    # line numbers) and the current mtime
    state.record(path, selected_content, offset=offset, limit=limit)

    # 4. The model receives the numbered version; append a hint when truncated
    result = _with_line_numbers(selected_content, start_line=offset)
    if truncated:
        result += f"\n\n(File has {total} lines total, {total - end} not shown. Continue with offset={end + 1})"
    return result


def read_file(ctx: RunContext[ReadFileState], path: str, offset: int = 1, limit: int | None = None) -> str:
    """
    Read a file's contents with line numbers (cat -n style).
    For large files, read in chunks with offset and limit instead of all at once.
    When editing later with edit_file, do not include line-number prefixes in
    old_string or new_string.

    Args:
        path: absolute path of the file to read
        offset: 1-based line to start from (default 1 = start of file)
        limit: number of lines to read; defaults to at most DEFAULT_MAX_LINES
    """
    # 1. Deduplicate: recorded in readFileState, came from read_file (offset not
    # None), same range, same mtime -> skip
    state = ctx.deps.read_file_state
    record = state.get(path)
    if record is not None and record.get("offset") is not None:
        if record["offset"] == offset and record["limit"] == limit:
            try:
                if os.path.getmtime(path) <= record["timestamp"]:
                    return "File contents are unchanged; refer to the previous read_file result instead of reading again"
            except FileNotFoundError:
                pass

    # 2. Reading and registration both go through read_and_register; translate
    # exceptions here into ModelRetry feedback for the model
    try:
        return read_and_register(state, path, offset, limit)
    except FileNotFoundError:
        raise ModelRetry(f"File {path} does not exist; verify the path or pick another file")
    except PermissionError:
        raise ModelRetry(f"No permission to read {path}; pick a readable file")
    except IsADirectoryError:
        raise ModelRetry(f"{path} is a directory; specify a file inside it")
    except UnicodeDecodeError:
        return f"Error: {path} is not a text file and cannot be read"


def edit_file(ctx: RunContext[ReadFileState], path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """
    Replace an exact string in a file, passing only the changed snippet to save tokens.
    The file must be read with read_file first, otherwise this raises an error.
    Prefer edit_file over write_file when modifying an existing file.
    old_string must match uniquely; add surrounding context to disambiguate it, or
    pass replace_all=true to replace every occurrence.
    Do not include read_file's line-number prefixes in old_string or new_string.

    Args:
        path: path of the file to edit
        old_string: the exact snippet to replace; must match the file character for character (without line-number prefixes)
        new_string: the replacement content
        replace_all: whether to replace every match; by default only the single unique match
    """
    # 1. No-op edit: nothing actually changes, reject it
    if old_string == new_string:
        raise ModelRetry("old_string and new_string are identical; this edit changes nothing")

    # 2. Read before write: absent from readFileState means an edit was attempted
    # without reading, so send it back to read first
    state = ctx.deps.read_file_state
    record = state.get(path)
    if record is None:
        raise ModelRetry(f"{path} has not been read yet; use read_file first, then edit based on the real content")

    # 3. mtime guard: the file changed after it was read, so editing from the old
    # content would clobber those changes; send it back to re-read
    try:
        current_mtime = os.path.getmtime(path)
    except FileNotFoundError:
        raise ModelRetry(f"{path} no longer exists and cannot be edited")
    if current_mtime > record["timestamp"]:
        raise ModelRetry(f"{path} was modified after you read it; call read_file again for the latest content before editing")

    # 4. Uniqueness check: count occurrences in the content snapshot recorded at
    # read time. Zero means it cannot be edited; several matches without
    # replace_all would be ambiguous
    snapshot = record["content"]
    count = snapshot.count(old_string)
    if count == 0:
        raise ModelRetry("old_string was not found in the file; check the original text against read_file's output")
    if count > 1 and not replace_all:
        raise ModelRetry(
            f"old_string appears {count} times in the file; add context to make it unique, or pass replace_all=true to replace all"
        )

    # 5. All checks passed; re-read the latest content from disk before replacing.
    # A permission prompt sits between validation and writing, during which the
    # file may change, so the on-disk content wins
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 6. Plain string replacement; the red/green diff in the terminal is display
    # only, and the fully replaced content is what gets written
    if replace_all:
        updated = content.replace(old_string, new_string)
    else:
        updated = content.replace(old_string, new_string, 1)

    # 7. Checkpoint pre-mutation content before writing to disk to support /rewind
    if ctx.deps.file_history:
        ctx.deps.file_history.track_edit(path)

    # 8. Write the whole file back, then refresh the record with the new content
    # and mtime
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    state.record(path, updated)
    if ctx.deps.file_history:
        ctx.deps.file_history.note_written(path)
    return f"Edited {path} ({count if replace_all else 1} replacement(s))"


def write_file(ctx: RunContext[ReadFileState], path: str, content: str) -> str:
    """
    Write content to a file wholesale; an existing file is overwritten.
    An existing file must be read with read_file first, otherwise this raises an error.
    Prefer edit_file when modifying an existing file (pass only the changed
    snippet to save tokens); use write_file only to create files or rewrite them fully.
    """
    # Overwriting an existing file follows the same read-before-write rule as
    # edit_file, so unread content is never wiped out
    state = ctx.deps.read_file_state
    if os.path.exists(path):
        record = state.get(path)
        if record is None:
            raise ModelRetry(f"{path} already exists; read it with read_file first to confirm nothing is lost")
        if os.path.getmtime(path) > record["timestamp"]:
            raise ModelRetry(f"{path} was modified after you read it; call read_file again before overwriting")
    # Checkpoint pre-mutation state before writing to disk
    if ctx.deps.file_history:
        ctx.deps.file_history.track_edit(path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except FileNotFoundError:
        raise ModelRetry(f"Directory does not exist, cannot write {path}; use an existing directory")
    except PermissionError:
        raise ModelRetry(f"No permission to write {path}; use a writable path")
    except OSError as e:
        return f"Error: failed to write {path} ({e})"
    # Newly written content is registered too, so later edits need not re-read it
    state.record(path, content)
    if ctx.deps.file_history:
        ctx.deps.file_history.note_written(path)
    return f"Wrote {path}"
