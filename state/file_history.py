"""
File checkpoints: record a checkpoint for tracked files on every user input, so
/rewind can roll both code and conversation back to an earlier point in time.

Two recording moments: track_edit backs up a file's pre-change contents as v1 the
first time a tool touches it; make_checkpoint runs on every user input and stores
a new version for any tracked file whose contents changed.

When rewinding to a checkpoint: files recorded in that checkpoint are restored to
the matching version; tracked files not recorded in it fall back to their v1 (the
state before they were first touched); a v1 of None means "did not exist yet", so
rewinding deletes the file.

Backup files live in the session bundle at sessions/<session_id>/file-history/;
checkpoint metadata is persisted in checkpoints.json, so checkpoints taken before
/resume remain usable afterwards. Files changed by run_command are out of scope for
tracking.
"""
import difflib
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.paths import get_session_dir


@dataclass
class Checkpoint:
    """
    A single checkpoint, created before the user input enters the conversation history.
    """
    # Conversation history length at checkpoint time; rewinding truncates back to this index
    history_index: int
    # Raw user input, refilled into the input box after rewinding; truncated for list display
    prompt: str
    timestamp: float
    # path -> version number of that file at checkpoint time
    backups: dict = field(default_factory=dict)
    # TasksStore.snapshot() taken at checkpoint time; None for checkpoints written
    # before task rollback existed
    tasks: dict | None = None


@dataclass
class FileChange:
    """
    A single file change: path, what happened, and the line count delta.
    """
    path: str
    # In a rewind plan: restore (restore the target version) or delete (target version absent, so remove the file).
    # In per-turn change stats: create / edit / delete
    action: str
    insertions: int
    deletions: int


def _count_diff(current, target) -> tuple:
    """
    Count the lines that would be added and removed when rewinding from current to target.
    """
    cur = (current or b"").decode("utf-8", "replace").splitlines()
    tgt = (target or b"").decode("utf-8", "replace").splitlines()
    insertions = deletions = 0
    matcher = difflib.SequenceMatcher(None, cur, tgt, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            deletions += i2 - i1
        if tag in ("replace", "insert"):
            insertions += j2 - j1
    return insertions, deletions


class FileHistory:
    """
    Session-level file checkpoint store: version backups on disk, checkpoint list in checkpoints.json.
    """

    def __init__(self, session_id: str):
        self.dir = get_session_dir(session_id) / "file-history"
        # Checkpoint list, oldest to newest
        self.checkpoints: list[Checkpoint] = []
        # path -> backup filenames per version, index i maps to version i+1; None means that version does not exist
        self.versions: dict[str, list] = {}
        # path -> sha256 of the content the agent last left on disk. Used to tell the
        # agent's own writes apart from edits made outside the file tools
        self.known: dict[str, str | None] = {}
        self._load()

    # —— Recording ——

    def track_edit(self, path: str) -> None:
        """
        Called before a tool writes a file: back up the pre-change contents as v1 the
        first time this file is touched.
        """
        path = os.path.abspath(path)
        if path in self.versions:
            return
        self.versions[path] = [self._backup(path, 1)]
        # The pre-write content is the baseline the agent is expected to find again
        self.known[path] = self._hash_current(path)
        self._save()

    def note_written(self, path: str) -> None:
        """
        Record the content the agent just wrote. Called by the file tools after a
        successful write, so a later rewind can distinguish the agent's own changes
        from edits made outside the file tools.
        """
        path = os.path.abspath(path)
        self.known[path] = self._hash_current(path)
        self._save()

    def externally_modified(self) -> set[str]:
        """
        Tracked files whose on-disk content no longer matches what the agent last wrote.

        These carry edits made outside the agent's file tools (manual edits, a
        formatter, another process). /rewind leaves them untouched rather than
        silently discarding that work.
        """
        return {
            path for path, expected in self.known.items()
            if self._hash_current(path) != expected
        }

    def make_checkpoint(self, history_index: int, prompt: str, tasks: dict | None = None) -> None:
        """
        Create a checkpoint on user input: store a new version for every tracked file
        whose contents changed.
        A checkpoint is still recorded when nothing is tracked, because rewinding the
        conversation needs it.
        """
        backups = {}
        for path, names in self.versions.items():
            latest = len(names)
            if self._current(path) != self._read_version(path, latest):
                names.append(self._backup(path, latest + 1))
            backups[path] = len(names)
        self.checkpoints.append(
            Checkpoint(history_index, prompt, time.time(), backups, tasks)
        )
        self._save()

    # —— Rewind ——

    def diff_stats(self, cp: Checkpoint, skip: set[str] | None = None) -> list:
        """
        Read-only rewind execution plan: one FileChange per file that would be touched,
        with unchanged files omitted.
        This is the cumulative difference from the current disk state back to the checkpoint.
        Paths in `skip` are left out entirely.
        """
        skip = skip or set()
        plan = []
        for path in self.versions:
            if path in skip:
                continue
            target = self._content_at(cp, path)
            current = self._current(path)
            if target == current:
                continue
            action = "delete" if target is None else "restore"
            insertions, deletions = _count_diff(current, target)
            plan.append(FileChange(path, action, insertions, deletions))
        return plan

    def turn_stats(self, cp: Checkpoint) -> list:
        """
        File changes produced by the turn that this checkpoint opened: compared against
        the next checkpoint, or against the current disk state for the latest checkpoint.
        """
        index = self.checkpoints.index(cp)
        nxt = self.checkpoints[index + 1] if index + 1 < len(self.checkpoints) else None
        changes = []
        for path in self.versions:
            before = self._content_at(cp, path)
            after = self._content_at(nxt, path) if nxt else self._current(path)
            if before == after:
                continue
            if before is None:
                action = "create"
            elif after is None:
                action = "delete"
            else:
                action = "edit"
            insertions, deletions = _count_diff(before, after)
            changes.append(FileChange(path, action, insertions, deletions))
        return changes

    def rewind_files(self, cp: Checkpoint, skip: set[str] | None = None) -> list:
        """
        Apply the execution plan to disk and return the FileChanges actually performed.
        Paths in `skip` are not touched.
        """
        plan = self.diff_stats(cp, skip)
        for change in plan:
            if change.action == "delete":
                os.remove(change.path)
            else:
                # copy2 brings file permissions back from the backup too
                version = self._target_version(cp, change.path)
                name = self.versions[change.path][version - 1]
                shutil.copy2(self.dir / name, change.path)
        # The restored contents are now what the agent last left on disk; refresh the
        # baseline so a later rewind does not mistake them for external edits
        for change in plan:
            self.known[change.path] = self._hash_current(change.path)
        self._save()
        return plan

    def drop_from(self, cp: Checkpoint) -> None:
        """
        After a conversation rewind, every checkpoint from this one onward points at
        messages that no longer exist, so drop that whole tail.
        """
        self.checkpoints = self.checkpoints[: self.checkpoints.index(cp)]
        self._save()

    # —— Internal helpers ——

    def _target_version(self, cp: Checkpoint, path: str) -> int:
        # The checkpoint has no record of this file, so it was first touched after the
        # checkpoint was taken; rewind to v1
        return cp.backups.get(path, 1)

    def _content_at(self, cp: Checkpoint, path: str):
        # The file's contents at checkpoint time; None means it did not exist then
        return self._read_version(path, self._target_version(cp, path))

    def _backup(self, path: str, version: int):
        """
        Copy the file's current contents into the backup directory and return the
        backup filename; returns None when the file does not exist.
        """
        if not os.path.exists(path):
            return None
        digest = hashlib.sha256(path.encode()).hexdigest()[:16]
        name = f"{digest}@v{version}"
        self.dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, self.dir / name)
        return name

    def _read_version(self, path: str, version: int):
        # Read the backup contents of one version; always bytes, so non-text files work too
        name = self.versions[path][version - 1]
        return None if name is None else (self.dir / name).read_bytes()

    def _current(self, path: str):
        try:
            return Path(path).read_bytes()
        except FileNotFoundError:
            return None

    @staticmethod
    def _hash_current(path: str):
        # sha256 of the file's bytes, or None when it does not exist
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return None

    # —— Persistence ——

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {
            "checkpoints": [asdict(s) for s in self.checkpoints],
            "versions": self.versions,
            "known": self.known,
        }
        (self.dir / "checkpoints.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        # Rebuild checkpoints from checkpoints.json when resuming; a new session has no
        # such file and starts empty
        path = self.dir / "checkpoints.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.checkpoints = [Checkpoint(**s) for s in data["checkpoints"]]
        self.versions = data["versions"]
        self.known = data.get("known", {})
