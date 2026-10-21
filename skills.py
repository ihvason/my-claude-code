"""
Skill discovery and progressive loading: only metadata is injected at startup, and the
full SKILL.md is read once a task matches.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.exceptions import ModelRetry

from utils import get_data_dir


USER_SKILLS_DIR = Path.home() / ".my-claude-code" / "skills"
PROJECT_SKILLS_SUBDIR = Path(".my-claude-code") / "skills"
FRONTMATTER_MAX_LINES = 100
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 1024
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


# Discovery keeps only metadata and the path; the body is read after a task matches
@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: Path
    source: str


def _decode_scalar(value: str) -> str:
    """
    Parse the single-line YAML string used by name and description without adding a YAML
    dependency.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            return str(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def _read_frontmatter(path: Path) -> dict[str, str]:
    """
    Read only the frontmatter header; discovery never pulls the instruction body into
    memory.
    """
    try:
        with open(path, encoding="utf-8") as f:
            if f.readline().strip() != "---":
                return {}
            fields: dict[str, str] = {}
            for _ in range(FRONTMATTER_MAX_LINES):
                line = f.readline()
                if not line:
                    return {}
                # The second delimiter ends the frontmatter; the body is ignored here
                if line.strip() == "---":
                    return fields
                if line.lstrip().startswith("#"):
                    continue
                key, sep, value = line.partition(":")
                if sep:
                    fields[key.strip()] = _decode_scalar(value)
    except (OSError, UnicodeError):
        return {}
    return {}


def _is_valid_metadata(fields: dict[str, str], directory_name: str) -> bool:
    name = fields.get("name", "")
    description = fields.get("description", "")
    return (
        name == directory_name
        and 0 < len(name) <= MAX_NAME_CHARS
        and SKILL_NAME_PATTERN.fullmatch(name) is not None
        and 0 < len(description) <= MAX_DESCRIPTION_CHARS
        and "<" not in name
        and ">" not in name
        and "<" not in description
        and ">" not in description
    )


def _scan_root(root: Path, source: str) -> list[SkillInfo]:
    """
    Only discover root/<skill-name>/SKILL.md; deeper directories hold resources and are
    not scanned as skills.
    """
    try:
        directories = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        return []

    result = []
    for directory in directories:
        path = directory / "SKILL.md"
        if not path.is_file():
            continue
        fields = _read_frontmatter(path)
        if not _is_valid_metadata(fields, directory.name):
            continue
        result.append(SkillInfo(
            name=fields["name"],
            description=fields["description"],
            path=path,
            source=source,
        ))
    return result


def _project_skills_dir(cwd: Path | None) -> Path:
    """
    Project-level skills directory.

    With no explicit project root it resolves from the application data directory, so
    launching the app from a subdirectory still maps to the same skills instead of an
    empty directory.
    """
    if cwd is not None:
        return Path(cwd) / PROJECT_SKILLS_SUBDIR
    return get_data_dir() / "skills"


def discover_skills(
    cwd: Path | None = None,
    user_skills_dir: Path | None = None,
) -> list[SkillInfo]:
    """
    Merge user-level and project-level skills; on a name collision the project-level
    definition wins because it is closer to the current project.
    """
    project_root = _project_skills_dir(cwd)
    user_root = user_skills_dir or USER_SKILLS_DIR
    # Collect user-level first, then overwrite by name with project-level, which gives
    # the desired override
    by_name = {
        skill.name: skill
        for skill in _scan_root(user_root, "user")
    }
    for skill in _scan_root(project_root, "project"):
        by_name[skill.name] = skill
    return sorted(by_name.values(), key=lambda skill: skill.name)


def format_skill_listing(skills: list[SkillInfo]) -> str:
    """
    Build the lightweight listing injected into the system prompt; no SKILL.md body ever
    appears here.
    """
    if not skills:
        return ""
    lines = [
        "Available skills (name and description only):",
    ]
    lines.extend(f"- {skill.name}: {skill.description}" for skill in skills)
    # The listing also says when to call the loading tool, so the user only has to describe
    # the task
    lines.extend([
        "",
        "When a skill matches the user's task, call load_skill before proceeding; do not load unrelated skills.",
    ])
    return "\n".join(lines)


def _read_body(path: Path) -> str:
    """
    The second loading layer happens here: read the whole file and drop the frontmatter
    that was only used for discovery.
    """
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return content.strip()
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[index + 1:]).strip()
    return content.strip()


def read_skill(
    name: str,
    cwd: Path | None = None,
    user_skills_dir: Path | None = None,
) -> str:
    # Resolve again from the current directory, keeping the same override rule as discovery
    skills = {skill.name: skill for skill in discover_skills(cwd, user_skills_dir)}
    skill = skills.get(name.strip())
    if skill is None:
        available = ", ".join(skills) or "(none)"
        raise ModelRetry(f"Unknown skill: {name}. Available skills: {available}")
    try:
        body = _read_body(skill.path)
    except (OSError, UnicodeError) as error:
        raise ModelRetry(f"Cannot read skill {name}: {error}")
    if not body:
        raise ModelRetry(f"Skill {name} has no instruction body")
    # Include the root directory so the model can resolve relative paths to references and
    # scripts mentioned in the body
    return (
        f"Skill {skill.name} root directory: {skill.path.parent}\n\n"
        f"{body}"
    )


def load_skill(name: str) -> str:
    """
    Load the full instructions of a skill once its description matches the current task.

    Args:
        name: exact name from the available skills list
    """
    return read_skill(name)


__all__ = [
    "SkillInfo",
    "discover_skills",
    "format_skill_listing",
    "load_skill",
    "read_skill",
]
