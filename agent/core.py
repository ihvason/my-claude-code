"""
Agent instantiation: wires model / instructions / tools / hooks together.
"""
import os
import platform
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from core.paths import find_agents_md
from memory import store

from . import skills, subagents
from .deps import AgentDeps
from .hooks import hooks
from .instructions import INSTRUCTIONS
from .model import model
from .tools import TOOLS


agent = Agent(
    model,
    # System Prompt.
    instructions=INSTRUCTIONS,
    tools=TOOLS,
    deps_type=AgentDeps,
    capabilities=[hooks],
)


# Dynamic instructions: re-evaluated on each model request to inject environment context and project-level AGENTS.md.
# Like static instructions, these are excluded from conversation history while delivering up-to-date values each turn,
# remaining invisible to users without polluting context history.
@agent.instructions
def project_context() -> str:
    cwd = os.getcwd()
    parts = [
        "Environment context:",
        f"- Working directory: {cwd}",
        f"- Operating system: {platform.system()}",
        f"- Today's date: {date.today().isoformat()}",
    ]

    # Read project-level guidelines from AGENTS.md at the project root; append them so
    # the model follows the project's conventions
    agents_md = find_agents_md(cwd)
    if agents_md is not None:
        try:
            content = agents_md.read_text(encoding="utf-8")
        except OSError:
            content = ""
        if content.strip():
            parts.append("")
            parts.append("Project AGENTS.md guidelines (adhere to these conventions):")
            parts.append(content)

    # Inject the available sub agent types: custom agents are loaded from disk at
    # startup, so dynamic injection avoids hardcoding the list in the tool description
    parts.append("")
    parts.append("Agent types available to run_agent:")
    parts.append(subagents.format_agent_listing())

    # The skill listing carries only name + description; the full SKILL.md is read through
    # load_skill once the model decides it is relevant
    skill_listing = skills.format_skill_listing(skills.discover_skills())
    if skill_listing:
        parts.append("")
        parts.append(skill_listing)

    # Inject memory index: MEMORY.md is fully injected each turn so the model always knows what memory topics are available.
    # The benefit of using dynamic instructions is retrieving fresh values each turn—memories written in the current turn immediately appear in the next turn's index.
    index = store.read_index()
    if index:
        parts.append("")
        parts.append(f"Below is your memory index MEMORY.md (located at {store.index_path()}, preserved across sessions). Use read_file on the corresponding file in the memory directory whenever you need the full content of a memory:")
        parts.append(index)

    return "\n".join(parts)
