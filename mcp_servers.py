"""
MCP server registry: reads server configs from .mcp.json, connects to them at
startup, and exposes the connected servers to the Agent as toolsets.
"""
import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.toolsets import AbstractToolset

# User-level and project-level configs; project-level wins on name conflicts
USER_CONFIG = Path.home() / ".my-claude-code" / "mcp.json"
PROJECT_CONFIG = Path(".mcp.json")

LOG_DIR = Path.home() / ".my-claude-code" / "mcp-logs"


@dataclass
class ServerRecord:
    """
    Connection record for a single MCP server.
    """
    name: str
    # Toolset handed to the Agent, already prefixed with mcp__<server>__
    toolset: AbstractToolset | None = None
    # Underlying MCPToolset; kept because PrefixedToolset does not expose list_tools()
    mcp_toolset: MCPToolset | None = None
    transport: str = ""
    # pending / connected / failed
    status: str = "pending"
    error: str = ""
    tool_names: list = field(default_factory=list)


# Process-level state: /new and /resume do not reconnect
RECORDS: list[ServerRecord] = []
_stack = AsyncExitStack()


def _read_config(path: Path) -> dict:
    """
    Read the `mcpServers` mapping from one config file.
    """
    return json.loads(path.read_text(encoding="utf-8")).get("mcpServers", {})


def _log_file(name: str):
    """
    Log file that will receive a stdio server's stderr. Returns None when the log
    directory cannot be created, in which case the child inherits our stderr.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return LOG_DIR / f"{name}.log"
    except OSError:
        return None


def _build_record(name: str, cfg: dict) -> ServerRecord:
    """
    Build one server record from a single `mcpServers` entry.
    """
    extra = {}
    if "command" in cfg:
        args = [str(a) for a in (cfg.get("args") or [])]
        transport = StdioTransport(
            command=cfg["command"],
            args=args,
            env=cfg.get("env"),
            cwd=cfg.get("cwd"),
            # Keep child stderr out of the terminal UI
            log_file=_log_file(name),
        )
        desc = "stdio: " + " ".join([cfg["command"], *args])
    elif "url" in cfg:
        # MCPToolset infers the transport type (SSE vs streamable HTTP) from the URL
        transport = cfg["url"]
        desc = f"http: {cfg['url']}"
        if cfg.get("headers"):
            extra["headers"] = cfg["headers"]
    else:
        raise ValueError(f"MCP server {name!r} must define either `command` or `url`")

    inner = MCPToolset(
        transport,
        id=name,
        include_instructions=True,
        # uvx/npx may fetch the package on first launch; the default is too short
        init_timeout=30,
        **extra,
    )
    return ServerRecord(
        name=name,
        # PrefixedToolset joins prefix and tool name with "_", so the trailing
        # underscore yields the fully qualified mcp__<server>__<tool> form
        toolset=inner.prefixed(f"mcp__{name}_"),
        mcp_toolset=inner,
        transport=desc,
    )


def load_servers() -> None:
    """
    Merge both config levels into RECORDS; project-level entries win on name conflicts.
    """
    merged: dict = {}
    for path in (USER_CONFIG, PROJECT_CONFIG):
        if path.is_file():
            merged.update(_read_config(path))

    RECORDS.clear()
    for name, cfg in merged.items():
        try:
            RECORDS.append(_build_record(name, cfg))
        except Exception as e:
            # A malformed entry is reported by /mcp instead of blocking startup
            RECORDS.append(
                ServerRecord(name=name, status="failed", error=f"{type(e).__name__}: {e}")
            )


async def startup() -> str:
    """
    Connect every configured server concurrently and return a one-line summary.
    A single failure is recorded without affecting the others or blocking startup.
    """
    load_servers()
    await asyncio.gather(*(_connect(record) for record in RECORDS if record.toolset))

    connected = [r for r in RECORDS if r.status == "connected"]
    failed = [r for r in RECORDS if r.status == "failed"]
    parts = []
    if connected:
        parts.append(
            "Connected MCP servers: "
            + ", ".join(f"{r.name} ({len(r.tool_names)} tools)" for r in connected)
        )
    if failed:
        parts.append("Failed: " + ", ".join(r.name for r in failed) + " (see /mcp)")
    return "; ".join(parts)


async def _connect(record: ServerRecord) -> None:
    try:
        await _stack.enter_async_context(record.toolset)
        tools = await record.mcp_toolset.list_tools()
        record.tool_names = [f"mcp__{record.name}__{tool.name}" for tool in tools]
        record.status = "connected"
    except Exception as e:
        # Unwrap exception groups to report the innermost cause
        while getattr(e, "exceptions", None):
            e = e.exceptions[0]
        record.status = "failed"
        record.error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


async def shutdown() -> None:
    """
    Disconnect every server on exit; errors caused by already-dead child processes
    are ignored.
    """
    try:
        await _stack.aclose()
    except Exception:
        pass


def active_toolsets() -> list[AbstractToolset]:
    """
    Toolsets of the servers that connected successfully.
    """
    return [record.toolset for record in RECORDS if record.status == "connected"]
