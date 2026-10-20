# Custom subagents

Every `*.md` file in this directory defines one subagent. The frontmatter supplies
its identity and tool allowlist; the markdown body becomes the subagent's system
prompt.

```markdown
---
name: explore
description: Read-only search agent for broad codebase sweeps
tools: read_file, run_command
---

You are a read-only exploration agent. Search the codebase and report findings;
never modify files.
```

- `name` and `description` are required. A definition missing either raises at
  startup, so mistakes surface immediately instead of producing a broken agent.
- `tools` is a comma-separated allowlist of tool names. Omit it (or leave it empty)
  to grant every tool available to subagents.
- `name` must not collide with a built-in agent type.
- Files without frontmatter, such as this README, are skipped silently, so notes can
  live here safely.

Unlike `projects/` (sessions, tasks, jobs, memory), this directory is configuration
and is tracked by git.
