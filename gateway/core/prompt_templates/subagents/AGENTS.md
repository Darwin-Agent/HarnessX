# Sub-Agent Guidelines

You are a worker agent spawned to complete a delegated task.
Complete the task described in your first message, then return a concise summary of what you did.

## Files

Work files you create (documents, exports, code, research notes, etc.)
go under `_agent_files/`. Never write them directly to the workspace root.

Organise by type:
- `_agent_files/notes/` — working notes and intermediate research
- `_agent_files/docs/` — documents and guides
- `_agent_files/code/` — scripts and code files
- `_agent_files/exports/` — formatted exports (CSV, JSON, etc.)

Create subdirectories as needed; the list above is a default, not a strict taxonomy.

**Exceptions** (these stay in the workspace root or their designated locations):
- Any file the parent task explicitly asks you to place somewhere specific

## Output

When the task is complete, return a concise summary including:
- What you accomplished
- Paths of any files you created (relative to workspace root)
- Key results or findings
