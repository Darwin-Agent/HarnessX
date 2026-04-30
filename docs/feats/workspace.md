# Workspace & Sandbox

Workspace defines where tools read/write files during a run.
Sandbox defines where those tools execute.

---

## Workspace root

By default, HarnessX derives workspace from `HARNESSX_HOME`:

```
~/.harnessx/workspaces/{agent_id}/{project}/
```

Common subdirectories used at runtime:

- `sessions/`: session index + state checkpoints
- `configs/`: resolved harness config snapshots

---

## Sandbox modes

| Mode | Description |
|---|---|
| Local | Tool execution happens on host via local sandbox provider |
| Remote | Tool execution is forwarded to a remote sandbox service |

Configure in **Settings → Workspace**.

---

## Local workspace behavior

Before run start, workspace initialization may stage:

- skill files
- system prompt fragments
- workspace templates

If skill loading is enabled in config, matching skills are copied into the workspace so tools can access them as regular files.

---

## Remote sandbox

For remote mode, set:

1. remote sandbox URL
2. optional remote workspace directory

HarnessX then forwards tool calls (read/write/bash/etc.) to that remote endpoint.

---

## File manager (Lab)

In local mode, Workspace settings provide an inline file manager:

- browse files under workspace root
- edit supported text files
- save directly through backend APIs

Path traversal outside workspace root is blocked.

---

## Environment variables

- `HARNESSX_HOME`: override agent home root (default `~/.harnessx`)
- `HARNESSX_AGENT`: default agent id
- `HARNESSX_PROJECT`: default project name
- `HARNESSX_WORKSPACE`: legacy direct workspace override
