# Plugin System

The HarnessX plugin system lets you bundle tools, processors, slash commands, MCP servers, and shell hooks into a single reusable unit. Plugins can be written in Python or declared as a `plugin.json` manifest. The manifest format is a strict superset of the Claude Code plugin format — any Claude Code plugin loads without modification.

---

## Concepts

A plugin is a self-contained capability pack:

| Capability | What it does |
|---|---|
| **tools** | Tool callables registered into the harness tool registry |
| **processors** | `MultiHookProcessor` instances wired into the run loop |
| **slash commands** | `/cmd` handlers — inject prompts, filter tools, or trigger processor logic via state slots |
| **MCP servers** | External tool servers connected at run time via stdio or HTTP |
| **shell hooks** | Shell scripts that fire at `task_end`, `before_tool`, or `after_tool` |
| **skills** | Reusable sub-agent skill packs (auto-discovered from `skills/*/SKILL.md`) |

---

## Python plugin (class-based)

The simplest way to write a plugin is to subclass `HarnessPlugin`:

```python
from harnessx.plugins import HarnessPlugin
from harnessx.core.processor import MultiHookProcessor
from typing import AsyncIterator

class LoggingProcessor(MultiHookProcessor):
    _order = 99

    async def on_task_start(self, event) -> AsyncIterator:
        print(f"Task started: {event.task.description}")
        yield event

class MyPlugin(HarnessPlugin):
    name = "my-plugin"
    version = "0.1.0"
    description = "Log every task start"
    processors = [LoggingProcessor()]
```

Register it with a `HarnessBuilder`:

```python
from harnessx.core.builder import HarnessBuilder

harness = (
    HarnessBuilder()
    .plugin(MyPlugin())
    .build()
)
```

Or register it at the global level so it applies to all harnesses in a process:

```python
from harnessx.plugins import plugin_registry
plugin_registry.register(MyPlugin())
```

---

## Manifest plugin (`plugin.json`)

A manifest plugin is a directory with a `plugin.json` at its root.

```
my-plugin/
  plugin.json
  commands/
    recall.md          # optional: command prompt files
  hooks/
    hooks.json         # optional: shell hooks
    stop.sh
  skills/
    recall/
      SKILL.md         # optional: skill pack
```

Load it:

```python
harness = HarnessBuilder().plugin("./my-plugin").build()
```

Or from the CLI:

```bash
hx plugin install ./my-plugin      # install to ~/.harnessx/plugins/ (same as: plugin add)
hx plugin list                     # list discovered plugins (installed + external)
hx plugin remove my-plugin         # uninstall
```

---

## `plugin.json` schema

All fields are optional. Unrecognised fields are silently ignored.

### Core metadata

```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "description": "What this plugin does",
  "setup": "bash ./scripts/setup.sh",
  "stop":  "bash ./scripts/teardown.sh"
}
```

`setup` and `stop` are shell commands run when the plugin is installed and removed respectively.

---

### Slash commands (`commands`)

Commands define `/cmd` handlers that prepend a system prompt when invoked. They are the primary Claude Code compatibility hook.

```json
{
  "commands": [
    {
      "name": "recall",
      "description": "Recall information from memory",
      "prompt": "You are in recall mode. Search your memory and summarise relevant facts. User query: $ARGUMENTS",
      "argument_hint": "<topic>",
      "allowed_tools": ["read_file", "search"],
      "hidden": false
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `name` | string | Command name without the `/` prefix |
| `description` | string | Shown in `/help` |
| `prompt` | string or path | System prompt prefix. Use `$ARGUMENTS` as placeholder for text typed after the command name. May be a relative path (`"./commands/recall.md"`) resolved at load time |
| `allowed_tools` | string[] | Restrict the model to *only* these tool names for this task. Empty/absent = no restriction |
| `argument_hint` | string | Placeholder shown in CLI input (e.g. `"<topic>"`) |
| `hidden` | bool | If `true`, the command is excluded from `/help` output |

**File-based prompts** — instead of inlining the prompt, point to a `.md` file:

```json
{ "name": "recall", "prompt": "./commands/recall.md" }
```

**Claude Code command directory** — place files in `commands/*.md`. The file stem becomes the command name; YAML frontmatter provides metadata:

```markdown
---
description: Recall something from memory
allowed-tools: [read_file, search]
argument-hint: "<topic>"
hide-from-slash-command-tool: false
---
You are in recall mode. $ARGUMENTS
```

---

### MCP servers (`mcpServers`)

Connect one or more [Model Context Protocol](https://modelcontextprotocol.io) servers. Their tools are injected into the harness tool registry at the first `task_start`.

**Claude Code format** (dict-of-dicts):

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "npx mcp-server-sqlite",
      "args": ["--db", "data.db"]
    },
    "search": {
      "url": "http://localhost:3001"
    }
  }
}
```

**HarnessX list format**:

```json
{
  "mcpServers": [
    {
      "name": "sqlite",
      "transport": "stdio",
      "command": "npx mcp-server-sqlite",
      "args": ["--db", "data.db"],
      "env": { "SQLITE_PATH": "/data" }
    },
    {
      "name": "search",
      "transport": "http",
      "url": "http://localhost:3001"
    }
  ]
}
```

`transport` is inferred automatically if absent: `"stdio"` when `command` is present, `"http"` when `url` is present.

The `env` field (HarnessX extension) injects extra environment variables into the MCP subprocess. They are merged on top of `os.environ`, so the subprocess inherits `PATH` and other system variables.

**Retry behaviour** — if the MCP server is unavailable at first `task_start`, the processor retries with exponential back-off (1 s, 2 s, 4 s) for up to 3 attempts. After the third failure the processor gives up silently (no exception propagates to the run loop).

---

### HarnessX processors (`processors`)

Register `MultiHookProcessor` subclasses defined in Python:

```json
{
  "processors": [
    {
      "target": "my_plugin.processors.RecallProcessor",
      "top_k": 5,
      "threshold": 0.7
    }
  ]
}
```

`target` is the dotted import path to the class. All other fields are passed as `**kwargs` to the constructor. Objects with a `_note` field are treated as TODO stubs and skipped.

---

### HarnessX tools (`tools`)

Register tool callables:

```json
{
  "tools": [
    { "target": "my_plugin.tools.search_memory" },
    { "target": "my_plugin.tools.write_memory" }
  ]
}
```

Objects with a `_note` field are skipped (generated by `harnessx plugin convert`).

---

### Slash command slots (`slash_commands`)

Map a slash command to a State slot key. This drives processor-triggered commands (where the command logic lives in a processor, not in a prompt injection):

```json
{
  "slash_commands": [
    { "command": "/compact", "slot": "_force_compact" },
    { "command": "/recall",  "slot": "_trigger_recall" }
  ]
}
```

When the user types `/compact`, the registry sets `state.slots["_force_compact"] = True` and calls `harness.run()`. A processor that checks this slot at `on_task_start` can then perform the compaction logic.

---

## Shell hooks (`hooks/hooks.json`)

Shell scripts that run at lifecycle events. Supports both HarnessX flat format and the Claude Code nested format.

**Flat format**:

```json
{
  "Stop": [
    { "type": "command", "command": "bash ./hooks/stop.sh" }
  ],
  "PreToolUse": [
    { "matcher": "Bash", "command": "bash ./hooks/pre-bash.sh" }
  ],
  "PostToolUse": [
    { "type": "command", "command": "bash ./hooks/post.sh" }
  ]
}
```

**Claude Code nested format**:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "done",
        "hooks": [{ "type": "command", "command": "bash ./hooks/stop.sh" }]
      }
    ]
  }
}
```

| Claude Code event | HarnessX hook | `matcher` tests against |
|---|---|---|
| `Stop` | `on_task_end` | `TaskEndEvent.exit_reason` |
| `PreToolUse` | `on_before_tool` | `ToolCallEvent.tool_name` |
| `PostToolUse` | `on_after_tool` | `ToolResultEvent.tool_name` |

The `matcher` field is a Python regex (`re.search`). Omitting it (or setting to `""` / `null`) fires the hook unconditionally.

`$CLAUDE_PLUGIN_ROOT` is set to the plugin directory so scripts can use relative paths:

```bash
#!/bin/bash
# hooks/stop.sh
echo "Plugin root: $CLAUDE_PLUGIN_ROOT"
python "$CLAUDE_PLUGIN_ROOT/scripts/export_session.py"
```

---

## Claude Code plugin compatibility

HarnessX loads Claude Code plugins without any conversion. Both directory layouts are supported:

**HarnessX layout**:
```
my-plugin/
  plugin.json
```

**Claude Code layout** (auto-detected):
```
my-plugin/
  .claude-plugin/
    plugin.json
  commands/
    recall.md
  skills/
    memory/
      SKILL.md
  hooks/
    hooks.json
```

Load either layout the same way:

```python
harness = HarnessBuilder().plugin("./my-plugin").build()
```

The only Claude Code capabilities not yet mapped to HarnessX are:

| Claude Code feature | Status |
|---|---|
| `commands/*.md` → prompt injection | Supported |
| `mcpServers` → MCP tool injection | Supported |
| `hooks/hooks.json` → lifecycle hooks | Supported (with `matcher` regex) |
| `skills/*/SKILL.md` → skill packs | Discovery supported; install wired via `SkillInstallProcessor` |
| `setup` / `stop` shell scripts | Supported |

---

## `HarnessBuilder.plugin()` API

```python
from harnessx.core.builder import HarnessBuilder
from pathlib import Path

harness = (
    HarnessBuilder()
    # From a Python class or instance:
    .plugin(MyPlugin())
    .plugin(MyPlugin)          # class is auto-instantiated

    # From a filesystem path (str or Path):
    .plugin("./plugins/memory")
    .plugin(Path("~/.harnessx/plugins/search").expanduser())

    # From a dotted Python import path:
    .plugin("my_package.plugins.MemoryPlugin")

    .build()
)
```

`plugin()` is chainable and idempotent — registering a plugin with the same `name` twice is a no-op.

When `plugin()` is called, it:

1. Loads the plugin via `load_plugin()`
2. Registers all `processors` into the builder (preserving `_order` and `_singleton_group`)
3. Registers all `tools` into the tool registry
4. Mounts a single `McpRuntimePlugin` and feeds plugin `mcp_servers` into it
5. Adds a `ShellHookProcessor` if `hooks/hooks.json` exists
6. Adds a `CommandInjectionProcessor` for all `commands`
7. Preserves discovered `skill_dirs` for runtime skill discovery

---

## Writing a processor-driven slash command

Some commands should trigger internal logic rather than prompt injection. Use the slot mechanism:

**1. Declare the command in your plugin:**

```python
class MyPlugin(HarnessPlugin):
    name = "my-plugin"
    slash_commands = {"/recall": "_trigger_recall"}
    processors = [RecallProcessor()]
```

**2. Check the slot in your processor:**

```python
class RecallProcessor(MultiHookProcessor):
    _order = 1

    async def on_task_start(self, event) -> AsyncIterator:
        if event.state.slots.pop("_trigger_recall", None):
            # inject recalled context into system_prompt
            recalled = await self._recall(event.task)
            event = dataclasses.replace(
                event,
                system_prompt=recalled + "\n\n" + event.system_prompt,
            )
        yield event
```

When the user types `/recall`, the registry sets `state.slots["_trigger_recall"] = True` and calls `harness.run()`. On the next `on_task_start`, your processor sees the slot, pops it, and injects its context.

---

## Plugin CLI reference

```
hx plugin list
    List discovered plugins with status/source/path.
    External plugins (for example from Claude cache) are shown as [external].

hx plugin add <path>
    Copy a plugin directory into ~/.harnessx/plugins/.

hx plugin install <src>
    Alias of `plugin add`.
    `src` can be:
      - a plugin directory path
      - a plugin name from `hx plugin list` (when uniquely resolvable)
      - a dotted Python path

hx plugin remove <name> [--yes]
    Delete a plugin from ~/.harnessx/plugins/.
    Prompts for confirmation unless --yes is given.

hx plugin convert <src-dir> [--out <dst-dir>]
    Convert a Claude Code plugin to an HarnessX extended manifest.
    Generates plugin.json with processor/tool stubs and a processors/stub.py
    template with TODO comments.
```

---

## Built-in plugins

| Plugin | Registered by | Slash commands |
|---|---|---|
| `_builtin.session` | `hx` CLI | `/new`, `/compact`, `/session`, `/help`, `/quit` |

The session plugin is registered automatically when you run the CLI. In API usage it is not registered by default — create and register it yourself if you need those commands:

```python
from harnessx.plugins.builtins.session import SessionPlugin
from harnessx.plugins import plugin_registry
plugin_registry.register(SessionPlugin())
```

---

## Example: full manifest plugin

```json
{
  "name": "memory-boost",
  "version": "0.1.0",
  "description": "Persistent memory via vector search",

  "setup": "bash ./scripts/setup.sh",
  "stop":  "bash ./scripts/teardown.sh",

  "commands": [
    {
      "name": "recall",
      "description": "Recall from memory",
      "prompt": "Relevant memories:\n$ARGUMENTS\n\nUse these as context.",
      "allowed_tools": ["read_file", "vector_search"],
      "argument_hint": "<topic>"
    },
    {
      "name": "save",
      "description": "Save to memory",
      "prompt": "Save the following to memory: $ARGUMENTS",
      "hidden": false
    }
  ],

  "mcpServers": {
    "vector": {
      "command": "npx @my-org/mcp-vector",
      "args": ["--db", "~/.memory.db"],
      "env": { "VECTOR_KEY": "secret" }
    }
  },

  "processors": [
    {
      "target": "memory_boost.processors.AutoSaveProcessor",
      "min_importance": 0.6
    }
  ],

  "tools": [
    { "target": "memory_boost.tools.vector_search" }
  ],

  "slash_commands": [
    { "command": "/forget", "slot": "_trigger_forget" }
  ]
}
```
