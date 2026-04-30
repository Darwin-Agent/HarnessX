# MCP Servers

[Model Context Protocol (MCP)](https://modelcontextprotocol.io) is an open standard for connecting AI models to external tools and data sources. HarnessX supports adding MCP servers that expose additional tools to the agent at runtime.

---

## How it works

An MCP server is a separate process (or HTTP service) that advertises a list of tools. When the harness starts a run, it connects to all enabled MCP servers and merges their tools into the agent's tool registry alongside the built-in tools.

```
Agent → tool call "mcp_sqlite_query" → harness → MCP server (sqlite) → result → agent
```

---

## Transport types

| Transport | When to use |
|---|---|
| **stdio** | Local process. The harness spawns the command and communicates over stdin/stdout. |
| **http** | Remote or separately-running server. The harness connects to a URL. |

---

## Adding a server in the UI

1. Open **Settings → Tools**.
2. Scroll to the **MCP Servers** section.
3. Click **Add MCP Server**.
4. Fill in:
   - **Name** — identifier shown in the UI
   - **Transport** — stdio or http
   - **Command** (stdio) — shell command to launch the server, e.g. `npx @modelcontextprotocol/server-sqlite`
   - **URL** (http) — endpoint URL, e.g. `http://localhost:3001`
   - **Environment** — optional key=value pairs passed to the server process
5. Click **Add**.

### Preview tools

Once added, click the **Preview tools** button on a server card to connect and list all tools the server exposes. This is useful for verifying the server starts correctly before running a task.

---

## Example: SQLite server

```json
{
  "name": "sqlite",
  "transport": "stdio",
  "command": "npx @modelcontextprotocol/server-sqlite --db /path/to/db.sqlite"
}
```

## Example: Custom HTTP server

```json
{
  "name": "search",
  "transport": "http",
  "url": "http://localhost:3001"
}
```

---

## plugin.json MCP integration

You can bundle MCP server configurations inside a plugin. They are automatically connected when the plugin is enabled:

```json
{
  "name": "my-plugin",
  "mcpServers": {
    "sqlite": {
      "command": "npx @modelcontextprotocol/server-sqlite",
      "args": ["--db", "data.db"]
    }
  }
}
```

Claude Code native format (`dict-of-dicts`) and HarnessX list format are both supported.

---

## Installed plugins vs. Settings UI

MCP servers can be configured in two places:

| Source | Scope |
|---|---|
| Settings → Tools → MCP Servers | Global, persisted in `~/.harnessx/mcp_servers.json` |
| Plugin `mcpServers` field | Per-plugin, enabled/disabled with the plugin |

Both are merged at run time. Plugin-defined servers cannot be edited from the Settings UI — edit the plugin's `plugin.json` instead.
