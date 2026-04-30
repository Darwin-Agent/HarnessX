# Harness Lab UI

Harness Lab is the web interface for HarnessX.

```bash
hx lab                # opens http://localhost:7861
hx lab --port 8080
```

---

## Layout overview

```
┌─────────────────────────────────────────┐
│  TopBar: Builder | Compare   ⚙ Settings │
├───────────────┬─────────────────────────┤
│  Sidebar      │  Main content area      │
│  (presets /   │  (config / chat)        │
│  custom list) │                         │
└───────────────┴─────────────────────────┘
```

---

## Builder page

Builder is a single harness workspace: configure one harness config, then run chat tasks.

### Sidebar

- **Presets**: built-in harness templates (Research, Coding, Minimal, ...)
- **Custom**: user-saved configs (browser local storage)
- **New blank harness**: start from an empty config
- **Import from YAML**: paste or drag a `.yaml` config

### Config panel

The panel exposes behavior dimensions. Each card provides:

| Control | Purpose |
|---|---|
| Enabled toggle | Include/exclude a processor or capability |
| Parameter sliders | Tune thresholds (for example token budget) |
| Expand arrow | Show advanced options |

Preset configs are read-only. Clone to a custom config for editing.

### Chat panel

After configuration, click **Start Chatting**.

- Enter task text and press `Enter`
- Use **Advanced** for success criteria and max steps
- Use **Stop** to cancel a running task
- **New chat** clears conversation history and keeps current config

---

## Compare page

Run one task across multiple configs in parallel.

- **Add column**: choose preset/custom config
- **Config** icon: edit column config
- Shared input bar sends the same task to all columns
- Each column streams independently

---

## Settings

Open from top-right **⚙ Settings**.

### Model

Manage model registry and slot mapping. See [Models](../feats/models.md).

### Workspace

Configure local workspace root or remote sandbox URL. See [Workspace & Sandbox](../feats/workspace.md).

### Tools

Enable/disable built-in tools and manage [MCP servers](../feats/mcp.md).

### Skills

Toggle global skill loading and per-skill enablement.

### Plugins

Manage plugin directories and plugin imports. See [Plugin System](../feats/plugins.md).

---

## Launch flow

```
Configure model + harness config
       ↓
Click Launch
       ↓
Backend dry-run build (validates config)
       ↓  ✓ OK             ✗ Error
    Ready badge        Error popover
       ↓
  Start Chatting
```

Launch becomes available only after backend validation succeeds.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` | Send message |
| `Shift+Enter` | New line |
| `Escape` | Close Settings / Docs panel |
