# Slack Channel Configuration Guide

---

## Prerequisites

You need administrator access to a Slack Workspace in order to install the application.

---

## Step 1: Create a Slack App

1. Open [Slack API Console](https://api.slack.com/apps) → **Create New App**
2. Select **From scratch**
3. Enter an App Name (e.g. `HarnessX`) and select the target Workspace
4. Click **Create App**

---

## Step 2: Enable Socket Mode (Recommended)

Socket Mode requires no public IP; Slack pushes events to your service via WebSocket.

1. Go to the application → **Settings** → **Socket Mode** → enable **Enable Socket Mode**
2. Enter a Token Name (e.g. `gateway-token`)
3. Click **Generate** → copy the generated **App-Level Token** (starts with `xapp-`)

---

## Step 3: Configure OAuth Permissions

Go to the application → **OAuth & Permissions** → **Scopes** → **Bot Token Scopes** and add the following permissions:

| Scope | Purpose |
|-------|---------|
| `app_mentions:read` | Receive @mention messages |
| `channels:history` | Read public channel message history |
| `channels:read` | Read channel information |
| `chat:write` | Send messages |
| `files:read` | Read files (images and other attachments) |
| `files:write` | Upload files |
| `groups:history` | Read private channel messages (optional) |
| `groups:read` | Read private channel information (optional) |
| `im:history` | Read DM messages |
| `im:read` | Read DM information |
| `im:write` | Send DMs |
| `mpim:history` | Read multi-party DMs (optional) |
| `reactions:read` | Read reactions |
| `reactions:write` | Add/remove reactions |
| `users:read` | Read user information |

---

## Step 4: Subscribe to Events

Go to the application → **Event Subscriptions** → **Enable Events**:

- Under **Subscribe to bot events**, add:
  - `message.channels` — Public channel messages
  - `message.groups` — Private channel messages (optional)
  - `message.im` — DM messages
  - `message.mpim` — Multi-party DMs (optional)
  - `app_mention` — @mention messages

---

## Step 5: Install the Application

1. Go to the application → **OAuth & Permissions** → **Install to Workspace** → authorize
2. After installation, copy the **Bot User OAuth Token** (starts with `xoxb-`)

---

## Step 6: Add the Bot to a Channel

In a Slack channel, use the `/invite @HarnessX` command to invite the bot to join the channel.

---

## Config File

```yaml
channels:
  slack:
    enabled: true
    channel_type: slack
    bot_token: "xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx"
    app_token: "xapp-1-xxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    require_mention: true    # Whether @mention is required in channels
    reply_in_thread: true    # Whether to reply in a Thread (recommended, keeps channels clean)
    reply_broadcast: false   # Whether Thread replies are also posted to the channel (broadcast)
```

---

## Complete Field Reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `bot_token` | string | ✅ | — | Bot User OAuth Token (starts with `xoxb-`) |
| `app_token` | string | ✅ | — | App-Level Token, for Socket Mode only (starts with `xapp-`) |
| `signing_secret` | string | | — | Signing Secret (for Webhook mode verification; not needed in Socket Mode) |
| `require_mention` | bool | | `true` | Whether @mention is required in channels to trigger the bot (DMs always respond) |
| `reply_in_thread` | bool | | `true` | Whether to reply in a message Thread |
| `reply_broadcast` | bool | | `false` | Whether Thread replies are also broadcast to the channel |
| `session_mode` | string | | `shared` | `shared` or `per_user` |
| `max_steps` | int | | `30` | Maximum steps per task |
| `token_budget` | int | | `100000` | Token limit per task |
| `workspace` | string | | `auto` | File workspace mode |

---

## Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Text messages | ✅ | Slack mrkdwn format |
| Image/file reception | ✅ | Auto-downloaded and passed to the Agent |
| Streaming edits (typewriter effect) | ✅ | edit_interval = 0.8s |
| Emoji Reaction progress | ✅ | ⏳ Processing → ✅ Done / ❌ Failed |
| Thread replies | ✅ | `reply_in_thread: true` (recommended) |
| Block Kit messages | ✅ | Send rich-text cards via `send_blocks()` |
| Message length limit | — | 40,000 characters |

---

## Thread Replies

When `reply_in_thread: true` (default), all bot replies appear within the original message's Thread, keeping the channel clean. This is recommended for large teams.

Setting `reply_broadcast: true` also makes Thread replies appear in the channel's main timeline, useful for important announcements.

DM conversations are not affected by this setting and always reply directly.

---

## Block Kit Support

Gateway supports sending Slack Block Kit rich-text messages via the `send_blocks()` method, suitable for code blocks, tables, and other structured output.

When a long reply contains code blocks, Gateway automatically converts them to Block Kit Section + Code Block format for correct rendering in Slack.

---

## Troubleshooting

### Socket Mode connection fails

- Confirm that `app_token` is an App-Level Token (starts with `xapp-`), not the Bot Token
- Confirm that Socket Mode has been enabled in the console
- View logs: `hx-gateway logs -n 100 | grep slack`

### Bot is not receiving channel messages

- Confirm the bot has joined the channel via `/invite`
- Check that the relevant events are enabled under **Event Subscriptions**
- When `require_mention: true`, confirm the message includes an @mention

### `missing_scope` error

This indicates a missing permission scope. Add the scope under **OAuth & Permissions**, then reinstall the application.

### Rate limit exceeded (rate_limit_exceeded)

Slack Web API limits:
- `chat.update` — Up to 5 times per second per channel (for streaming edits)
- `chat.postMessage` — Approximately 100 messages per minute per channel

If limits are triggered, increase `stream_edit_interval` or reduce trigger frequency. Gateway has built-in backoff retry.

---

## Complete Configuration Example

```yaml
channels:
  slack:
    enabled: true
    channel_type: slack
    bot_token: "xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx"
    app_token: "xapp-1-xxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    require_mention: true
    reply_in_thread: true
    reply_broadcast: false
    session_mode: shared
    max_steps: 30
    token_budget: 100000
```
