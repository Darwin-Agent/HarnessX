# Discord Channel Configuration Guide

---

## Prerequisites

You need a Discord account and **administrator** access to the target server (Guild) in order to invite the bot.

---

## Step 1: Create a Discord Application

1. Open [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application** and enter an application name (e.g. `HarnessX`)
3. Go to the application → **Bot** tab
4. Click **Add Bot** → confirm

---

## Step 2: Configure Bot Permissions and Intents

### Enable Privileged Gateway Intents

Go to the application → **Bot** → **Privileged Gateway Intents** and enable:

- **MESSAGE CONTENT INTENT** — Required; needed to read message content (must be explicitly enabled since 2022)
- **SERVER MEMBERS INTENT** — Optional; used to read member information
- **PRESENCE INTENT** — Optional

> **Important**: If the bot is in more than 100 servers, Discord verification (Verified Bot) is required.

### Get the Bot Token

1. Go to the **Bot** tab → **Token** → **Reset Token**
2. Copy the token (format similar to `MTxxxx.Gxxxx.xxxxxxxx`)
3. Keep it safe and do not expose it (the token is equivalent to a password)

---

## Step 3: Invite the Bot to a Server

1. Go to the application → **OAuth2** → **URL Generator**
2. Under **Scopes**, check: `bot`, `applications.commands`
3. Under **Bot Permissions**, check:

   | Permission | Description |
   |------------|-------------|
   | Send Messages | Send messages (required) |
   | Send Messages in Threads | Send in Threads (optional) |
   | Read Message History | Read message history |
   | Add Reactions | Add reactions (progress feedback) |
   | Embed Links | Send Embed messages |
   | Attach Files | Send files (optional) |
   | Manage Messages | Edit messages (required for streaming output) |
   | Use Slash Commands | Use slash commands |

4. Copy the generated OAuth2 URL, open it in a browser, select the target server, and complete the invite

---

## Step 4: Get the Application ID (Optional)

If you need Slash Command interactions (`/help`, etc.), you need the **Application ID**:

Go to the application → **General Information** → copy the **Application ID**

---

## Config File

### Basic Configuration (Gateway Bot Mode)

```yaml
channels:
  discord:
    enabled: true
    channel_type: discord
    bot_token: "MTxxxxxxxxxxxxxxxxxx.Gxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    require_mention: false   # Whether @mention is required (DMs always respond)
```

### Configuration with Access Control

```yaml
channels:
  discord:
    enabled: true
    bot_token: "MTxxxxxxxxxxxxxxxxxx.Gxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    application_id: "1234567890123456789"  # Required for slash commands
    allowed_guilds:
      - "1234567890123456789"   # Server ID; empty = allow all servers
    require_mention: false
    reply_in_thread: false
    max_steps: 30
```

### Enable Interactions (Slash Commands)

To support Slash Commands from Discord Interactions (via Webhook rather than the Gateway Bot):

```yaml
channels:
  discord:
    enabled: true
    bot_token: "MTxxxxxxxxxxxxxxxxxx..."
    application_id: "1234567890123456789"
    public_key: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

> `public_key` is found under **General Information** → **Public Key** and is used to verify Interactions Webhook signatures.

---

## Complete Field Reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `bot_token` | string | ✅ | — | Bot Token (obtained from the Bot tab) |
| `public_key` | string | Required for Interactions | — | Application public key (for Interactions Webhook signature verification) |
| `application_id` | string | | — | Application ID (required for Slash Command replies) |
| `require_mention` | bool | | `false` | Whether @mention is required in server channels |
| `allowed_guilds` | list | | `[]` (all servers) | Allowlist of server IDs to respond to |
| `reply_in_thread` | bool | | `false` | Whether to reply in a Thread |
| `session_mode` | string | | `shared` | `shared` or `per_user` |
| `max_steps` | int | | `30` | Maximum steps per task |
| `token_budget` | int | | `100000` | Token limit per task |
| `workspace` | string | | `auto` | File workspace mode |

---

## Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Text messages | ✅ | Discord Markdown format |
| Image/file reception | ✅ | Auto-downloads attachments |
| Streaming edits (typewriter effect) | ✅ | edit_interval = 0.8s, uses message.edit() |
| Emoji Reaction progress | ✅ | 🤔 Processing → ✅ Done / ❌ Failed |
| Thread replies | ✅ | Creates a Thread when `reply_in_thread: true` |
| Embed messages | ✅ | Send structured Embeds via `send_embed()` |
| Slash Commands (Interactions) | ✅ | Requires `public_key` and `application_id` |
| Message length limit | — | 2,000 characters (auto-split and continued if exceeded) |

---

## Getting the Server ID (Guild ID)

1. In Discord, go to **User Settings** → **Advanced** → enable **Developer Mode**
2. Right-click the target server icon → **Copy Server ID**

The same method can be used to get a Channel ID.

---

## Thread Replies

When `reply_in_thread: true`, the bot creates a Thread for each user message and replies within it. This is suitable for avoiding channel flooding.

Each user's Thread is independent, providing natural `per_user` session isolation.

---

## Embed Messages

Embeds can be used to send structured rich-text messages:

```python
await channel.send_embed(
    target,
    title="Task Complete",
    description="Code generated and tests passed",
    color=0x00ff00,     # Green
    fields=[
        {"name": "File", "value": "main.py", "inline": True},
        {"name": "Tests", "value": "8/8 passed", "inline": True},
    ]
)
```

---

## Troubleshooting

### Bot login fails (LoginFailure)

- Confirm that `bot_token` is correct and has not been reset
- The Bot Token and Client Secret are different; be careful not to confuse them

### Bot receives empty message content

- You must enable **MESSAGE CONTENT INTENT** in the Developer Portal
- Confirm the application has been re-invited (permission changes require re-authorization)

### Streaming edit error (Missing Permissions)

- Confirm the bot has **Manage Messages** permission (required to edit its own messages)

### Interactions Webhook verification fails (401)

- Confirm that `public_key` matches the one in General Information
- Discord requires HTTPS with a valid certificate

### Messages are truncated

Discord's single message limit is 2,000 characters. Gateway automatically splits and continues messages when approaching this limit, but a single code block that exceeds the limit cannot be split (it will be displayed truncated).

---

## Complete Configuration Example

```yaml
channels:
  discord:
    enabled: true
    channel_type: discord
    bot_token: "MTxxxxxxxxxxxxxxxxxx.Gxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    application_id: "1234567890123456789"
    allowed_guilds:
      - "9876543210987654321"
    require_mention: false
    reply_in_thread: false
    session_mode: shared
    max_steps: 30
    token_budget: 80000
```
