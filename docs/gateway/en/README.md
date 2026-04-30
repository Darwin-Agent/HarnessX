# HarnessX Gateway

HarnessX Gateway is a multi-platform IM message gateway that routes messages from IM platforms such as Feishu, DingTalk, Telegram, Slack, and Discord to the HarnessX Agent, and streams the Agent's responses back to each platform in real time.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          hx-gateway process                          │
│                                                                     │
│  ┌────────────┐   MessageEvent    ┌─────────────────────────────┐   │
│  │  Channel   │ ──────────────→  │     ChannelDispatcher        │   │
│  │  (per IM   │                  │  ┌──────────────────────┐    │   │
│  │  platform) │ ←─────────────── │  │  PriorityQueue       │    │   │
│  └────────────┘    send_stream   │  │  (cmd=0, msg=1)      │    │   │
│                                  │  └──────────────────────┘    │   │
│  ┌────────────┐                  │  ┌──────────────────────┐    │   │
│  │  FastAPI   │                  │  │  SessionStore        │    │   │
│  │  Server    │                  │  │  (session per conv)  │    │   │
│  └────────────┘                  │  └──────────────────────┘    │   │
│       │                          │  ┌──────────────────────┐    │   │
│       │ REST API                 │  │  Harness.run()       │    │   │
│       │ /gateway/*               │  │  (per session)       │    │   │
│       │ /console/                │  └──────────────────────┘    │   │
│                                  └─────────────────────────────┘   │
│  ┌────────────┐                                                     │
│  │ CronManager│ — scheduled jobs that periodically prompt the Agent │
│  └────────────┘                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `BaseChannel` | `gateway/core/base_channel.py` | Connection management, reconnection, deduplication, debounce, rate limiting |
| `ChannelDispatcher` | `gateway/core/dispatch.py` | Message priority queue, session management, Harness scheduling |
| `SessionStore` | `gateway/core/session_store.py` | Conversation state persistence, idle timeout GC |
| `CronManager` | `gateway/core/cron.py` | 5-field cron / `every:` scheduled jobs |
| `IMProgressProcessor` | `gateway/core/im_stream.py` | Streaming progress injection (tool-call progress cards) |
| FastAPI Server | `gateway/server.py` | REST API + Web Console + Webhook entry point |

### Message Processing Flow

```
1. Channel._connect()      — Establish WebSocket / register Webhook / start LongPoll
2. Channel._listen()       — Receive platform push, call _enqueue(event)
3. _enqueue()              — Dedup check → 50ms debounce merge → dispatcher.enqueue()
4. Dispatcher              — PriorityQueue(cmd=0, msg=1) → serial consumption per session
5. harness.run(task)       — Invoke HarnessX Agent, stream tokens into asyncio.Queue
6. channel.send_stream()   — Consume token queue, update/edit IM message in real time
```

### Key Design Decisions

- **Serial per session**: Messages within the same conversation (session_id) are processed serially; different conversations run in parallel.
- **Priority queue**: `/cancel`, `/reset`, and other commands have priority 0; regular messages have priority 1, ensuring commands take effect immediately.
- **Streaming edits**: Platforms that support message editing (Feishu, Telegram, Slack, Discord) achieve a typewriter effect via periodic edits; platforms that do not support editing (DingTalk) send new messages instead.
- **Token Bucket rate limiting**: `OutboundRateLimiter` prevents exceeding platform API rate limits, with built-in jitter to prevent thundering-herd cascades.
- **Persistent deduplication**: `~/.harnessx/dedup/{channel}.json` prevents duplicate message processing across restarts (TTL 10 minutes).

---

## Installation

```bash
pip install harnessx[gateway]
# Or install platform-specific dependencies manually:
pip install lark-oapi           # Feishu
pip install dingtalk-stream     # DingTalk
pip install python-telegram-bot # Telegram
pip install slack-bolt          # Slack
pip install "discord.py>=2.0"   # Discord
```

---

## Quick Start

```bash
# 1. Create a config file (see next section)
mkdir -p ~/.harnessx
vi ~/.harnessx/gateway.yaml

# 2. Start (background daemon)
hx-gateway start

# 3. Check status
hx-gateway status

# 4. View logs
hx-gateway logs -f

# 5. Stop
hx-gateway stop
```

After starting, open **`http://localhost:8080/console/`** to access the web management console.

---

## gateway.yaml Configuration Reference

```yaml
# ~/.harnessx/gateway.yaml

# ── Gateway server settings ────────────────────────────────────────────────
gateway:
  host: "0.0.0.0"        # Listen address (default 0.0.0.0)
  port: 8080             # Listen port (default 8080)
  agent_id: "gateway"    # Global agent_id for session isolation

# ── Default settings (inherited by all channels) ──────────────────────────
default:
  workspace: "auto"      # "auto" = create an isolated file workspace per channel
  max_steps: 30          # Maximum steps per run()
  token_budget: 100000   # Token limit per run()

# ── Channel definitions ───────────────────────────────────────────────────
channels:
  # Each key is a channel instance name (customizable, used in logs/API routes)
  feishu:
    enabled: true
    channel_type: feishu   # Optional, defaults to the key name
    app_id: "cli_xxxx"
    app_secret: "xxxx"
    mode: websocket        # websocket | webhook (see Feishu docs)
    require_mention: true  # Whether @mention is required in group chats
    reply_in_thread: false # Whether to reply within a thread

  dingtalk:
    enabled: false
    channel_type: dingtalk
    client_id: "dingxxxx"
    client_secret: "xxxx"
    card_template_id: ""   # Optional: AI streaming card template ID

  telegram_bot:
    enabled: false
    channel_type: telegram
    bot_token: "123456:ABC-DEF..."
    allowed_users:
      - "123456789"        # Telegram user_id allowlist (empty = everyone)

  slack:
    enabled: false
    channel_type: slack
    bot_token: "xoxb-..."
    app_token: "xapp-..."  # Socket Mode only
    reply_in_thread: true

  discord:
    enabled: false
    channel_type: discord
    bot_token: "MTxxxx..."

# ── Heartbeat job (optional) ──────────────────────────────────────────────
heartbeat:
  enabled: true
  cron: "0 9 * * 1-5"    # Every weekday at 9:00 (UTC)
  prompt: "Please summarize today's to-do items"
  channel: feishu
  chat_id: "oc_xxxx"      # Feishu chat_id (group or DM)
```

### Model Configuration

Gateway uses the same model configuration priority as the CLI:

1. `{agent_home}/model_config.yaml`
2. `~/.harnessx/model_config.yaml`
3. Environment variable `HARNESSX_MODEL`
4. Default `gpt-4o`

```yaml
# ~/.harnessx/model_config.yaml
main:
  provider: litellm
  model: claude-sonnet-4-6
```

---

## REST API

Once started, Gateway exposes the following API at `http://localhost:8080`:

### Basic

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/gateway/health` | Health check |
| `GET` | `/gateway/config` | Get gateway global config |
| `PUT` | `/gateway/config` | Update gateway global config |

### Channel Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/gateway/channels` | List all channels and their status |
| `GET` | `/gateway/channels/{name}/config` | Get config for a specific channel |
| `PUT` | `/gateway/channels/{name}/config` | Update channel config (hot reload) |
| `GET` | `/gateway/channels/{name}/status` | Get channel connection status |
| `POST` | `/gateway/channels/{name}/restart` | Restart a specific channel |
| `POST` | `/gateway/channels/{name}/reset_session` | Reset sessions for a specific channel |
| `POST` | `/gateway/channels/create` | Create a new channel |
| `DELETE` | `/gateway/channels/{name}` | Delete a channel |
| `GET` | `/gateway/channel-types` | List registered channel types and their schemas |

### Webhook

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/gateway/webhook/{channel_name}` | Unified webhook entry point (platform callbacks) |
| `GET` | `/gateway/webhook/{channel_name}` | Verification endpoint for some platforms (e.g. WeChat) |

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/gateway/sessions` | List active sessions (supports `?channel=` filter) |

### Scheduled Jobs (Cron)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/gateway/cron/jobs` | List all scheduled jobs |
| `POST` | `/gateway/cron/jobs` | Create a scheduled job |
| `GET` | `/gateway/cron/jobs/{job_id}` | Get scheduled job details |
| `PUT` | `/gateway/cron/jobs/{job_id}` | Update a scheduled job |
| `DELETE` | `/gateway/cron/jobs/{job_id}` | Delete a scheduled job |
| `POST` | `/gateway/cron/jobs/{job_id}/run` | Trigger immediate execution |
| `POST` | `/gateway/cron/jobs/{job_id}/pause` | Pause a job |
| `POST` | `/gateway/cron/jobs/{job_id}/resume` | Resume a job |

### Heartbeat

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/gateway/heartbeat` | Get heartbeat status |
| `GET` | `/gateway/heartbeat/config` | Get heartbeat config |
| `PUT` | `/gateway/heartbeat/config` | Update heartbeat config |

### Web Console

| Path | Description |
|------|-------------|
| `GET /console/` | Web management console (frontend UI) |

---

## Scheduled Jobs (Cron)

Scheduled jobs are stored in `~/.harnessx/im-workspaces/{agent_id}/cron_jobs.json` and can be managed via API or config file.

### Creating a Scheduled Job

```bash
curl -X POST http://localhost:8080/gateway/cron/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Daily Report",
    "cron": "0 9 * * 1-5",
    "prompt": "Please generate today'\''s work summary",
    "channel": "feishu",
    "chat_id": "oc_xxxx",
    "timeout": 120
  }'
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Job name |
| `cron` | string | 5-field cron expression (e.g. `"0 9 * * 1-5"`) |
| `every` | string | Interval shorthand, mutually exclusive with `cron` (e.g. `"30m"`, `"1h"`, `"2h30m"`) |
| `prompt` | string | Prompt text sent to the Agent |
| `channel` | string | Channel name for the reply (empty = silent run, no IM reply) |
| `chat_id` | string | Platform chat/channel/group ID |
| `session_id` | string | Session ID (default `"cron"`; same session_id shares conversation history) |
| `timeout` | int | Timeout in seconds (default 120) |
| `timezone` | string | Timezone (default UTC, e.g. `"Asia/Shanghai"`) |
| `enabled` | bool | Whether the job is enabled (default true) |

### Cron Expression Examples

```
0 9 * * 1-5    — Every weekday at 9:00
0 */2 * * *    — Every 2 hours
*/30 * * * *   — Every 30 minutes
0 10 * * 1     — Every Monday at 10:00
```

---

## Session Modes

Use `session_mode` to control conversation isolation within a group:

| Mode | Description | Use Case |
|------|-------------|----------|
| `shared` (default) | All members in a group share one session | Team collaboration, public bots |
| `per_user` | Each member has an independent session | Personal assistant, privacy-sensitive scenarios |

DMs always use `per_user` mode and are not affected by this setting.

---

## Built-in Commands

Users on all platforms can send the following commands:

| Command | Description |
|---------|-------------|
| `/help` | Show help information |
| `/reset` | Clear current session history and start a new conversation |
| `/cancel` | Cancel the currently running task (priority queue guarantees immediate effect) |

---

## Runtime Directories

| Path | Contents |
|------|----------|
| `~/.harnessx/gateway.yaml` | Main config file |
| `~/.harnessx/dedup/` | Message deduplication state (per channel, JSON) |
| `~/.harnessx/store/` | Persistent storage (webhook cache, receive_id, etc.) |
| `~/.harnessx/media_cache/` | Media file cache (images, audio, etc.) |
| `~/.harnessx/im-workspaces/{agent_id}/` | Agent workspace root |
| `~/.harnessx/im-workspaces/{agent_id}/{channel}/sessions/` | Session traces (JSONL) |
| `~/.harnessx/im-workspaces/{agent_id}/cron_jobs.json` | Scheduled job persistence |
| `/tmp/hx-gateway/gateway.pid` | Daemon PID |
| `/tmp/hx-gateway/gateway.log` | Daemon log |

---

## Channel Configuration Guides

- [Feishu](channels/feishu.md)
- [DingTalk](channels/dingtalk.md)
- [Telegram](channels/telegram.md)
- [Slack](channels/slack.md)
- [Discord](channels/discord.md)
