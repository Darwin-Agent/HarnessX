# HarnessX IM Gateway

Connect HarnessX agents to instant messaging platforms. One bot per channel, all managed from a single web console.

**Supported platforms:** Feishu · Telegram · Slack · Discord · DingTalk

---

## How it works

```
IM Platform  ──webhook/WS──▶  Gateway  ──▶  Harness.run()  ──▶  reply
```

The gateway runs a FastAPI server that receives messages from IM platforms, routes them to a HarnessX agent, and streams replies back. A built-in React console lets you configure channels, view sessions, and chat directly in the browser.

---

## Quick Start

### 1. Install

**Recommended — use the HarnessX one-click installer** (installs harnessx + gateway + console in one go):

```bash
curl -sSf https://raw.githubusercontent.com/Darwin-Agent/HarnessX/main/scripts/install.sh | bash
```

The installer will ask whether to install the IM Gateway and which channel extras you need.
For a fully non-interactive install:

```bash
curl -sSf https://raw.githubusercontent.com/Darwin-Agent/HarnessX/main/scripts/install.sh | bash -s -- --all
```

<details>
<summary>Manual install (existing HarnessX environment)</summary>

```bash
# From the repo root — install only the platforms you need
pip install -e "gateway/[feishu]"
pip install -e "gateway/[telegram]"
pip install -e "gateway/[slack]"
pip install -e "gateway/[discord]"
pip install -e "gateway/[dingtalk]"
# Or install all channel extras at once
pip install -e "gateway/[all]"
```

Requires Python ≥ 3.11 and a HarnessX install (`pip install -e .` from repo root).

**Build the console (one-time, requires Node.js ≥ 18):**

```bash
cd gateway/console && npm install && npm run build && cd ../..
```

Or use the helper script:

```bash
bash scripts/build-frontend.sh --gateway
```

The built assets are served automatically by the gateway at `/console/`.

</details>

### 2. Configure

Create `~/.harnessx/gateway.yaml`:

```yaml
gateway:
  agent_id: gateway       # agent identity (used for workspace isolation)
  host: 0.0.0.0
  port: 8080

channels:
  telegram:
    enabled: true
    channel_type: telegram  # platform type; only needed when instance name differs
    bot_token: "your-bot-token"
    allowed_users: []       # empty = open to everyone

  feishu:
    enabled: false
    channel_type: feishu
    app_id: "cli_xxx"
    app_secret: "xxx"
    verification_token: "xxx"
    encrypt_key: ""
```

See [Channel Configuration](#channel-configuration) below for all options.

### 3. Set up a model

Create `~/.harnessx/model_config.yaml` (same format as the CLI):

```yaml
main:
  provider: litellm
  model: claude-sonnet-4-6
```

Or set the environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Start

```bash
hx-gateway start
```

Open the console at **http://localhost:8080/console/**

```
hx-gateway status      # check if running
hx-gateway logs        # tail logs
hx-gateway stop        # graceful shutdown
hx-gateway restart     # restart after config changes
```

---

## Console

The web console is available at `/console/` when the gateway is running. From there you can:

- **Channels** — add/remove channels, view connection status, edit config, restart individual bots
- **Settings → Models** — configure which LLM the agent uses
- **Settings → Environment** — manage workspaces, tools, skills, and plugins
- **Chat** — talk to the agent directly in the browser (via the built-in `web_ui` channel)

### Development mode (hot-reload)

```bash
# Terminal 1: run the gateway backend
hx-gateway _serve --log-level DEBUG

# Terminal 2: run the console dev server
cd gateway/console
npm run dev
```

The Vite dev server proxies API requests to the gateway and hot-reloads on code changes.

---

## Channel Configuration

All channels share these common fields:

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Whether to start this channel |
| `channel_type` | _(same as key)_ | Platform type (`telegram`, `feishu`, etc.) — set this when your instance name differs from the platform type |
| `allowed_users` | `[]` | Sender IDs allowed to use the bot; empty = everyone |
| `require_mention` | `true` | Group/channel messages: require @mention before responding |
| `reply_in_thread` | `false` | Reply in a new thread instead of the main channel |
| `auth_mode` | — | Set to `"pairing"` to require users to `/pair <code>` before chatting |
| `session_mode` | `"shared"` | Group session scope: `"shared"` (one session per group) or `"per_user"` |
| `workspace` | — | Set to `"auto"` to give the channel its own file workspace |

### Telegram

```yaml
telegram:
  enabled: true
  bot_token: "123456:ABC-..."
  allowed_users: ["@alice", "123456789"]
```

[Create a bot](https://t.me/BotFather) with `/newbot`. No extra permissions needed.

### Feishu (Lark)

```yaml
feishu:
  enabled: true
  app_id: "cli_xxx"
  app_secret: "xxx"
  verification_token: "xxx"   # from Event Subscriptions page
  encrypt_key: ""              # optional
```

In the Feishu developer console: enable **Bot** capability, subscribe to `im.message.receive_v1`, set the webhook URL to `https://your-host/gateway/webhook/feishu`.

### Slack

```yaml
slack:
  enabled: true
  bot_token: "xoxb-..."
  signing_secret: "xxx"
  app_token: "xapp-..."       # for Socket Mode (no public URL needed)
```

### Discord

```yaml
discord:
  enabled: true
  bot_token: "Bot xxx"
  allowed_guilds: []          # server IDs to respond in; empty = all
  require_mention: false
```

Enable **Message Content Intent** in the Discord Developer Portal → Bot → Privileged Gateway Intents.

### DingTalk

```yaml
dingtalk:
  enabled: true
  client_id: "xxx"
  client_secret: "xxx"
```

---

## Workspace isolation

Each channel gets an isolated workspace under:

```
~/.harnessx/im-workspaces/{agent_id}/{channel_name}/
```

Prompt templates (for example `AGENTS.md`, `SOUL.md`, `PROFILE.md`) are copied into each workspace on first start and can be customized per channel. The console's **Workspace** tab provides a file browser.

---

## Architecture

```
gateway/
├── main.py              CLI entry point (hx-gateway start/stop/logs/status)
├── server.py            FastAPI app + all REST endpoints
├── channels/
│   ├── telegram/        python-telegram-bot integration
│   ├── feishu/          lark-oapi integration
│   ├── slack/           slack-bolt integration
│   ├── discord_/        discord.py integration
│   └── dingtalk/        dingtalk-stream integration
├── core/
│   ├── base_channel.py  BaseChannel ABC — implement this to add a new platform
│   ├── dispatch.py      ChannelDispatcher — routes events to Harness.run()
│   ├── auth.py          PairingAuth — optional pairing-code authorization
│   └── processors/      IMSystemProcessor, IMUserContextProcessor
└── console/             React + Vite frontend (TypeScript)
```

### Adding a new platform

1. Create `channels/{name}/channel.py` with a class extending `BaseChannel`
2. Implement `_connect`, `_listen`, `stop`, `send`, `send_stream`, `send_typing`
3. Call `register_builtin(YourChannel)` at the bottom of the file
4. Add optional dependencies to `pyproject.toml`

No other files need to change.
