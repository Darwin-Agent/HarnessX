# Feishu Channel Configuration Guide

---

## Prerequisites

You need a **Feishu Open Platform** enterprise self-built application. Personal accounts can use the Feishu developer sandbox environment for testing.

---

## Step 1: Create a Self-Built Application

1. Open [Feishu Open Platform](https://open.feishu.cn/app) → **Create Enterprise Self-Built App**
2. Fill in the application name (e.g. `HarnessX Bot`), description, and upload an icon
3. Note the **App ID** and **App Secret** (Home page → Credentials & Basic Info)

---

## Step 2: Enable Permissions

Go to the application → **Permission Management** → **Bulk Import Permissions**, and paste the following JSON to enable all required permissions at once:

```json
{
  "scopes": {
    "tenant": [
      "im:message",
      "im:message.group_msg",
      "im:message.p2p_msg:readonly",
      "im:message.reactions:read",
      "im:resource",
      "im:chat",
      "contact:user.base:readonly"
    ],
    "user": []
  }
}
```

> **Notes**:
> - `im:message` — Send and edit messages
> - `im:message.group_msg` — Receive group messages
> - `im:message.p2p_msg:readonly` — Receive direct messages
> - `im:message.reactions:read` — Use Emoji Reactions (⏳/✅/❌ progress feedback)
> - `im:resource` — Upload/download images and files
> - `im:chat` — Read group information
> - `contact:user.base:readonly` — Read basic user info (for @mention resolution)

After enabling permissions, click **Apply for Release** and wait for administrator approval (not required in the sandbox environment).

---

## Step 3: Choose a Connection Mode

### Mode 1: WebSocket (Recommended)

No public IP required. Feishu servers initiate the connection to your service, making it suitable for local development and intranet deployment.

1. Go to the application → **Event Subscriptions** → **Request URL Verification**, and select "Use persistent connection to receive events"
2. Under **Event Subscriptions**, add the following events:
   - `im.message.receive_v1` — Receive messages (required)
   - `im.chat.member.bot.added_v1` — Bot invited to a group (optional)
3. No `verification_token` is needed; WebSocket mode does not verify signatures

Config file:

```yaml
channels:
  feishu:
    enabled: true
    app_id: "cli_xxxx"
    app_secret: "xxxxxxxxxxxx"
    mode: websocket           # Default is websocket, can be omitted
    require_mention: true     # @mention required in group chats
    reply_in_thread: false
```

### Mode 2: Webhook

Requires a publicly accessible HTTPS address (or use an intranet tunneling tool such as ngrok).

1. Go to the application → **Event Subscriptions** → fill in the request URL:
   ```
   https://your-domain.com/gateway/webhook/feishu
   ```
2. Copy the **Verification Token** shown on the page and enter it in the config
3. If you want to enable AES encryption: record the **Encrypt Key**

Config file:

```yaml
channels:
  feishu:
    enabled: true
    app_id: "cli_xxxx"
    app_secret: "xxxxxxxxxxxx"
    mode: webhook
    verification_token: "xxxx"   # Verification Token from the Event Subscriptions page
    encrypt_key: "xxxx"          # Optional: AES encryption key
    require_mention: true
```

---

## Step 4: Add the Bot to Your Application

Go to the application → **App Features** → **Bot** → enable the Bot feature.

Once done, publish and install the application to your organization, or test directly in the sandbox.

---

## Step 5: Get the Chat ID (Optional, for Cron Push)

If you need to push messages to a specific group or DM via scheduled jobs, you need the `chat_id`.

**Method**: Send any message to the bot and look in the Gateway logs:

```
hx-gateway logs -n 50 | grep chat_id
```

Or query via the Feishu API:

```bash
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://open.feishu.cn/open-apis/im/v1/chats?page_size=20"
```

---

## Complete Field Reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `app_id` | string | ✅ | — | Application App ID (starts with `cli_`) |
| `app_secret` | string | ✅ | — | Application App Secret |
| `mode` | string | | `websocket` | Connection mode: `websocket` or `webhook` |
| `verification_token` | string | Required for Webhook mode | — | Event Subscriptions Verification Token |
| `encrypt_key` | string | | — | AES encryption key (optional in Webhook mode) |
| `require_mention` | bool | | `true` | Whether group messages require @mention to trigger |
| `reply_in_thread` | bool | | `false` | Whether to reply within a Thread to avoid flooding the channel |
| `session_mode` | string | | `shared` | `shared` (group-shared) or `per_user` (per-member) |
| `max_steps` | int | | `30` | Maximum steps per task |
| `token_budget` | int | | `100000` | Token limit per task |
| `allowed_users` | list | | `[]` (everyone) | Allowlist of open_ids; empty = allow everyone |
| `workspace` | string | | `auto` | File workspace mode |

---

## Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Text messages | ✅ | Markdown rendering |
| Image/file reception | ✅ | Auto-downloaded locally and passed to the Agent |
| Streaming edits (typewriter effect) | ✅ | edit_interval = 200ms |
| Emoji Reaction progress | ✅ | ⏳ Processing → ✅ Done / ❌ Failed |
| Thread replies | ✅ | Enabled when `reply_in_thread: true` |
| Webhook AES decryption | ✅ | Automatically decrypted when `encrypt_key` is configured |
| Proactive push (Cron) | ✅ | Requires receive_id to be recorded (automatically recorded after first message) |
| Message length limit | — | 30,000 characters |

---

## Troubleshooting

### Bot is not receiving messages

1. Confirm the application has been published and installed to the organization/team
2. Check that the `im.message.receive_v1` event subscription is enabled
3. For group messages, confirm that the bot was @mentioned when `require_mention: true`
4. View logs: `hx-gateway logs -n 100 | grep feishu`

### WebSocket connection drops frequently

- Check network stability; Gateway has built-in exponential backoff reconnection (5s → 10s → 30s → 60s → 120s)
- Check the error message after `[feishu] disconnected` in the logs

### Webhook signature verification failure

- Confirm that `verification_token` matches what is in the Feishu developer console
- If encryption is enabled, confirm that `encrypt_key` is correct

### Insufficient permissions (403 error)

- Check that the application permissions have been requested and approved
- For enterprise editions, the administrator must approve in the Feishu management console

---

## Complete Configuration Example

```yaml
channels:
  feishu:
    enabled: true
    channel_type: feishu
    app_id: "cli_a1b2c3d4e5f6"
    app_secret: "your_app_secret_here"
    mode: websocket
    require_mention: true
    reply_in_thread: false
    session_mode: shared
    max_steps: 30
    token_budget: 100000
```
