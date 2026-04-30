# DingTalk Channel Configuration Guide

---

## Prerequisites

You need a **DingTalk Open Platform** enterprise internal application (requires enterprise administrator privileges or the DingTalk developer sandbox).

---

## Step 1: Create an Enterprise Internal Application

1. Open [DingTalk Open Platform](https://open.dingtalk.com/) → **App Development** → **Enterprise Internal Development** → **Bot**
2. Click **Create App**, fill in the name and description
3. Go to the application details → **Credentials & Basic Info**, and note:
   - **Client ID** (i.e. AppKey)
   - **Client Secret** (i.e. AppSecret)

---

## Step 2: Enable Bot Features

1. Go to the application → **Message Push** → enable **Bot**
2. Set the bot name and avatar
3. Configure the **Message Receive Mode**:
   - Select **Stream Mode** (recommended, no public IP required)
   - Or select HTTP callback (requires a public HTTPS address)

---

## Step 3: Enable Permissions

Go to the application → **Permission Management** and enable the following permissions:

| Permission | Description |
|------------|-------------|
| `qyapi_robot_sendmsg` | Bot sends messages |
| `Contact.User.Read` | Read user information (for username resolution) |

Some permissions require administrator approval.

---

## Step 4: Publish the Application

1. Go to the application → **Version Management & Release** → **Confirm Release**
2. The enterprise administrator installs the application in the **OA backend** → **Workbench**
3. After installation, add the bot to the target group (Group → Group Settings → Bots → Add)

---

## Config File

### Basic Configuration (Stream Mode, Recommended)

```yaml
channels:
  dingtalk:
    enabled: true
    channel_type: dingtalk
    client_id: "dingxxxxxxxxxxxx"
    client_secret: "your_client_secret"
    require_mention: false   # Whether @mention is required in group chats
    max_steps: 30
```

### Enable AI Streaming Cards (Optional)

DingTalk supports AI interactive cards, which enable a streaming typewriter effect (similar to ChatGPT's token-by-token output):

```yaml
channels:
  dingtalk:
    enabled: true
    client_id: "dingxxxxxxxxxxxx"
    client_secret: "your_client_secret"
    card_template_id: "your_card_template_id"   # AI card template ID
    require_mention: false
```

#### Getting the AI Card Template ID

1. Go to [DingTalk Card Platform](https://card.dingtalk.com/)
2. **Create Card** → select **AI Card Template** (or refer to DingTalk's AI card sample templates)
3. After publishing, copy the **Template ID** from the template list

> **Note**: AI card state machine:
> - `PROCESSING` → Card created (when message is received)
> - `INPUTING` → Streaming output in progress (when first token arrives)
> - `FINISHED` → Agent completed
> - `FAILED` → On error

---

## Complete Field Reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `client_id` | string | ✅ | — | App Key (Client ID) |
| `client_secret` | string | ✅ | — | App Secret |
| `card_template_id` | string | | — | AI streaming card template ID; omit to send plain text messages |
| `msg_key` | string | | `sampleMarkdown` | Open API message template type |
| `require_mention` | bool | | `false` | Whether @mention is required in group chats |
| `mention_patterns` | list | | `[]` | Custom trigger regex patterns (e.g. `["AI.*", "bot"]`) |
| `max_steps` | int | | `30` | Maximum steps per task |
| `token_budget` | int | | `100000` | Token limit per task |
| `workspace` | string | | `auto` | File workspace mode |

---

## Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Text messages | ✅ | Markdown rendering |
| Image/file reception | ✅ | Auto-downloaded |
| Streaming output | ✅ | Requires `card_template_id`; otherwise sent as a single message after completion |
| Message editing | ❌ | DingTalk does not support editing sent messages |
| Emotion Reaction progress | ✅ | 🤔 Processing → 🥳 Done / ☹️ Failed |
| Access Token auto-refresh | ✅ | Automatically refreshed before expiry (TTL 7200s) |
| Webhook persistence | ✅ | session_webhook written to disk; still sendable after restart |
| Open API fallback | ✅ | Falls back to Open API when webhook expires |
| AI card persistence recovery | ✅ | Orphaned cards set to FAILED after restart |
| Message length limit | — | 20,000 characters |

---

## Webhook Persistence

In DingTalk Stream mode, each user message carries a `session_webhook` URL (valid for approximately 1 hour).

Gateway persists this URL to `~/.harnessx/store/dingtalk_webhooks.json`. Even after a process restart, previously recorded webhook URLs can be used to send messages to users (useful for Cron scheduled pushes).

When a webhook expires, it automatically falls back to sending via the DingTalk Open API (requires `client_id` and `client_secret` to be correctly configured).

---

## Trigger Patterns

By default (`require_mention: false`), all messages sent to the bot in a group will trigger it.

To customize trigger conditions:

```yaml
dingtalk:
  require_mention: false
  mention_patterns:
    - "^AI"       # Starts with AI
    - "bot"       # Contains "bot"
    - "@Bot"
```

Any message matching at least one regex will trigger the bot; messages that match none are ignored.

---

## Troubleshooting

### Stream connection fails

- Confirm that `client_id` and `client_secret` are correct
- Confirm that the enterprise has installed the application
- View logs: `hx-gateway logs -n 100 | grep dingtalk`

### Bot cannot send messages to a group

- Confirm the bot has been added to the target group
- The session_webhook may have expired (> 1 hour without interaction); the user must send another message to trigger a new one
- Open API fallback requires the enterprise application to have the `qyapi_robot_sendmsg` permission

### AI card is not displayed

- Confirm `card_template_id` is correct and the template has been published
- Check whether the enterprise has enabled the AI card feature permission

---

## Complete Configuration Example

```yaml
channels:
  dingtalk:
    enabled: true
    channel_type: dingtalk
    client_id: "dingxxxxxxxxxxxxxxxx"
    client_secret: "your_client_secret_here"
    card_template_id: "your_ai_card_template_id"
    require_mention: false
    max_steps: 30
    token_budget: 80000
```
