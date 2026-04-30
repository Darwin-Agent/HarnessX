# Slack Channel 配置指南

---

## 前提条件

需要一个 Slack 工作区（Workspace）的管理员权限，用于安装应用。

---

## 第一步：创建 Slack App

1. 打开 [Slack API 控制台](https://api.slack.com/apps) → **Create New App**
2. 选择 **From scratch**
3. 填写 App Name（如 `HarnessX`），选择目标 Workspace
4. 点击 **Create App**

---

## 第二步：开启 Socket Mode（推荐）

Socket Mode 无需公网 IP，Slack 通过 WebSocket 主动推送事件。

1. 进入应用 → **Settings** → **Socket Mode** → 开启 **Enable Socket Mode**
2. 填写 Token Name（如 `gateway-token`）
3. 点击 **Generate** → 复制生成的 **App-Level Token**（`xapp-` 开头）

---

## 第三步：开通 OAuth 权限

进入应用 → **OAuth & Permissions** → **Scopes** → **Bot Token Scopes**，添加以下权限：

| Scope | 用途 |
|-------|------|
| `app_mentions:read` | 接收 @机器人 消息 |
| `channels:history` | 读取公开频道消息历史 |
| `channels:read` | 读取频道信息 |
| `chat:write` | 发送消息 |
| `files:read` | 读取文件（图片等附件） |
| `files:write` | 上传文件 |
| `groups:history` | 读取私有频道消息（可选） |
| `groups:read` | 读取私有频道信息（可选） |
| `im:history` | 读取 DM 消息 |
| `im:read` | 读取 DM 信息 |
| `im:write` | 发送 DM |
| `mpim:history` | 读取多人 DM（可选） |
| `reactions:read` | 读取 Reaction |
| `reactions:write` | 添加/删除 Reaction |
| `users:read` | 读取用户信息 |

---

## 第四步：订阅事件

进入应用 → **Event Subscriptions** → **Enable Events**：

- 在 **Subscribe to bot events** 中添加：
  - `message.channels` — 公开频道消息
  - `message.groups` — 私有频道消息（可选）
  - `message.im` — DM 消息
  - `message.mpim` — 多人 DM（可选）
  - `app_mention` — @机器人 消息

---

## 第五步：安装应用

1. 进入应用 → **OAuth & Permissions** → **Install to Workspace** → 授权
2. 安装完成后复制 **Bot User OAuth Token**（`xoxb-` 开头）

---

## 第六步：将 Bot 添加到频道

在 Slack 频道中，使用 `/invite @HarnessX` 命令邀请 Bot 加入频道。

---

## 配置文件

```yaml
channels:
  slack:
    enabled: true
    channel_type: slack
    bot_token: "xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx"
    app_token: "xapp-1-xxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    require_mention: true    # 频道中是否需要 @机器人
    reply_in_thread: true    # 是否在 Thread 中回复（推荐，保持频道整洁）
    reply_broadcast: false   # Thread 回复是否同时发到频道（广播）
```

---

## 配置字段完整参考

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `bot_token` | string | ✅ | — | Bot User OAuth Token（`xoxb-` 开头） |
| `app_token` | string | ✅ | — | App-Level Token，Socket Mode 专用（`xapp-` 开头） |
| `signing_secret` | string | | — | Signing Secret（Webhook 模式校验用，Socket Mode 不需要） |
| `require_mention` | bool | | `true` | 频道中是否需要 @机器人 才触发（DM 始终响应） |
| `reply_in_thread` | bool | | `true` | 是否在消息 Thread 中回复 |
| `reply_broadcast` | bool | | `false` | Thread 回复是否同时广播到频道 |
| `session_mode` | string | | `shared` | `shared` 或 `per_user` |
| `max_steps` | int | | `30` | 每次任务最大步骤数 |
| `token_budget` | int | | `100000` | 每次任务 token 上限 |
| `workspace` | string | | `auto` | 文件工作区模式 |

---

## 功能特性

| 功能 | 支持 | 说明 |
|------|------|------|
| 文本消息 | ✅ | Slack mrkdwn 格式 |
| 图片/文件接收 | ✅ | 自动下载，传递给 Agent |
| 流式编辑（打字机效果） | ✅ | edit_interval = 0.8s |
| Emoji Reaction 进度 | ✅ | ⏳ 处理中 → ✅ 完成 / ❌ 失败 |
| Thread 回复 | ✅ | `reply_in_thread: true`（推荐） |
| Block Kit 消息 | ✅ | `send_blocks()` 发送富文本卡片 |
| 消息长度上限 | — | 40,000 字符 |

---

## Thread 回复说明

`reply_in_thread: true`（默认）时，Bot 的所有回复都在原消息的 Thread 中，保持频道整洁，适合大型团队使用。

`reply_broadcast: true` 可以让 Thread 回复同时出现在频道主时间线，适合重要通知。

DM 会话不受此设置影响，始终直接回复。

---

## Block Kit 支持

Gateway 支持通过 `send_blocks()` 方法发送 Slack Block Kit 格式的富文本消息，适用于代码块、表格等结构化输出。

长回复中包含代码块时，Gateway 会自动将其转为 Block Kit Section + Code Block 格式，在 Slack 中正确渲染。

---

## 常见问题

### Socket Mode 连接失败

- 确认 `app_token` 是 App-Level Token（`xapp-` 开头），而非 Bot Token
- 确认 Socket Mode 已在控制台启用
- 查看日志：`hx-gateway logs -n 100 | grep slack`

### Bot 收不到频道消息

- 确认 Bot 已通过 `/invite` 加入频道
- 检查 **Event Subscriptions** 是否开启了对应事件
- `require_mention: true` 时，确认消息中 @了机器人

### `missing_scope` 错误

对应缺少某个权限 scope，在 **OAuth & Permissions** 中补充，然后重新安装（reinstall）应用。

### 速率限制（rate_limit_exceeded）

Slack Web API 限制：
- `chat.update` — 每频道每秒最多 5 次（流式编辑）
- `chat.postMessage` — 每频道每分钟约 100 条

如触发限制，适当增加 `stream_edit_interval` 或减少频繁触发。Gateway 内置退避重试。

---

## 完整配置示例

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
