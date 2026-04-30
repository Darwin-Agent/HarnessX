# Discord Channel 配置指南

---

## 前提条件

需要一个 Discord 账号，以及目标服务器（Guild）的**管理员**权限（用于邀请 Bot）。

---

## 第一步：创建 Discord 应用

1. 打开 [Discord Developer Portal](https://discord.com/developers/applications)
2. 点击 **New Application**，填写应用名称（如 `HarnessX`）
3. 进入应用 → **Bot** 标签页
4. 点击 **Add Bot** → 确认

---

## 第二步：配置 Bot 权限和 Intents

### 开启 Privileged Gateway Intents

进入应用 → **Bot** → **Privileged Gateway Intents**，开启：

- **MESSAGE CONTENT INTENT** — 必须开启，用于读取消息内容（2022 年后必须显式启用）
- **SERVER MEMBERS INTENT** — 可选，用于读取成员信息
- **PRESENCE INTENT** — 可选

> **重要**：如果 Bot 所在服务器超过 100 个，需要通过 Discord 的验证（Verified Bot）。

### 获取 Bot Token

1. 进入 **Bot** 标签页 → **Token** → **Reset Token**
2. 复制 Token（格式类似 `MTxxxx.Gxxxx.xxxxxxxx`）
3. 妥善保存，不要泄露（Token 与账号密码等同）

---

## 第三步：邀请 Bot 到服务器

1. 进入应用 → **OAuth2** → **URL Generator**
2. 在 **Scopes** 中勾选：`bot`、`applications.commands`
3. 在 **Bot Permissions** 中勾选：

   | 权限 | 说明 |
   |------|------|
   | Send Messages | 发送消息（必须） |
   | Send Messages in Threads | 在 Thread 中发送（可选） |
   | Read Message History | 读取历史消息 |
   | Add Reactions | 添加 Reaction（进度反馈） |
   | Embed Links | 发送 Embed 消息 |
   | Attach Files | 发送文件（可选） |
   | Manage Messages | 编辑消息（流式输出需要） |
   | Use Slash Commands | 使用斜线命令 |

4. 复制生成的 OAuth2 URL，在浏览器中打开，选择目标服务器，完成邀请

---

## 第四步：获取 Application ID（可选）

如需支持 Slash Commands 交互（`/help` 等），需要 **Application ID**：

进入应用 → **General Information** → 复制 **Application ID**

---

## 配置文件

### 基础配置（Gateway Bot 模式）

```yaml
channels:
  discord:
    enabled: true
    channel_type: discord
    bot_token: "MTxxxxxxxxxxxxxxxxxx.Gxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    require_mention: false   # 是否需要 @机器人（DM 始终响应）
```

### 带权限控制的配置

```yaml
channels:
  discord:
    enabled: true
    bot_token: "MTxxxxxxxxxxxxxxxxxx.Gxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    application_id: "1234567890123456789"  # 斜线命令需要
    allowed_guilds:
      - "1234567890123456789"   # 服务器 ID，空=允许所有服务器
    require_mention: false
    reply_in_thread: false
    max_steps: 30
```

### 启用 Interactions（Slash Commands）

如需支持来自 Discord Interactions 的 Slash Commands（通过 Webhook 而非 Gateway Bot）：

```yaml
channels:
  discord:
    enabled: true
    bot_token: "MTxxxxxxxxxxxxxxxxxx..."
    application_id: "1234567890123456789"
    public_key: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

> `public_key` 在 **General Information** → **Public Key** 中获取，用于验证 Interactions Webhook 签名。

---

## 配置字段完整参考

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `bot_token` | string | ✅ | — | Bot Token（Bot 标签页获取） |
| `public_key` | string | Interactions 必填 | — | 应用公钥（用于 Interactions Webhook 签名验证） |
| `application_id` | string | | — | 应用 ID（Slash Commands 回复必填） |
| `require_mention` | bool | | `false` | 服务器频道中是否需要 @机器人 |
| `allowed_guilds` | list | | `[]`（所有服务器） | 限制响应的服务器 ID 白名单 |
| `reply_in_thread` | bool | | `false` | 是否在 Thread 中回复 |
| `session_mode` | string | | `shared` | `shared` 或 `per_user` |
| `max_steps` | int | | `30` | 每次任务最大步骤数 |
| `token_budget` | int | | `100000` | 每次任务 token 上限 |
| `workspace` | string | | `auto` | 文件工作区模式 |

---

## 功能特性

| 功能 | 支持 | 说明 |
|------|------|------|
| 文本消息 | ✅ | Discord Markdown 格式 |
| 图片/文件接收 | ✅ | 自动下载附件 |
| 流式编辑（打字机效果） | ✅ | edit_interval = 0.8s，使用 message.edit() |
| Emoji Reaction 进度 | ✅ | 🤔 处理中 → ✅ 完成 / ❌ 失败 |
| Thread 回复 | ✅ | `reply_in_thread: true` 时创建 Thread |
| Embed 消息 | ✅ | `send_embed()` 发送结构化 Embed |
| Slash Commands（Interactions） | ✅ | 需配置 `public_key` 和 `application_id` |
| 消息长度上限 | — | 2,000 字符（超出自动截断并续发） |

---

## 获取服务器 ID（Guild ID）

1. 在 Discord 中，**用户设置** → **高级** → 开启**开发者模式**
2. 右键点击目标服务器图标 → **复制服务器 ID**

类似方法可获取频道 ID（Channel ID）。

---

## Thread 回复说明

`reply_in_thread: true` 时，Bot 会为每条用户消息创建一个 Thread，Bot 的回复在 Thread 内。适合避免频道刷屏。

不同用户的 Thread 相互独立，天然实现 `per_user` 会话隔离。

---

## Embed 消息

Embed 可用于发送结构化的富文本消息：

```python
await channel.send_embed(
    target,
    title="任务完成",
    description="已生成代码并通过测试",
    color=0x00ff00,     # 绿色
    fields=[
        {"name": "文件", "value": "main.py", "inline": True},
        {"name": "测试", "value": "8/8 通过", "inline": True},
    ]
)
```

---

## 常见问题

### Bot 登录失败（LoginFailure）

- 确认 `bot_token` 正确且未被 Reset
- Bot Token 与 Client Secret 不同，注意区分

### Bot 收不到消息内容（消息内容为空）

- 必须在 Developer Portal 中开启 **MESSAGE CONTENT INTENT**
- 确认应用重新邀请（权限变更需要重新授权）

### 流式编辑报错（Missing Permissions）

- 确认 Bot 有 **Manage Messages** 权限（用于编辑自己的消息）

### Interactions Webhook 验证失败（401）

- 确认 `public_key` 与 General Information 中一致
- Discord 要求 HTTPS 且证书有效

### 消息被截断

Discord 单条消息上限 2,000 字符。Gateway 会自动在接近上限时分割并续发，但单个代码块超过上限时无法分割（显示截断）。

---

## 完整配置示例

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
