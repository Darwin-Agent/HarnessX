# Telegram Channel 配置指南

---

## 前提条件

需要一个 Telegram 账号，以及用于创建 Bot 的 **BotFather**（Telegram 官方 Bot 管理工具）。

---

## 第一步：创建 Bot

1. 在 Telegram 中搜索 `@BotFather` 并开启对话
2. 发送 `/newbot`
3. 输入 Bot 的**显示名称**（如 `HarnessX Assistant`）
4. 输入 Bot 的**用户名**（必须以 `bot` 结尾，如 `harnessx_bot`）
5. BotFather 返回 **Bot Token**，格式为：
   ```
   123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
   ```
   妥善保存，不要泄露。

---

## 第二步：配置 Bot 权限（可选）

对 BotFather 发送以下命令来配置 Bot：

```
/setprivacy — 设置隐私模式
  选择 DISABLED 可让 Bot 接收群组中所有消息（不仅是 /命令）
  选择 ENABLED（默认）仅接收 /命令 和 @mention 消息

/setjoingroups — 允许或禁止将 Bot 加入群组
/setcommands — 设置 Bot 命令列表（显示在输入框底部）
```

推荐设置命令列表：

```
/setcommands
help - 显示帮助信息
reset - 清除对话历史
cancel - 取消当前任务
```

---

## 第三步：获取用户 ID（可选，用于白名单）

如需限制只有特定用户能使用 Bot：

1. 在 Telegram 中搜索 `@userinfobot`
2. 发送任意消息，它会返回你的 **user ID**（纯数字）
3. 将此 ID 填入 `allowed_users` 列表

---

## 第四步：将 Bot 添加到群组（可选）

1. 在 Telegram 群组设置中 → **添加成员** → 搜索你的 Bot 用户名
2. 添加后，确认 Bot 有**发送消息**权限
3. 如果隐私模式为 ENABLED，需要在消息中 @机器人 才会触发（或设置 `require_mention: false` 并关闭隐私模式）

---

## 配置文件

### 私聊模式（最简配置）

```yaml
channels:
  telegram:
    enabled: true
    channel_type: telegram
    bot_token: "123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
```

### 带白名单的个人助理

```yaml
channels:
  telegram:
    enabled: true
    bot_token: "123456789:ABC-DEF1234..."
    allowed_users:
      - "123456789"    # 你的 Telegram user_id
    max_steps: 20
```

### 群组机器人

```yaml
channels:
  telegram:
    enabled: true
    bot_token: "123456789:ABC-DEF1234..."
    require_mention: false   # 群组中不需要 @机器人（需关闭隐私模式）
    reply_in_thread: true    # 在 thread 中回复（减少刷屏）
    session_mode: per_user   # 每个用户独立会话
```

---

## 配置字段完整参考

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `bot_token` | string | ✅ | — | BotFather 生成的 Bot Token |
| `allowed_users` | list | | `[]`（所有人） | 允许使用 Bot 的 user_id 白名单，空=所有人 |
| `require_mention` | bool | | `false` | 群组中是否需要 @机器人 才触发 |
| `reply_in_thread` | bool | | `false` | 是否在 Thread 内回复 |
| `webhook_secret_token` | string | | — | Webhook 模式的 secret token（Polling 模式不需要） |
| `session_mode` | string | | `shared` | `shared` 或 `per_user` |
| `max_steps` | int | | `20` | 每次任务最大步骤数 |
| `token_budget` | int | | `100000` | 每次任务 token 上限 |
| `workspace` | string | | `auto` | 文件工作区模式 |

---

## 功能特性

| 功能 | 支持 | 说明 |
|------|------|------|
| 文本消息 | ✅ | MarkdownV2 格式 |
| 图片/视频/文件接收 | ✅ | 自动下载，传递给 Agent |
| 语音消息接收 | ✅ | 下载 ogg 文件 |
| 流式编辑（打字机效果） | ✅ | edit_interval = 1.0s |
| Emoji Reaction 进度 | ✅ | 👀 处理中 → ✅ 完成 / ❌ 失败 |
| Thread 回复 | ✅ | `reply_in_thread: true` |
| Inline Keyboard | ✅ | 通过 `send_with_keyboard()` 发送按钮 |
| Callback Query | ✅ | 按钮点击转为 MessageEvent 入队 |
| 消息长度上限 | — | 4,096 字符 |

---

## 接入模式说明

Gateway 使用 **Long Polling** 模式接收消息，无需公网 IP 或 Webhook 配置，适合本地和内网部署。

> Telegram Bot API 的 Long Polling 每次轮询有效期 ≤ 100 条更新，Gateway 持续轮询保证消息不丢失。

---

## 常见问题

### Bot 不响应消息

1. 确认 `bot_token` 正确（包含完整的 `:` 之后的部分）
2. 群组消息：检查 Bot 隐私模式，ENABLED 状态下只接收 /命令 和 @提及
3. 查看日志：`hx-gateway logs -n 100 | grep telegram`

### 消息发送失败（429 Too Many Requests）

Telegram 有严格的速率限制：
- 每秒最多 30 条消息（全局）
- 对同一聊天每分钟最多 20 条

Gateway 内置 Token Bucket 限速，自动处理。如仍触发，适当增加 `edit_interval`。

### MarkdownV2 格式错误（消息中有特殊字符）

Telegram MarkdownV2 需要转义 `_*[]()~>#+-=|{}.!` 等字符。Gateway 的 formatter 会自动转义，如出现问题检查 Agent 输出是否包含未转义的特殊字符。

### 群组中 `allowed_users` 无效

`allowed_users` 使用 `sender_id`（string 类型的数字），确保填写的是用户的数字 ID，不是用户名。

---

## 完整配置示例

```yaml
channels:
  telegram:
    enabled: true
    channel_type: telegram
    bot_token: "123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    allowed_users:
      - "123456789"
      - "987654321"
    require_mention: false
    reply_in_thread: false
    session_mode: per_user
    max_steps: 20
    token_budget: 80000
```
