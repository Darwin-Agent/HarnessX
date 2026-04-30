# 飞书（Feishu）Channel 配置指南

---

## 前提条件

需要一个**飞书开放平台**企业自建应用。个人账号可用飞书开发者沙盒环境测试。

---

## 第一步：创建自建应用

1. 打开 [飞书开放平台](https://open.feishu.cn/app) → **创建企业自建应用**
2. 填写应用名称（如 `HarnessX Bot`）、描述，上传图标
3. 记录 **App ID** 和 **App Secret**（首页 → 凭证与基础信息）

---

## 第二步：开通权限

进入应用 → **权限管理** → **批量导入权限**，粘贴以下 JSON 一键开通所有必要权限：

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

> **说明**：
> - `im:message` — 发送、编辑消息
> - `im:message.group_msg` — 接收群组消息
> - `im:message.p2p_msg:readonly` — 接收单聊消息
> - `im:message.reactions:read` — 使用 Emoji Reaction（⏳/✅/❌ 进度反馈）
> - `im:resource` — 上传/下载图片、文件
> - `im:chat` — 读取群组信息
> - `contact:user.base:readonly` — 读取用户基本信息（用于 @机器人 解析）

权限开通后，点击**申请发布**，等待管理员审批（沙盒环境无需审批）。

---

## 第三步：选择接入模式

### 模式一：WebSocket（推荐）

无需公网 IP，飞书服务器主动连接你的服务，适合本地开发和内网部署。

1. 进入应用 → **事件订阅** → **请求网址校验** 选择「使用长连接接收事件」
2. 在**事件订阅**中添加以下事件：
   - `im.message.receive_v1` — 接收消息（必须）
   - `im.chat.member.bot.added_v1` — 机器人被邀请入群（可选）
3. 无需填写 `verification_token`，WebSocket 模式不校验签名

配置文件：

```yaml
channels:
  feishu:
    enabled: true
    app_id: "cli_xxxx"
    app_secret: "xxxxxxxxxxxx"
    mode: websocket           # 默认即 websocket，可省略
    require_mention: true     # 群聊中需要 @机器人
    reply_in_thread: false
```

### 模式二：Webhook

需要公网可访问的 HTTPS 地址（或使用 ngrok 等内网穿透工具）。

1. 进入应用 → **事件订阅** → 填写请求网址：
   ```
   https://your-domain.com/gateway/webhook/feishu
   ```
2. 复制页面上显示的 **Verification Token**，填入配置
3. 如需开启 AES 加密：记录 **Encrypt Key**

配置文件：

```yaml
channels:
  feishu:
    enabled: true
    app_id: "cli_xxxx"
    app_secret: "xxxxxxxxxxxx"
    mode: webhook
    verification_token: "xxxx"   # 事件订阅页面的 Verification Token
    encrypt_key: "xxxx"          # 可选：AES 加密 Key
    require_mention: true
```

---

## 第四步：将机器人添加到应用

进入应用 → **应用功能** → **机器人** → 开启机器人功能。

完成后，将应用发布并安装到企业，或直接在沙盒中测试。

---

## 第五步：获取 Chat ID（可选，用于 Cron 主动推送）

如需通过定时任务向指定群组或单聊推送消息，需要知道 `chat_id`。

**方法**：向机器人发送任意消息，在 Gateway 日志中查找：

```
hx-gateway logs -n 50 | grep chat_id
```

或通过飞书 API 查询：

```bash
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://open.feishu.cn/open-apis/im/v1/chats?page_size=20"
```

---

## 配置字段完整参考

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `app_id` | string | ✅ | — | 应用 App ID（`cli_` 开头） |
| `app_secret` | string | ✅ | — | 应用 App Secret |
| `mode` | string | | `websocket` | 接入模式：`websocket` 或 `webhook` |
| `verification_token` | string | Webhook 模式必填 | — | 事件订阅 Verification Token |
| `encrypt_key` | string | | — | AES 加密 Key（Webhook 模式可选） |
| `require_mention` | bool | | `true` | 群组消息是否需要 @机器人 才触发 |
| `reply_in_thread` | bool | | `false` | 是否在话题（Thread）内回复，避免刷屏 |
| `session_mode` | string | | `shared` | `shared`（群共享）或 `per_user`（每人独立） |
| `max_steps` | int | | `30` | 每次任务最大步骤数 |
| `token_budget` | int | | `100000` | 每次任务 token 上限 |
| `allowed_users` | list | | `[]`（所有人） | 白名单 open_id 列表，空=允许所有人 |
| `workspace` | string | | `auto` | 文件工作区模式 |

---

## 功能特性

| 功能 | 支持 | 说明 |
|------|------|------|
| 文本消息 | ✅ | Markdown 渲染 |
| 图片/文件接收 | ✅ | 自动下载到本地，传递给 Agent |
| 流式编辑（打字机效果） | ✅ | edit_interval = 200ms |
| Emoji Reaction 进度 | ✅ | ⏳ 处理中 → ✅ 完成 / ❌ 失败 |
| 话题（Thread）回复 | ✅ | `reply_in_thread: true` 时启用 |
| Webhook AES 解密 | ✅ | 配置 `encrypt_key` 后自动解密 |
| 主动推送（Cron） | ✅ | 需要 receive_id 已记录（历史消息触发后自动记录） |
| 消息长度上限 | — | 30,000 字符 |

---

## 常见问题

### 机器人没有收到消息

1. 确认应用已发布并安装到企业/团队
2. 检查事件订阅是否已开启 `im.message.receive_v1`
3. 群组消息确认 `require_mention: true` 时已 @机器人
4. 查看日志：`hx-gateway logs -n 100 | grep feishu`

### WebSocket 连接频繁断开

- 检查网络稳定性，Gateway 内置指数退避重连（5s → 10s → 30s → 60s → 120s）
- 查看日志中 `[feishu] disconnected` 后的错误信息

### Webhook 签名验证失败

- 确认 `verification_token` 与飞书后台一致
- 如开启加密，确认 `encrypt_key` 正确

### 权限不足（403 错误）

- 检查应用权限是否已申请并通过审批
- 企业版需要管理员在飞书管理后台批准

---

## 完整配置示例

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
