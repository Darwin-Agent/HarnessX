# 钉钉（DingTalk）Channel 配置指南

---

## 前提条件

需要一个**钉钉开放平台**企业内部应用（需要企业管理员权限或使用钉钉开发者沙盒）。

---

## 第一步：创建企业内部应用

1. 打开 [钉钉开放平台](https://open.dingtalk.com/) → **应用开发** → **企业内部开发** → **机器人**
2. 点击**创建应用**，填写名称和描述
3. 进入应用详情 → **凭据与基础信息**，记录：
   - **Client ID**（即 AppKey）
   - **Client Secret**（即 AppSecret）

---

## 第二步：开通机器人功能

1. 进入应用 → **消息推送** → 开启**机器人**
2. 设置机器人名称和头像
3. 设置**消息接收模式**：
   - 选择 **Stream 模式**（推荐，无需公网 IP）
   - 或选择 HTTP 回调（需要公网 HTTPS 地址）

---

## 第三步：开通权限

进入应用 → **权限管理**，开通以下权限：

| 权限 | 说明 |
|------|------|
| `qyapi_robot_sendmsg` | 机器人发送消息 |
| `Contact.User.Read` | 读取用户信息（用于用户名解析） |

部分权限需要管理员审批。

---

## 第四步：发布应用

1. 进入应用 → **版本管理与发布** → **确认发布**
2. 企业管理员在**OA 后台** → **工作台** 中安装应用
3. 安装后，将机器人添加到目标群组（群组 → 群设置 → 机器人 → 添加）

---

## 配置文件

### 基础配置（Stream 模式，推荐）

```yaml
channels:
  dingtalk:
    enabled: true
    channel_type: dingtalk
    client_id: "dingxxxxxxxxxxxx"
    client_secret: "your_client_secret"
    require_mention: false   # 群组中是否需要 @机器人
    max_steps: 30
```

### 启用 AI 流式卡片（可选）

钉钉支持 AI 交互卡片，可实现流式打字机效果（类似 ChatGPT 的逐字输出）：

```yaml
channels:
  dingtalk:
    enabled: true
    client_id: "dingxxxxxxxxxxxx"
    client_secret: "your_client_secret"
    card_template_id: "your_card_template_id"   # AI 卡片模板 ID
    require_mention: false
```

#### 获取 AI 卡片模板 ID

1. 进入 [钉钉卡片平台](https://card.dingtalk.com/)
2. **创建卡片** → 选择 **AI 卡片模板**（或参考钉钉 AI 卡片示例模板）
3. 发布后，在模板列表复制**模板 ID**

> **注意**：AI 卡片状态机：
> - `PROCESSING` → 创建卡片（收到消息时）
> - `INPUTING` → 流式输出中（收到第一个 token）
> - `FINISHED` → Agent 完成
> - `FAILED` → 出错时

---

## 配置字段完整参考

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `client_id` | string | ✅ | — | App Key（Client ID） |
| `client_secret` | string | ✅ | — | App Secret |
| `card_template_id` | string | | — | AI 流式卡片模板 ID，不填则发送普通文本消息 |
| `msg_key` | string | | `sampleMarkdown` | Open API 消息模板类型 |
| `require_mention` | bool | | `false` | 群组中是否需要 @机器人 |
| `mention_patterns` | list | | `[]` | 自定义唤醒词正则列表（如 `["AI.*", "机器人"]`） |
| `max_steps` | int | | `30` | 每次任务最大步骤数 |
| `token_budget` | int | | `100000` | 每次任务 token 上限 |
| `workspace` | string | | `auto` | 文件工作区模式 |

---

## 功能特性

| 功能 | 支持 | 说明 |
|------|------|------|
| 文本消息 | ✅ | Markdown 渲染 |
| 图片/文件接收 | ✅ | 自动下载 |
| 流式输出 | ✅ | 需配置 `card_template_id`，否则完成后一次发送 |
| 消息编辑 | ❌ | 钉钉不支持编辑已发送消息 |
| Emotion Reaction 进度 | ✅ | 🤔 处理中 → 🥳 完成 / ☹️ 失败 |
| Access Token 自动刷新 | ✅ | 过期前自动刷新（TTL 7200s） |
| Webhook 持久化 | ✅ | session_webhook 写盘，重启后仍可发送 |
| Open API 回退 | ✅ | webhook 过期后通过 Open API 发送 |
| AI 卡片持久化恢复 | ✅ | 重启后孤儿卡片设为 FAILED |
| 消息长度上限 | — | 20,000 字符 |

---

## Webhook 持久化说明

钉钉 Stream 模式下，每次用户发消息都会携带一个 `session_webhook` URL（有效期约 1 小时）。

Gateway 会将此 URL 持久化到 `~/.harnessx/store/dingtalk_webhooks.json`，即使进程重启，也可通过已记录的 webhook URL 向用户发消息（适用于 Cron 定时推送）。

当 webhook 过期时，自动回退到 DingTalk Open API 发送（需要 `client_id` 和 `client_secret` 配置正确）。

---

## 唤醒词配置

默认情况下（`require_mention: false`），群组中所有发给机器人的消息都会触发。

如需自定义唤醒条件：

```yaml
dingtalk:
  require_mention: false
  mention_patterns:
    - "^AI"       # 以 AI 开头
    - "机器人"     # 包含"机器人"
    - "@Bot"
```

满足任一正则即触发，不满足则忽略。

---

## 常见问题

### Stream 连接失败

- 确认 `client_id` 和 `client_secret` 正确
- 确认企业已安装该应用
- 查看日志：`hx-gateway logs -n 100 | grep dingtalk`

### 机器人无法发消息到群组

- 确认机器人已被添加到目标群组
- session_webhook 可能已过期（> 1 小时无互动），需用户再次发消息触发
- Open API 回退需要企业应用已获得 `qyapi_robot_sendmsg` 权限

### AI 卡片不显示

- 确认 `card_template_id` 正确，且模板已发布
- 检查企业是否已开通 AI 卡片功能权限

---

## 完整配置示例

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
