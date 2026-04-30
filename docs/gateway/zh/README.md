# HarnessX Gateway

HarnessX Gateway 是一个多平台 IM 消息网关，将飞书、钉钉、Telegram、Slack、Discord 等 IM 平台的消息路由到 HarnessX Agent，并将 Agent 的流式回复实时推送回各平台。

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                          hx-gateway 进程                             │
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
│  │ CronManager│ — 定时任务，定期向 Agent 发起 prompt                   │
│  └────────────┘                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `BaseChannel` | `gateway/core/base_channel.py` | 连接管理、断线重连、去重、防抖、限速 |
| `ChannelDispatcher` | `gateway/core/dispatch.py` | 消息优先级队列、Session 管理、Harness 调度 |
| `SessionStore` | `gateway/core/session_store.py` | 对话状态持久化，idle timeout GC |
| `CronManager` | `gateway/core/cron.py` | 5-field cron / `every:` 定时任务 |
| `IMProgressProcessor` | `gateway/core/im_stream.py` | 流式进度注入（工具调用进度卡片） |
| FastAPI Server | `gateway/server.py` | REST API + Web Console + Webhook 入口 |

### 消息处理流程

```
1. Channel._connect()      — 建立 WebSocket / 注册 Webhook / 启动 LongPoll
2. Channel._listen()       — 接收平台推送，调用 _enqueue(event)
3. _enqueue()              — 去重检查 → 50ms 防抖合并 → dispatcher.enqueue()
4. Dispatcher              — PriorityQueue(cmd=0, msg=1) → 按 session 串行消费
5. harness.run(task)       — 调用 HarnessX Agent，流式 token 写入 asyncio.Queue
6. channel.send_stream()   — 消费 token queue，实时更新/编辑 IM 消息
```

### 关键设计

- **按 session 串行**：同一对话（session_id）的消息串行处理，不同对话并行。
- **优先级队列**：`/cancel`、`/reset` 等命令优先级为 0，普通消息为 1，确保命令立即生效。
- **流式编辑**：支持编辑消息的平台（飞书、Telegram、Slack、Discord）通过定时 edit 实现打字机效果；不支持编辑的平台（钉钉）发送新消息替代。
- **Token Bucket 限速**：`OutboundRateLimiter` 防止超过平台 API 速率限制，内置抖动（jitter）防止雪崩。
- **持久化去重**：`~/.harnessx/dedup/{channel}.json` 跨重启防消息重复处理（TTL 10 分钟）。

---

## 安装

```bash
pip install harnessx[gateway]
# 或手动安装所需平台依赖：
pip install lark-oapi           # 飞书
pip install dingtalk-stream     # 钉钉
pip install python-telegram-bot # Telegram
pip install slack-bolt          # Slack
pip install "discord.py>=2.0"   # Discord
```

---

## 快速启动

```bash
# 1. 创建配置文件（见下节）
mkdir -p ~/.harnessx
vi ~/.harnessx/gateway.yaml

# 2. 启动（后台守护进程）
hx-gateway start

# 3. 查看状态
hx-gateway status

# 4. 查看日志
hx-gateway logs -f

# 5. 停止
hx-gateway stop
```

启动后访问 **`http://localhost:8080/console/`** 打开 Web 管理控制台。

---

## gateway.yaml 配置参考

```yaml
# ~/.harnessx/gateway.yaml

# ── Gateway 服务器设置 ─────────────────────────────────────────────────────
gateway:
  host: "0.0.0.0"        # 监听地址（默认 0.0.0.0）
  port: 8080             # 监听端口（默认 8080）
  agent_id: "gateway"    # 全局 agent_id，用于会话隔离

# ── 默认设置（所有 channel 继承）──────────────────────────────────────────
default:
  workspace: "auto"      # "auto" = 为每个 channel 创建独立文件工作区
  max_steps: 30          # 每次 run() 最大步骤数
  token_budget: 100000   # 每次 run() token 上限

# ── Channel 定义 ─────────────────────────────────────────────────────────
channels:
  # 每个 key 是 channel 实例名（可自定义，用于日志/API 路由）
  feishu:
    enabled: true
    channel_type: feishu   # 可选，默认与 key 相同
    app_id: "cli_xxxx"
    app_secret: "xxxx"
    mode: websocket        # websocket | webhook（见飞书文档）
    require_mention: true  # 群组中是否需要 @机器人
    reply_in_thread: false # 是否在话题内回复

  dingtalk:
    enabled: false
    channel_type: dingtalk
    client_id: "dingxxxx"
    client_secret: "xxxx"
    card_template_id: ""   # 可选：AI 流式卡片模板 ID

  telegram_bot:
    enabled: false
    channel_type: telegram
    bot_token: "123456:ABC-DEF..."
    allowed_users:
      - "123456789"        # Telegram user_id 白名单（空列表=所有人）

  slack:
    enabled: false
    channel_type: slack
    bot_token: "xoxb-..."
    app_token: "xapp-..."  # Socket Mode 专用
    reply_in_thread: true

  discord:
    enabled: false
    channel_type: discord
    bot_token: "MTxxxx..."

# ── 心跳任务（可选）─────────────────────────────────────────────────────
heartbeat:
  enabled: true
  cron: "0 9 * * 1-5"    # 每工作日 9:00（UTC）
  prompt: "请总结今日待办事项"
  channel: feishu
  chat_id: "oc_xxxx"      # 飞书 chat_id（群组或单聊）
```

### 模型配置

Gateway 使用与 CLI 相同的模型配置优先级：

1. `{agent_home}/model_config.yaml`
2. `~/.harnessx/model_config.yaml`
3. 环境变量 `HARNESSX_MODEL`
4. 默认 `gpt-4o`

```yaml
# ~/.harnessx/model_config.yaml
main:
  provider: litellm
  model: claude-sonnet-4-6
```

---

## REST API

Gateway 启动后在 `http://localhost:8080` 提供以下 API：

### 基础

| 方法 | 路径 | 描述 |
|------|------|------|
| `GET` | `/gateway/health` | 健康检查 |
| `GET` | `/gateway/config` | 获取 gateway 全局配置 |
| `PUT` | `/gateway/config` | 更新 gateway 全局配置 |

### Channel 管理

| 方法 | 路径 | 描述 |
|------|------|------|
| `GET` | `/gateway/channels` | 列出所有 channel 及状态 |
| `GET` | `/gateway/channels/{name}/config` | 获取指定 channel 配置 |
| `PUT` | `/gateway/channels/{name}/config` | 更新 channel 配置（热更新） |
| `GET` | `/gateway/channels/{name}/status` | 获取 channel 连接状态 |
| `POST` | `/gateway/channels/{name}/restart` | 重启指定 channel |
| `POST` | `/gateway/channels/{name}/reset_session` | 重置指定 channel 的会话 |
| `POST` | `/gateway/channels/create` | 创建新 channel |
| `DELETE` | `/gateway/channels/{name}` | 删除 channel |
| `GET` | `/gateway/channel-types` | 列出已注册的 channel 类型及 schema |

### Webhook

| 方法 | 路径 | 描述 |
|------|------|------|
| `POST` | `/gateway/webhook/{channel_name}` | 统一 Webhook 入口（平台回调） |
| `GET` | `/gateway/webhook/{channel_name}` | 部分平台验证用（如微信） |

### 会话

| 方法 | 路径 | 描述 |
|------|------|------|
| `GET` | `/gateway/sessions` | 列出活跃会话（支持 `?channel=` 过滤） |

### 定时任务（Cron）

| 方法 | 路径 | 描述 |
|------|------|------|
| `GET` | `/gateway/cron/jobs` | 列出所有定时任务 |
| `POST` | `/gateway/cron/jobs` | 创建定时任务 |
| `GET` | `/gateway/cron/jobs/{job_id}` | 获取定时任务详情 |
| `PUT` | `/gateway/cron/jobs/{job_id}` | 更新定时任务 |
| `DELETE` | `/gateway/cron/jobs/{job_id}` | 删除定时任务 |
| `POST` | `/gateway/cron/jobs/{job_id}/run` | 立即触发执行 |
| `POST` | `/gateway/cron/jobs/{job_id}/pause` | 暂停 |
| `POST` | `/gateway/cron/jobs/{job_id}/resume` | 恢复 |

### 心跳

| 方法 | 路径 | 描述 |
|------|------|------|
| `GET` | `/gateway/heartbeat` | 获取心跳状态 |
| `GET` | `/gateway/heartbeat/config` | 获取心跳配置 |
| `PUT` | `/gateway/heartbeat/config` | 更新心跳配置 |

### Web Console

| 路径 | 描述 |
|------|------|
| `GET /console/` | Web 管理控制台（前端 UI） |

---

## 定时任务（Cron）

定时任务存储在 `~/.harnessx/im-workspaces/{agent_id}/cron_jobs.json`，支持通过 API 或配置文件管理。

### 创建定时任务

```bash
curl -X POST http://localhost:8080/gateway/cron/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "每日报告",
    "cron": "0 9 * * 1-5",
    "prompt": "请生成今日工作日报",
    "channel": "feishu",
    "chat_id": "oc_xxxx",
    "timeout": 120
  }'
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 任务名称 |
| `cron` | string | 5-field cron 表达式（`"0 9 * * 1-5"`） |
| `every` | string | 间隔简写，与 `cron` 二选一（`"30m"`、`"1h"`、`"2h30m"`） |
| `prompt` | string | 发送给 Agent 的提示文本 |
| `channel` | string | 回复所用 channel 名称（空=静默运行，不发送回 IM） |
| `chat_id` | string | 平台的 chat/channel/group ID |
| `session_id` | string | 会话 ID（默认 `"cron"`，同 session_id 共享对话历史） |
| `timeout` | int | 超时秒数（默认 120） |
| `timezone` | string | 时区（默认 UTC，如 `"Asia/Shanghai"`） |
| `enabled` | bool | 是否启用（默认 true） |

### Cron 表达式示例

```
0 9 * * 1-5    — 每工作日 9:00
0 */2 * * *    — 每 2 小时
*/30 * * * *   — 每 30 分钟
0 10 * * 1     — 每周一 10:00
```

---

## 会话模式

通过 `session_mode` 控制同一群组内的会话隔离策略：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `shared`（默认） | 群组内所有成员共享同一会话 | 团队协作、公共机器人 |
| `per_user` | 群组内每个成员独立会话 | 个人助理、隐私场景 |

DM（私聊）始终为 `per_user` 模式，不受此配置影响。

---

## 内置命令

所有平台的用户均可发送以下命令：

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/reset` | 清除当前会话历史，开始新对话 |
| `/cancel` | 取消正在执行的任务（优先级队列保证立即生效） |

---

## 运行时目录

| 路径 | 内容 |
|------|------|
| `~/.harnessx/gateway.yaml` | 主配置文件 |
| `~/.harnessx/dedup/` | 消息去重状态（per channel，JSON） |
| `~/.harnessx/store/` | 持久化存储（webhook 缓存、receive_id 等） |
| `~/.harnessx/media_cache/` | 媒体文件缓存（图片、语音等） |
| `~/.harnessx/im-workspaces/{agent_id}/` | Agent 工作区根目录 |
| `~/.harnessx/im-workspaces/{agent_id}/{channel}/sessions/` | 会话 trace（JSONL） |
| `~/.harnessx/im-workspaces/{agent_id}/cron_jobs.json` | 定时任务持久化 |
| `/tmp/hx-gateway/gateway.pid` | 守护进程 PID |
| `/tmp/hx-gateway/gateway.log` | 守护进程日志 |

---

## Channel 配置指南

- [飞书（Feishu）](channels/feishu.md)
- [钉钉（DingTalk）](channels/dingtalk.md)
- [Telegram](channels/telegram.md)
- [Slack](channels/slack.md)
- [Discord](channels/discord.md)
