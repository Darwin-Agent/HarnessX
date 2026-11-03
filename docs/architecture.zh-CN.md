# 架构

[English](architecture.md) | **简体中文**

<p align="center">
  <img src="assets/harnessx_architecture.png" alt="HarnessX 架构" width="800"/>
</p>

HarnessX 将代理行为拆分为 **9 个正交维度**，对应三个核心方向：

> 🧩 **组合**：模型、上下文、记忆、工具、沙箱
>
> ⚙️ **适配**：评估、控制
>
> 🚀 **进化**：观测、训练

| 维度 | 职责 |
| --- | --- |
| 模型选择 | 多服务商路由与主模型、裁判、评估器角色分配；`ModelConfig` 与 `HarnessConfig` 分离 |
| 上下文组装 | 系统提示策略、历史截断与用户消息包装；`processors/context/` |
| 记忆管理 | 提取、存储与检索，包括 Light-Memory 等可组合策略；`processors/memory/` |
| 工具生态 | 内置工具、MCP、技能与过滤；`processors/tools/` 和 `tools/` |
| 执行环境 | 本地、Docker、E2B 沙箱隔离；`sandbox/` |
| 评估与奖励 | 模型裁判、自检、过程奖励与基准评估器；`processors/evaluation/` |
| 控制与安全 | 循环检测、成本保护、上下文压缩等控制处理器；`processors/control/` |
| 可观测性 | HarnessJournal、OpenTelemetry、检查点与会话恢复；`processors/observability/` |
| 训练桥接 | 将轨迹转换为监督微调或强化学习记录，支持词元级标注；`rl/` |

## 事件驱动的处理器流水线

所有行为通过处理器在 8 个事件钩子上实现：

<p align="center">
  <img src="assets/pipeline_hooks.png" alt="事件驱动的处理器流水线" width="800"/>
</p>

使用 `|` 组合能力包，构建时检查单例分组冲突：

```python
from harnessx.bundles.coding import make_coding
from harnessx.bundles.reliability import make_reliability

config = (make_coding(working_dir=".") | make_reliability()).build()
```

冲突会抛出 `HarnessConflictError`，不会静默覆盖已有配置。

Harness 搜索与模型训练的组合方式见[进化指南](guide/evolution.zh-CN.md)。
