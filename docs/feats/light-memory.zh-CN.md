# Light-Memory

[English](light-memory.md) | **简体中文**

> **记忆维度（3）** · `harnessx/plugins/dimensions/light_memory/`

Light-Memory 是仓库内实现的文件记忆系统，适合长期积累和召回知识的个人助手，
无需外部向量数据库。

## 工作方式

```text
任务提示
  → 关键词提取与实体增强
  → 检索记忆文件并计算匹配分数
  → 按指数时间衰减排序
  → 将前 K 个候选注入系统提示
  → 代理使用内置工具读取、写入与更新记忆
  → 每日压缩、后台整理与 Git 历史
```

## 核心机制

| 机制 | 说明 |
| --- | --- |
| 文件存储 | 每条记忆保存为带 YAML 元数据的 Markdown，无需外部数据库或二进制索引 |
| 记忆类型 | `style`、`profile`、`session`、`skill`、`learning`、`entity`、`daily` |
| 指数衰减 | `importance × exp(−ln2 × days / half_life)`，长期未访问的记忆自然降权 |
| 每日压缩 | 会话结束时，将对话压缩为结构化每日日志 |
| 后台整理 | 后台协程周期性合并与去重记忆文件 |
| Git 历史 | 可选集成，每次记忆写入自动提交，保留完整版本记录 |
| 代理自主召回 | 代理通过自己的工具直接读取文件 |
| 轻量存储 | 仅依赖 Python 标准库和文件系统，不需要外部存储服务 |

## 使用

```python
from harnessx.core import HarnessBuilder
from harnessx.plugins.dimensions.light_memory import LightMemoryPlugin

plugin = LightMemoryPlugin(
    memory_root="~/.harnessx/memory",
    half_life_days=30,
    top_k=15,
    auto_recall=True,
    auto_capture=False,
    auto_commit=True,
)

builder = HarnessBuilder().plugin(plugin)
```

`half_life_days` 控制衰减速度，`top_k` 控制每轮注入的候选数量。
`auto_recall` 在任务开始时提供召回候选；关闭 `auto_capture` 时由代理决定写入时机；
`auto_commit` 为记忆写入保留 Git 版本。
