# 路线图

[English](ROADMAP.md) | **简体中文**

本页列出 HarnessX 已提供的功能和后续方向。各功能的安装与使用方式见对应文档。

## 1. 记忆扩展

通过 `MemoryBundle` 和记忆处理器组合存储、提取与召回策略，无需修改模型层。

### 已提供

- ✅ **[Light-Memory](feats/light-memory.zh-CN.md)** —— 文件记忆、时间衰减、每日压缩和可选 Git 历史。
- ✅ **[LoCoMo 基准](../benchmarks/locomo/README.zh-CN.md)** —— 多会话记忆评估、可配置压缩、问答评分与模型裁判。

### 规划中

- ◻️ **第三方记忆适配器** —— MemPalace、SuperMemory、OpenVKing，作为 `recipe/` 中的可选集成。
- ◻️ **记忆质量指标** —— 在 HarnessJournal 中呈现检索精度和召回率。
- ◻️ **多模态记忆** —— 通过插件系统接入图像与视频记忆。

## 2. 强化学习与数据飞轮

核心轨迹提供 `to_training_records()` 和 `to_rl_records()`，供监督微调和强化学习使用。

### 已提供

- ✅ **[Slime 配方](../recipe/slime/README.md)** —— SGLang 轨迹、词元标注与 GRPO 训练。
- ✅ **[veRL 集成](../recipe/verl/README.zh-CN.md)** —— HarnessX 轨迹与分布式 GRPO 训练。
- ✅ **[Harness 与模型协同演进](../recipe/coevolution/README.zh-CN.md)** —— 交替执行 MetaAgent 搜索和跨 Harness GRPO，支持权重导出与轮次恢复。

### 规划中

- ◻️ **种群搜索** —— 并行评估多组 Harness，通过变异和选择改进行为配置。
- ◻️ **数据合成** —— 任务分布、多样性约束和质量过滤，以及监督微调与偏好数据导出。
- ◻️ **数据飞轮** —— 将部署轨迹接入离线训练流程。
- ◻️ **记忆专项进化** —— 面向记忆策略的种群搜索与训练数据循环。

## 3. 代理基准

`benchmarks/` 提供统一的任务加载、环境设置与评分适配器。

| 基准 | 方向 | 状态 |
| --- | --- | --- |
| GAIA | 通用助手、工具使用、网页推理 | 已提供 |
| SWE-bench | 真实代码仓库问题解决 | 已提供 |
| TAU2-Bench | 工具增强的用户模拟 | 已提供 |
| EvoClAW | 编程与算法任务 | 开发中 |
| OSWorld | 桌面界面与操作系统任务 | 开发中 |
| TerminalBench 2.0 | 终端与命令行能力 | 已提供 |
| LoCoMo | 长期对话记忆 | 已提供 |

各适配器的状态与运行入口见[基准目录](../benchmarks/README.md)。
后续方向包括 Lab 统一排行榜，以及第三方基准评分注册。

## 4. MetaHarness

### 已提供

- ✅ 观察者 Harness、元代理提案与沙箱晋升流程。

### 工作流

1. 观察者记录处理器事件、轨迹与任务得分。
2. 元代理分析表现，提出处理器或配置变更。
3. 评估候选配置，将达到晋升标准的配置用于后续运行。

独立搜索与模型协同训练的使用方式见[进化指南](guide/evolution.zh-CN.md)。
贝叶斯优化的代理模型配置搜索仍在规划中。

## 5. HarnessHUB

### 规划中

- ◻️ 一行命令拉取并运行命名的 `HarnessConfig`。
- ◻️ 版本固定与不可变发布。
- ◻️ Lab 浏览、搜索与导入面板。
- ◻️ 团队私有注册表。

## 参与贡献

欢迎贡献基准、记忆适配器和训练配方。参阅 `CONTRIBUTING.md`，
通过 GitHub 问题或拉取请求交流。
