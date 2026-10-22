# Harness 与模型进化

[English](evolution.md) | **简体中文**

HarnessX 提供三种相互配合的工作流：

| 工作流 | 用途 | 入口 |
| --- | --- | --- |
| 交替协同演进 | 多轮协同改进 Harness 配置与模型权重 | `python -m recipe.coevolution.run --backend verl` |
| Harness 进化 | 固定模型权重，搜索更好的配置 | `python recipe/evolution_pipeline/run_gaia.py` |
| 模型训练 | 在指定 Harness 下优化模型权重 | `bash recipe/verl/launch/run_research_rl.sh` |

所有命令在仓库根目录执行。Harness 搜索需要任务模型和元模型的服务凭据；
分布式训练需要 CUDA 环境、本地模型权重、训练集和验证集。

## Harness 与模型交替协同演进

启动时只需一个 Harness。每轮使用最新 Harness 正常运行 Agent 探索，并保存轨迹、奖励和行为策略
logprob。探索结束后，MetaAgent 基于这些轨迹执行 AEGIS Harness 进化，最新生成的有效配置
直接成为下一轮 Harness。累计 buffer 中存在符合条件的跨 Harness 轨迹组时，执行离线 GRPO；
模型更新阶段不生成回答。

```bash
python -m recipe.coevolution.run --backend verl \
  --initial-model /path/to/local-compatible-model \
  --initial-harness /path/to/initial-harness.yaml \
  --meta-model /path/to/meta-model.yaml \
  --search-tasks /path/to/search-dev.jsonl \
  --train-file /path/to/train.parquet \
  --output /path/to/coevo-run --rounds 2 --steps 10 \
  --train-batch-size 1 --gpus 8 \
  --server-args '["--tp", "1"]'
```

安装、数据格式、元模型配置、buffer 分组、硬件设置、输出与恢复方式见
[协同演进运行手册](../../recipe/coevolution/README.zh-CN.md)。轨迹采集环境固定
SGLang/veRL 依赖，模型离线更新使用分布式 Transformers/PyTorch。

Harness 搜索涵盖系统提示词和提前提交参数，工具、评估器与奖励保持固定。启动时只需一个
Harness；每轮用最新 Harness 探索，并将 AEGIS 最新生成的有效配置交给下一轮。探索阶段每个任务
采样一条轨迹；模型更新时使用轨迹保存的行为策略 logprob 计算 off-policy ratio。Buffer 默认最多
保存 2,048 条轨迹，裁剪时仅移除完整的跨轮、跨 Harness 任务组。

## 独立 Harness 进化

GAIA Evolver 执行任务、收集轨迹，并由元代理提出 Harness 变更。
候选配置与历史配置比较，整个流程保持模型权重不变。

使用 `pip install -e .` 安装 HarnessX，设置所选服务商的凭据，并按
[GAIA Evolver 说明](../../recipe/gaia_evolver/README.md)准备本地 GAIA JSON 数据。

```bash
python recipe/evolution_pipeline/run_gaia.py \
  --model "<provider/model-for-task-agent>" \
  --meta-model "<provider/model-for-meta-agent>" \
  --provider-id "<provider-id>" \
  --data-path /path/to/gaia.json \
  --max-tasks 30 --num-rounds 3 --concurrency 1
```

完整参数见 `python recipe/evolution_pipeline/run_gaia.py --help`。
[tau2](../../recipe/tau2_evolver/README.md) 和
[TerminalBench 2](../../recipe/tb2_evolver/README.md) 配方提供对应任务的进化流程。

## 使用 veRL 训练模型

[veRL 配方](../../recipe/verl/README.zh-CN.md)在分布式 GRPO 中执行 HarnessX 研究代理。
指定 Harness 的独立训练阶段使用此入口，多轮协同演进使用本页开头的交替入口。

按 veRL 安装说明配置环境后，准备以下文件：

- `$DATA_DIR/train.parquet`
- `$DATA_DIR/val.parquet`

```bash
export DATA_DIR=/path/to/training-data
export MODEL_DIR=/path/to/local-compatible-model
export CKPT_DIR=/path/to/checkpoints
bash recipe/verl/launch/run_research_rl.sh
```

根据 GPU 数量、模型、上下文长度与批大小调整 `recipe/verl/launch/config.yaml`。
模型权重、生成记录和评估输出写入 `CKPT_DIR`。

研究代理直接使用 HarnessX 网页搜索。
`SERPAPI_API_KEY` 和 `TAVILY_API_KEY` 可选；公开 Wikipedia、Bing、DuckDuckGo
搜索回退需要外部网络访问。

## 先 Harness 后模型的顺序流程

先使用 GAIA Evolver 搜索 Harness，再运行
[独立 GRPO 训练器](../../recipe/verl_harnessX/README.md)：

```bash
python recipe/evolution_pipeline/run_gaia.py \
  --model "<provider/model>" --meta-model "<provider/model>" \
  --provider-id "<provider-id>" --data-path /path/to/gaia.json \
  --max-tasks 30 --num-rounds 3 --concurrency 1

export DATA_DIR=/path/to/directory-containing-train-and-val-parquet
export MODEL_DIR=/path/to/local-base-model
export CKPT_DIR=/path/to/checkpoints
bash recipe/verl_harnessX/run_train.sh
```

此训练器读取 `DATA_DIR` 中的 `train.parquet` 与 `val.parquet`。
训练数据单独准备，选定的 Harness 在训练器中配置。

需要顺序编排时，设置 `GAIA_DATA_PATH`、`GAIA_MODEL`、`GAIA_META_MODEL`、
`GAIA_PROVIDER_ID`、`DATA_DIR`、`MODEL_DIR` 和 `CKPT_DIR`，再运行
`bash recipe/evolution_pipeline/run_sequential.sh`。
该脚本依次执行两个阶段，数据准备和 Harness 传递由用户配置。

独立训练器的网页搜索支持可选 `MCP_SEARCH_URL` 与 `MCP_SEARCH_KEY`，
随后依次使用 SerpAPI、Tavily 和公开 DuckDuckGo 回退。
按需配置搜索服务，并允许对应的外部网络访问。
