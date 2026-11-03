# Harness 与模型协同演进

[English](README.md) | **简体中文**

运行从一个 Harness 开始。每轮只使用最新 Harness 正常进行 Agent 探索并保存探索轨迹；探索结束后，MetaAgent 基于这些轨迹执行 AEGIS Harness 进化，最新生成的有效配置直接成为下一轮 Harness。累计 buffer 中存在符合条件的跨 Harness 轨迹组时，执行离线模型更新。

## 运行

在仓库根目录执行。准备 Python 3.12、支持 CUDA 的 PyTorch 环境、本地 Hugging Face 任务模型、MetaAgent `ModelConfig`、搜索开发集 JSONL 和训练 Parquet 数据。

```bash
python3.12 -m venv .venv-coevo
source .venv-coevo/bin/activate
python -m pip install -e .
python -m recipe.verl.bootstrap
python -m pip install -c recipe/verl/constraints.txt -e './.deps/verl[sglang]'
export PYTHONPATH="$PWD:$PWD/.deps/verl"

export MODEL_DIR=/path/to/local-model
export META_CONFIG=/path/to/meta-model.yaml
export SEARCH_FILE=/path/to/search-dev.jsonl
export TRAIN_FILE=/path/to/train.parquet
export OUTPUT_DIR=/path/to/coevolution-run
export HARNESS_CONFIG=/path/to/initial-harness.yaml

python -m recipe.coevolution.run --backend verl \
  --initial-model "$MODEL_DIR" \
  --initial-harness "$HARNESS_CONFIG" \
  --meta-model "$META_CONFIG" \
  --search-tasks "$SEARCH_FILE" \
  --train-file "$TRAIN_FILE" \
  --output "$OUTPUT_DIR" \
  --rounds 6 --model-warmup-rounds 4 --steps 10 \
  --train-batch-size 1 --gpus 8 \
  --server-args '["--tp", "1"]'
```

`MODEL_DIR` 必须包含模型权重、tokenizer 文件和 chat template。`HARNESS_CONFIG` 可选；省略时使用内置基线 Harness。启动时只需一个 Harness。每轮探索后，最新生成的有效 AEGIS 配置无条件成为下一轮 Harness，不按分数回退到旧版。轨迹按任务跨轮累计，并为每个 Harness 版本保留最新样本。前 `--model-warmup-rounds` 轮只填充 replay buffer，之后在任务具备跨 Harness 奖励差异时执行模型更新；默认预热四轮。`META_CONFIG` 配置提出 Harness 修改的元模型，与任务模型目录分开。根据所用模型和硬件调整 `--gpus`、`--train-batch-size`、`--steps` 和学习率。

中断后使用相同输入和参数，并添加 `--resume`：

```bash
python -m recipe.coevolution.run --backend verl \
  --initial-model "$MODEL_DIR" --meta-model "$META_CONFIG" \
  --initial-harness "$HARNESS_CONFIG" \
  --search-tasks "$SEARCH_FILE" --train-file "$TRAIN_FILE" \
  --output "$OUTPUT_DIR" --rounds 6 --model-warmup-rounds 4 --steps 10 \
  --train-batch-size 1 --gpus 8 \
  --server-args '["--tp", "1"]' --resume
```

## 每轮流程

1. 冻结当前任务模型，仅使用最新 Harness 正常运行 Agent 探索。每个任务按配置执行单次 rollout；这是常规 Agent 采样，不是模型训练。将完整轨迹、奖励、token 标注、来源模型、Harness ID 和行为策略 logprob 保存到 buffer。
2. 本轮探索结束后执行 AEGIS Harness 进化。MetaAgent 根据刚保存的探索轨迹生成新 Harness；最新生成的有效配置直接作为下一轮 Harness，不与旧 Harness 比分后回退。
3. 从 buffer 读取跨轮、跨 Harness 轨迹组。默认容量为 `2048` 条轨迹，可用 `--buffer-capacity N` 调整；每轮加入数据后优先淘汰最旧的完整任务组。预热期间不触发模型更新；预热结束后，缺少多个 Harness 轨迹或组内奖励无差异的任务组不参与 GRPO，但本轮仍完成最新 Harness 交接。
4. 分布式更新器只使用 buffer 中已有轨迹执行跨 Harness 离线 GRPO：按任务计算组内 advantage，并使用 `exp(current_logprob - saved_behavior_logprob)` 计算 clipped off-policy ratio。此阶段不生成回答，也不启动 rollout 服务。
5. 有符合条件的轨迹组时，将更新后的 Hugging Face 模型导出到该轮目录，并保留 tokenizer 和 chat template；尚无合格组时沿用当前模型。下一轮始终使用 AEGIS 最新生成的 Harness 开始探索。

Buffer 使用 JSONL。每条记录包含 `task_uid`、`harness_id`、`round`、`source_model`、`prompt_ids`、`response_ids`、`response_mask`、`old_logprobs` 和 `reward`。`old_logprobs` 在采集轨迹时保存，训练时不会伪造或重新计算。

## 输入格式

搜索 JSONL 每行包含一道问题和参考答案：

```json
{"question": "What is 6 multiplied by 7?", "answer": "42"}
```

训练 Parquet 使用以下字段：

```python
{
    "prompt": [{"role": "user", "content": question}],
    "reward_model": {"style": "rule", "ground_truth": answer},
    "data_source": "research",
    "extra_info": {"task_type": "research", "id": task_id},
}
```

必须提供 `prompt` 和 `reward_model.ground_truth`。如果没有 `extra_info.id`，系统根据 prompt 内容生成稳定任务 ID。最终留出测试集应与搜索集、训练集分开。

MetaAgent 配置采用 HarnessX `ModelConfig`，例如：

```yaml
schema_version: 2
models:
  - id: meta
    _target_: harnessx.providers.litellm_provider.LiteLLMProvider
    model: your-provider/your-meta-model
roles:
  main:
    default: meta
```

通过环境变量提供所选 MetaAgent 服务商凭据。进化范围限定为系统提示词和 early-commit 参数；任务标签、评估器、工具和奖励定义保持固定。

## 输出与恢复

每个完成轮次会在 `state.json` 记录最新 Harness、输入/输出模型哈希、该轮是否更新模型，以及 buffer 路径和哈希。即使预热阶段尚无可训练的跨 Harness 组，Harness 交接也会正常提交；`--resume` 会校验已提交的模型、Harness、提案和 buffer，并从最近完成的模型/Harness 组合继续。未完成的尝试保留在各自的 `round_NNN_attempt_NN` 目录。

本配方不同于[进化指南](../../docs/guide/evolution.zh-CN.md)中“先 Harness 进化，再在线更新模型”的顺序入口。
