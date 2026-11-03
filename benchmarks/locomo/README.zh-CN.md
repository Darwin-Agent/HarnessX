# LoCoMo

[English](README.md) | **简体中文**

LoCoMo 用于评估长期对话记忆。每个样本包含两位参与者的多次会话，问答任务覆盖
单跳、多跳、时间推理、总结，以及无法从对话中回答的问题。

HarnessX 提供会话导入、可配置压缩、记忆召回、批量问答和模型裁判评分。

## 工作流

```text
LoCoMo 数据集
  → SessionIngester 导入会话
  → 原文、摘要、事实或 Light-Memory 压缩
  → 记忆后端
  → 问答与检索
  → 分类指标与可选模型裁判
```

启用 Light-Memory 的模型记忆模式时，问答使用专用召回路径；
其他压缩策略使用包含系统提示、记忆检索、词元预算和用户消息包装的 Harness 流水线。

## 准备环境

- Python >=3.11，并安装 HarnessX：`pip install -e .`。
- 配置所选 Anthropic 或 OpenAI 兼容服务的模型凭据。
- 准备本地 `locomo10.json`，或使用 Hugging Face `snap-research/locomo` 数据集。

从仓库根目录运行命令，并设置 `PYTHONPATH=.`。

## 使用

原文记忆基线：

```bash
PYTHONPATH=. python benchmarks/locomo/run_eval.py \
  --compressor verbatim \
  --model openai/your-model \
  --data-path /path/to/locomo10.json \
  --max-samples 10 \
  --output reports/locomo_verbatim.jsonl
```

Light-Memory 配合模型完成记忆导入和召回：

```bash
PYTHONPATH=. python benchmarks/locomo/run_eval.py \
  --compressor light-memory --llm-memory \
  --model openai/your-model \
  --memory-model openai/your-memory-model \
  --top-k 40 --half-life 9999 \
  --data-path /path/to/locomo10.json \
  --max-samples 10 \
  --persist-dir .locomo_lm \
  --output reports/locomo_lm.jsonl \
  --resume --workers 4
```

将模型占位符替换为实际服务标识。添加 `--judge --judge-model openai/your-judge-model
--judge-runs 3` 可启用模型裁判。

## 压缩策略

| 策略 | 行为 |
| --- | --- |
| `verbatim` | 保存带日期、会话编号和发言人的原文 |
| `summary` | 使用模型为每次会话生成摘要 |
| `facts` | 使用模型抽取会话事实 |
| `light-memory` | 使用文件记忆和规则处理 |
| `light-memory` + `--llm-memory` | 使用模型导入、整理与召回文件记忆 |

## 常用参数

| 参数 | 含义 |
| --- | --- |
| `--model` | 问答模型 |
| `--memory-model` | 记忆模型，默认与问答模型相同 |
| `--top-k` | 召回候选数量，默认 40 |
| `--half-life` | 衰减半衰期天数，默认 365；可设较大值减弱衰减 |
| `--no-org` | 关闭按会话整理记忆 |
| `--persist-dir` | 文件记忆目录，默认 `.locomo_lm` |
| `--workers` | 并行处理的样本数量 |
| `--resume` | 从已有 JSONL 结果继续运行 |
| `--judge` | 启用模型裁判 |
| `--judge-runs` | 每题独立裁判次数，默认 3 |
| `--max-samples` | 限制会话样本数量 |
| `--categories` | 逗号分隔的题目类别 |

完整参数见 `python benchmarks/locomo/run_eval.py --help`。

## 评分

| 类别 | 指标 |
| --- | --- |
| `single_hop_qa` | 词元 F1 |
| `multi_hop_qa` | 词元 F1 |
| `temporal_reasoning` | 词元 F1 |
| `summarization` | ROUGE-L |
| `adversarial_qa` | 对无法作答表述进行精确匹配 |

`LoCoMoEvaluator.aggregate()` 返回分类均值与整体平均分。
启用 `--judge` 后，还会报告对齐 EverMemOS 方式的模型裁判准确率，默认不包含对抗类别。
