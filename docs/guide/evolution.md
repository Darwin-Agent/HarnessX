# Harness and Model Evolution

**English** | [简体中文](evolution.zh-CN.md)

HarnessX offers three complementary workflows:

| Workflow | Use it to | Entry point |
| --- | --- | --- |
| Alternating co-evolution | Improve harness configurations and model weights together over multiple rounds | `python -m recipe.coevolution.run --backend verl` |
| Harness evolution | Search for a better configuration with fixed model weights | `python recipe/evolution_pipeline/run_gaia.py` |
| Model training | Optimize model weights with a configured harness | `bash recipe/verl/launch/run_research_rl.sh` |

Run commands from the repository root. Harness search needs credentials for your
task and meta-model providers. Distributed training needs a CUDA environment,
local model weights, and training and validation data.

## Alternating Harness + Model Co-Evolution

Start with one Harness. Each round runs normal agent exploration with the
latest Harness and saves the resulting trajectories, rewards, and behavior-policy
log probabilities. After exploration, MetaAgent performs AEGIS Harness evolution
using those trajectories. Its newest valid proposal becomes the next round's
Harness. Cross-Harness model evolution updates from accumulated buffer records
when eligible groups exist; this update stage does not generate responses.

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

Start with the [co-evolution runbook](../../recipe/coevolution/README.md) for
installation, data formats, meta-model setup, buffer grouping, hardware
settings, outputs, and resume. The recipe pins its SGLang/veRL environment for
trajectory collection and uses distributed Transformers/PyTorch for offline
model updates.

Harness search covers system prompts and early-commit settings while keeping
tools, evaluator, and reward fixed. Start with one Harness; each round uses the
latest Harness for exploration and carries the newest valid AEGIS proposal into
the next round. Exploration uses one rollout per task. Saved behavior log
probabilities drive the off-policy ratio during model updates. The buffer holds
up to 2,048 trajectories by default and trims only complete cross-Harness task
groups accumulated across rounds.

## Standalone Harness Evolution

GAIA Evolver runs tasks, collects trajectories, and asks a meta-agent to propose
harness changes. Candidates are evaluated against prior configurations.
Model weights remain fixed throughout this workflow.

Install HarnessX with `pip install -e .`, configure your provider credentials, and
prepare a local GAIA JSON dataset as described in the
[GAIA Evolver guide](../../recipe/gaia_evolver/README.md).

```bash
python recipe/evolution_pipeline/run_gaia.py \
  --model "<provider/model-for-task-agent>" \
  --meta-model "<provider/model-for-meta-agent>" \
  --provider-id "<provider-id>" \
  --data-path /path/to/gaia.json \
  --max-tasks 30 --num-rounds 3 --concurrency 1
```

Use `python recipe/evolution_pipeline/run_gaia.py --help` for all options.
The [tau2](../../recipe/tau2_evolver/README.md) and
[TerminalBench 2](../../recipe/tb2_evolver/README.md) recipes provide
task-specific evolution workflows.

## Model Training with veRL

The [veRL recipe](../../recipe/verl/README.md) runs HarnessX research agents
inside distributed GRPO. Use it for a standalone training stage with a configured
harness; use the alternating entry point above for multi-round co-evolution.

After following the veRL installation instructions, prepare these files:

- `$DATA_DIR/train.parquet`
- `$DATA_DIR/val.parquet`

```bash
export DATA_DIR=/path/to/training-data
export MODEL_DIR=/path/to/local-compatible-model
export CKPT_DIR=/path/to/checkpoints
bash recipe/verl/launch/run_research_rl.sh
```

Adjust `recipe/verl/launch/config.yaml` for your GPU count, model, context lengths,
and batch sizes. Checkpoints, rollout records, and evaluation outputs are written
under `CKPT_DIR`.

Research agents use HarnessX web search directly. `SERPAPI_API_KEY` and `TAVILY_API_KEY` are
optional; public Wikipedia, Bing, and DuckDuckGo fallbacks require network access.

## Sequential Harness-Then-Model Workflow

To run harness search first and model training second, use GAIA Evolver followed
by the [standalone GRPO trainer](../../recipe/verl_harnessX/README.md):

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

This trainer expects `train.parquet` and `val.parquet` in `DATA_DIR`. Configure
the chosen harness in the trainer and prepare training data separately.

For sequential orchestration, set `GAIA_DATA_PATH`, `GAIA_MODEL`,
`GAIA_META_MODEL`, `GAIA_PROVIDER_ID`, `DATA_DIR`, `MODEL_DIR`, and `CKPT_DIR`,
then run `bash recipe/evolution_pipeline/run_sequential.sh`. This wrapper runs
the two stages in order; dataset preparation and harness transfer are manual.

The standalone trainer's WebSearch supports optional `MCP_SEARCH_URL` and
`MCP_SEARCH_KEY`, followed by SerpAPI, Tavily, and public DuckDuckGo fallbacks.
Configure the providers you want to use and allow outbound access to them.
