# HarnessX × veRL

**English** | [简体中文](README.zh-CN.md)

Train research agents with HarnessX rollouts and distributed veRL GRPO.
The recipe includes SGLang token/logprob capture and tool-result masking. For
alternating harness search and model training, use the [co-evolution recipe](../coevolution/README.md).

Run commands from the repository root.

## Install

Use Python 3.12 and CUDA suitable for PyTorch 2.9.1. The dependency baseline is:

| Component | Version |
| --- | --- |
| veRL | 0.7.1, commit `bec9ef74768dd201881cd4e54cd0385e87caae27` |
| PyTorch | 2.9.1 |
| SGLang | 0.5.6 |
| transformers | >=4.51,<5 |

```bash
python3.12 -m venv .venv-verl
source .venv-verl/bin/activate
python -m pip install -e .
python -m recipe.verl.bootstrap
python -m pip install -c recipe/verl/constraints.txt -e './.deps/verl[sglang]'
export PYTHONPATH="$PWD:$PWD/.deps/verl"
```

Install platform-specific attention/CUDA packages as required by your model and
hardware. Use a model supported by transformers 4.x and a matching tool-call
template. The research task spec uses Qwen3-Coder tool calls and user-role tool
results.

For offline setup, supply the upstream archive:

```bash
python -m recipe.verl.bootstrap --archive /path/to/verl-archive.tar.gz
```

## Launch model training

Prepare a local model directory and these Parquet files:

- `$DATA_DIR/train.parquet`
- `$DATA_DIR/val.parquet`

Data rows use the [research-agent schema](../coevolution/README.md#2-prepare-the-data).

```bash
export DATA_DIR=/path/to/training-data
export MODEL_DIR=/path/to/local-compatible-model
export CKPT_DIR=/path/to/checkpoints
bash recipe/verl/launch/run_research_rl.sh
```

The default configuration targets eight GPUs. Adjust `launch/config.yaml` or
pass Hydra overrides for your model, sequence lengths, batch sizes, and GPU
resources. Prepare model weights and datasets before launching. Outputs are
written under `CKPT_DIR`.

## Runtime configuration

- **Search:** optional `SERPAPI_API_KEY` and `TAVILY_API_KEY` enable their
  respective providers. Public search fallbacks require outbound network access.
- **Hugging Face:** `HF_ENDPOINT` defaults to `https://huggingface.co`; set it to
  your mirror endpoint when needed.
- **Ray:** the launcher manages a local runtime. `STOP_EXISTING_SERVICES=1`
  stops the existing local Ray cluster before launch; use this only on a
  dedicated training node.
- **Optional nested-agent mode:** MATPO remains available for workflows that use
  explicit parent/child episode relationships.

## Dependency distribution

The recipe materializes a pinned upstream checkout in `.deps/verl` and applies
the tracked integration patch. It uses these files:

| File | Purpose |
| --- | --- |
| `dependencies/verl.lock.json` | Upstream commit, archive URL, and SHA-256 |
| `dependencies/harnessx.patch` | HarnessX rollout, GRPO, and optional nested-agent integration |
| `dependencies/LICENSE` | Upstream Apache-2.0 license |
| `dependencies/Notice.txt` | Upstream attribution |
| `constraints.txt` | Training dependency constraints |

Bootstrap checks archive integrity and records source and patch hashes. Keep
the generated checkout unchanged; after updating the pin or patch, move the old
checkout aside and bootstrap a new one.

### Integration patch

| Upstream module | Integration |
| --- | --- |
| `experimental/agent_loop/agent_loop.py` | Child outputs, episode linkage, parent-first ordering, and parent-only evaluation |
| `trainer/ppo/ray_trainer.py` | Parent/child advantage handling, cross-harness routing with shared GRPO prompt IDs, and batch padding |
| `trainer/ppo/core_algos.py` | Mask-aware loss for empty microbatches |
| `trainer/ppo/rollout_corr_helper.py` | Mask-aware rollout correction metrics |
| `utils/fsdp_utils.py` | Parameter metadata handling |
| `utils/torch_functional.py` | Native PyTorch cross-entropy for sequence-parallel training |
| `workers/actor/dp_actor.py` | Input-aware multimodal sequence-parallel routing |
