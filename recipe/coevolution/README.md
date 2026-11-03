# Harness + Model Co-Evolution

**English** | [简体中文](README.zh-CN.md)

Each run starts with one Harness. Every round explores with the latest Harness
and saves the resulting trajectories. After exploration, MetaAgent performs the
AEGIS Harness-evolution stage using those trajectories; its newest valid
proposal becomes the Harness for the next round. Cross-Harness model evolution
updates the model offline from accumulated saved trajectories when eligible
groups are available.

## Run

Run from the repository root. Use Python 3.12, a CUDA-enabled PyTorch setup, a
local Hugging Face task model, a MetaAgent `ModelConfig`, search-development
JSONL, and a training Parquet dataset.

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

`MODEL_DIR` must contain model weights, tokenizer files, and a chat template.
`HARNESS_CONFIG` is optional; if omitted, the built-in baseline Harness is used.
Only one Harness is supplied at startup. Each round explores with the latest
Harness, then unconditionally carries the newest valid AEGIS proposal into the
next round. Cross-round trajectories are grouped by task, retaining the newest
rollout for each Harness version. The first `--model-warmup-rounds` rounds only
populate the replay buffer; model updates begin afterward when an eligible
cross-Harness reward comparison exists. The default warm-up is four rounds.
`META_CONFIG` configures the model that proposes Harness changes; it is separate
from the task model. Adjust `--gpus`, `--train-batch-size`, `--steps`, and the
learning rate for the chosen model and hardware.

Resume with the same inputs and settings, adding `--resume`:

```bash
python -m recipe.coevolution.run --backend verl \
  --initial-model "$MODEL_DIR" --meta-model "$META_CONFIG" \
  --initial-harness "$HARNESS_CONFIG" \
  --search-tasks "$SEARCH_FILE" --train-file "$TRAIN_FILE" \
  --output "$OUTPUT_DIR" --rounds 6 --model-warmup-rounds 4 --steps 10 \
  --train-batch-size 1 --gpus 8 \
  --server-args '["--tp", "1"]' --resume
```

## Round flow

1. Freeze the current task-model checkpoint and run normal agent exploration
   with the single latest Harness. Each task uses the configured single rollout;
   this is ordinary agent sampling, not model training. Save the complete trajectory, reward, token annotations, source
   model, Harness ID, and behavior-policy log probabilities to the buffer.
2. After exploration finishes, run the AEGIS Harness-evolution phase. MetaAgent
   uses the saved exploration trajectories to produce a new Harness. The latest
   valid proposal becomes the next round's Harness; it is not selected by
   comparing scores with older Harness versions.
3. Read accumulated cross-round, cross-Harness trajectory groups from the
   buffer. The default capacity is `2048` trajectories and can be changed with `--buffer-capacity N`;
   trimming preserves complete task groups and evicts the oldest complete task
   groups first. During the configured warm-up, the buffer is collected but no
   model update is attempted; afterward, groups without multiple Harnesses or
   reward variation are retained but do not contribute to GRPO.
4. The distributed updater performs offline cross-Harness GRPO using only saved
   trajectories. It computes within-task advantages and applies the clipped
   off-policy ratio `exp(current_logprob - saved_behavior_logprob)`. This stage
   does not generate responses or launch a rollout service.
5. When an eligible group exists, export the updated Hugging Face model under
   the round directory, including the input tokenizer and chat template. If no
   eligible group exists yet, carry the unchanged model forward. The next round
   always explores with the latest Harness produced by AEGIS.

The buffer is JSONL. Every record contains `task_uid`, `harness_id`, `round`,
`source_model`, `prompt_ids`, `response_ids`, `response_mask`, `old_logprobs`,
and `reward`. `old_logprobs` are recorded when the trajectory is collected;
they are never fabricated or recomputed during training.

## Inputs

Search JSONL has one question and reference answer per line:

```json
{"question": "What is 6 multiplied by 7?", "answer": "42"}
```

Training Parquet rows use the following fields:

```python
{
    "prompt": [{"role": "user", "content": question}],
    "reward_model": {"style": "rule", "ground_truth": answer},
    "data_source": "research",
    "extra_info": {"task_type": "research", "id": task_id},
}
```

The `prompt` and `reward_model.ground_truth` fields are required. If
`extra_info.id` is absent, the prompt content supplies a stable task ID. Keep
the final held-out test set separate from search and training data.

The MetaAgent configuration is a HarnessX `ModelConfig`, for example:

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

Provide credentials for the chosen MetaAgent provider through the environment.
The evolved behavior is limited to the system prompt and early-commit settings;
task labels, evaluator, tools, and reward definition remain fixed.

## Output and resume

Each committed round records its selected Harness, input and output model hashes,
search scores, and buffer path and hash in `state.json`. A round is committed
only after the offline updater exports a new model. `--resume` requires identical
input contents and settings, verifies committed model, Harness, and buffer
artifacts, and continues from the latest committed model/Harness pair. Incomplete
attempts remain in their own `round_NNN_attempt_NN` directory.

This recipe is distinct from the sequential “Harness evolution, then online
model training” entry points documented in the [evolution guide](../../docs/guide/evolution.md).
