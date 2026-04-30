# Collecting Training Data

HarnessX can produce training-ready trajectories during normal agent execution.

- SFT/offline: `trajectory.to_training_records()`
- RL episode: `trajectory.to_rl_records(fmt)` (after token annotation)

---

## Minimal setup

```python
from harnessx import BaseTask
from harnessx.core.builder import HarnessBuilder
from harnessx.core.model_config import ModelConfig
from harnessx.providers.litellm_provider import LiteLLMProvider
from harnessx.bundles import context, coding
from harnessx.processors.evaluation.evaluation import EvaluationProcessor
from harnessx.processors.evaluation.strategies.evaluators.self_verify import SelfVerifyEvaluator

harness_config = (
    HarnessBuilder()
    | context
    | coding
).add(
    EvaluationProcessor(SelfVerifyEvaluator())
).build()

model = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6"))
harness = model.agentic(harness_config)

task = BaseTask(
    description="Sort [3,1,4,1,5,9] and return only the sorted list",
    success_criteria="contains the sorted list in ascending order",
)

result = await harness.run(task)
if result.eval_result and result.eval_result.passed:
    records = result.trajectory.to_training_records()
```

---

## Evaluators

Evaluators run on `task_end` and produce terminal `EvalResult`.

### SelfVerifyEvaluator

```python
from harnessx.processors.evaluation.strategies.evaluators.self_verify import SelfVerifyEvaluator
```

### LLMJudgeEvaluator

```python
from harnessx.processors.evaluation.strategies.evaluators.llm_judge import LLMJudgeEvaluator

judge = LLMJudgeEvaluator(max_conv_messages=15)
```

---

## Process Reward Models (PRM)

PRMs provide per-step rewards for RL-style training.

```python
from harnessx.processors.evaluation.strategies.evaluators.prm import DiscountedPRM

traj = result.trajectory
prm = DiscountedPRM(gamma=0.9)
step_rewards = await prm.score_steps(traj, task)

for step, r in zip(traj.steps, step_rewards):
    step.reward = r
```

Available PRM implementations:

- `TerminalPRM`
- `DiscountedPRM`
- `ToolSuccessPRM`
- `LLMJudgePRM`

---

## Batch collection

```python
import asyncio
from harnessx.data.trajectory_store import JsonlTrajectoryStore

store = JsonlTrajectoryStore(path="trajectories/my_experiment")

async def run_task(task):
    result = await harness.run(task)
    if result.eval_result and result.eval_result.passed:
        await store.save(result.trajectory)

await asyncio.gather(*[run_task(t) for t in task_batch])
```

---

## RL bridge examples

### Slime format

```python
from recipe.slime.formats.slime_format import SlimeRLFormat

fmt = SlimeRLFormat(tokenizer=tokenizer)
episode = result.trajectory.to_rl_records(fmt)
```

### Token annotation helpers

Token annotations are filled by providers/rollout helpers that support token capture.
After annotation is available, call `trajectory.to_rl_records(fmt)`.

---

## Session traces

Default tracer is `HarnessJournal`.
It writes session event streams and trace metadata under workspace `sessions/`, including state checkpoints for resume/replay.

For structure details, see:

- `docs/concepts/trajectory.md`
- `docs/concepts/harnesses.md`
