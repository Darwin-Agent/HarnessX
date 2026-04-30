# Stateful Trajectory

`StatefulTrajectory` is the primary structured output of `harness.run()`.
It records step-by-step state, model actions, tool observations, and rewards.

---

## Runtime flow

```
harness.run(task)
  -> run_loop()
     -> snapshot before step (z_t)
     -> model + tools execution
     -> state delta after step (Δz_t)
     -> TrajectoryStep appended
  -> TaskEndEvent
  -> HarnessResult(trajectory=StatefulTrajectory)
```

---

## Core structures

### StatefulTrajectory

- `run_id`
- `steps: list[TrajectoryStep]`
- `parent_run_id` (for child trajectories)
- `metadata`

### TrajectoryStep

- `state_snapshot: FullStateSnapshot` (`z_t`, before step)
- `state_delta: StateDelta` (`Δz_t`, slot changes)
- `action: ModelResponseEvent | None`
- `observation: list[ToolResultEvent]`
- `event: StepEndEvent | None`
- `reward: float`
- `step_start_event: StepStartEvent | None` (assembled context snapshot)
- `subagent_trajectories`
- `token_annotation: TokenAnnotation | None`

### StateDelta

Slot-level change operations:

- `create`
- `update`
- `delete`

---

## Rewards

Default flow:

1. step rewards start at `0.0`
2. evaluator computes terminal result (`EvalResult`)
3. `backfill_rewards()` writes reward to steps

Custom per-step reward with PRM:

```python
from harnessx.processors.evaluation.strategies.evaluators.prm import DiscountedPRM

prm = DiscountedPRM(gamma=0.9)
step_rewards = await prm.score_steps(result.trajectory, task)
for step, r in zip(result.trajectory.steps, step_rewards):
    step.reward = r
```

---

## Export formats

### SFT / offline records

```python
records = result.trajectory.to_training_records()
```

Each record contains OpenAI-format messages for one step plus metadata/reward.

### RL episode records

`to_rl_records(fmt)` requires `token_annotation` on every step.
Token annotations are populated by providers/rollout helpers that support token capture.

```python
# fmt is an RLFormat implementation (for example SlimeRLFormat)
episode = result.trajectory.to_rl_records(fmt)
```

---

## Storage backends

HarnessX provides trajectory stores:

- `JsonlTrajectoryStore`
- `SQLiteTrajectoryStore`

```python
from harnessx.data.trajectory_store import JsonlTrajectoryStore

store = JsonlTrajectoryStore(path="trajectories/")
await store.save(result.trajectory)
loaded = await store.load(result.trajectory.run_id)
```

---

## Multi-agent linkage

Child trajectories are linked via `parent_run_id` and embedded in step exports, preserving parent-step -> child-run causality.

---

## Related docs

- [Processors](processors.md)
- [Harness Composition](harnesses.md)
- [Training Data Recipe](../recipes/training_data.md)
