# Harnesses: Composition Model

HarnessX uses explicit separation between model configuration and behavior configuration.

- `ModelConfig`: model selection/routing
- `HarnessConfig`: behavior pipeline (tools, processors, workspace, tracing)

They are combined only at runtime:

```python
agent = model.agentic(harness_config)
```

---

## Three behavior layers

```
Processor  ->  Bundle  ->  HarnessConfig
```

### Layer 1: Processor

`Processor` / `MultiHookProcessor` is the atomic behavior unit.

Examples:

- `LoopDetectionProcessor`
- `ParseRetryProcessor`
- `ToolCallCorrectionLayer`
- `EvaluationProcessor`

### Layer 2: Bundle

A bundle is an unbuilt `HarnessBuilder` fragment for one capability dimension.

Common built-ins:

- `context`
- `window_mgmt`
- `reliability`
- `coding`
- `control`
- `make_tools(...)`
- `make_execution(...)`

### Layer 3: HarnessConfig

`HarnessBuilder(...).build()` creates a final `HarnessConfig`.

```python
from harnessx.core.builder import HarnessBuilder
from harnessx.bundles import context, coding

harness_config = (
    HarnessBuilder()
    | context
    | coding
).build()
```

---

## Merge and conflict rules

`a | b` merges two builders.

Conflict checks include:

- duplicate `singleton_group` processors
- duplicate tool names
- contradictory `_after` ordering constraints

Processor ordering is controlled by:

- `_order` (ascending)
- `_after` (dependency edges in same order bucket)

---

## Hook routing

RunLoop dispatches events to:

- `processors["*"]`
- `processors[hook_key]`

`MultiHookProcessor` instances are typically added under `"*"` and route internally by event type.

---

## Typical composition patterns

### Minimal assistant

```python
from harnessx.core.builder import HarnessBuilder
from harnessx.bundles import context, window_mgmt

config = (HarnessBuilder() | context | window_mgmt).build()
```

### Coding agent

```python
from harnessx.core.builder import HarnessBuilder
from harnessx.bundles import context, coding

config = (HarnessBuilder() | context | coding).build()
```

### Add custom guard

```python
from harnessx import MultiHookProcessor
from harnessx.core.builder import HarnessBuilder
from harnessx.bundles import context

class CostGuard(MultiHookProcessor):
    async def on_before_model(self, event):
        if event.cumulative_cost_usd > 1.0:
            raise RuntimeError("budget exceeded")
        yield event

config = (HarnessBuilder().add(CostGuard()) | context).build()
```

---

## Tracing and replay

`HarnessJournal` is the default tracer implementation.
It writes session/event/trace/state artifacts under workspace `sessions/` and supports resume via session state checkpoints.

For details:

- [Processors](processors.md)
- [Trajectory](trajectory.md)
- [Composable Harness Guide](../guide/composable_harness.md)
