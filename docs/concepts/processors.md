# Processor Architecture

Processors are the behavior units in HarnessX.
Each processor listens to one or more runtime events and can pass through, modify, inject, or block events.

---

## Event hooks

RunLoop dispatches events in this order:

1. `task_start`
2. `step_start`
3. `before_model`
4. `after_model`
5. `before_tool`
6. `after_tool`
7. `step_end`
8. `task_end`

A processor can subscribe to all hooks (`"*"`) or specific hook keys.

---

## MultiHookProcessor

Most processors should subclass `MultiHookProcessor`.

```python
from harnessx import MultiHookProcessor

class MyGuard(MultiHookProcessor):
    _singleton_group = "my_guard"
    _order = 50

    async def on_before_model(self, event):
        # inspect or modify event
        yield event
```

Key conventions:

- `_singleton_group`: merge-time conflict group
- `_order`: execution order inside the same hook
- `_after`: soft ordering dependencies by singleton group

`MultiHookProcessor` also supports `@on(EventClass)` for custom handler names.

---

## Processor composition

`HarnessBuilder.add(proc)` registers processors.
`HarnessBuilder` instances compose with `|`.

```python
from harnessx.core.builder import HarnessBuilder
from harnessx.bundles import context, control

config = (
    HarnessBuilder()
    | context
    | control
).build()
```

Merge-time checks include:

- singleton group conflicts
- ordering contradictions/cycles

---

## Built-in processor families

### Context

- `SystemPromptProcessor`
- `UserWrapperProcessor`
- `EnvironmentContextInjector`

### Memory

- `MemoryRetrievalProcessor`
- `MemoryExtractionProcessor`

### Control / reliability

- `ToolCallCorrectionLayer`
- `ParseRetryProcessor`
- `LoopDetectionProcessor`
- `SelfVerifyProcessor`
- `ToolFailureGuard`
- `TokenBudgetProcessor`
- `CompactionProcessor`

### Observability

- `CheckpointProcessor`
- `OTelProcessor`
- `HarnessJournal` (tracer, not processor) records event/trace streams

### Evaluation

- `EvaluationProcessor` with evaluators such as `SelfVerifyEvaluator`, `LLMJudgeEvaluator`
- PRM strategies (`TerminalPRM`, `DiscountedPRM`, `ToolSuccessPRM`, `LLMJudgePRM`)

### Multi-agent

- Use the `spawn_subagent` tool (`harnessx/tools/spawn_subagent.py`); multi-agent routing is tool-driven, not a processor.

---

## Trigger trace

When a processor modifies an event, `ProcessorChain` automatically detects the change
and emits a `ProcessorTriggerEvent` on behalf of that processor.
Processors do not emit `ProcessorTriggerEvent` manually.
Triggers are written to the trace JSONL by `HarnessJournal` and can be used for debugging, replay analysis, and training diagnostics.

---

## Plugin integration

Plugins can register processors via `plugin.json`:

```json
{
  "name": "my-plugin",
  "processors": [
    { "target": "my_plugin.processors.RecallProcessor", "top_k": 5 }
  ]
}
```

Each `target` points to a `MultiHookProcessor` subclass; other fields are constructor kwargs.

See [Plugin System](../feats/plugins.md).
