# HarnessX — Agent Implementation Guide

This file is for AI coding assistants (Claude Code, Cursor, Copilot, etc.) implementing
features with HarnessX. Read this before generating any code.

---

## What HarnessX is

A composable Python harness for running LLM agents and collecting training trajectories.
Core loop: `Harness.run(task)` → `run_loop()` → `StatefulTrajectory` + `HarnessResult`.

```
User request
    │
    ▼
Harness.run(BaseTask)
    │
    ▼
run_loop()
    ├── Processors assemble context → messages sent to model
    ├── provider.complete() → ModelResponseEvent
    ├── tool_registry.execute() → ToolResultEvent
    └── EvaluationProcessor → backfill trajectory rewards
    │
    ▼
HarnessResult
    ├── .final_output    (str)
    ├── .exit_reason     ("done" | "budget_exceeded" | "loop_detected" | "error")
    ├── .trajectory      (StatefulTrajectory — all steps with rewards)
    └── .eval_result     (EvalResult — passed, score, reward)
```

---

## Canonical imports

```python
from harnessx import Harness, BaseTask, HarnessConfig, MultiHookProcessor
from harnessx.core.model_config import ModelConfig
from harnessx.providers.litellm_provider import LiteLLMProvider
from harnessx.providers.anthropic_provider import AnthropicProvider
from harnessx.tools.inmemory import InMemoryToolRegistry
from harnessx.tools.base import tool
from harnessx.core.builder import HarnessBuilder
from harnessx.bundles.context import make_context
from harnessx.bundles import context, coding, control
from harnessx.processors.memory.strategies.sliding_window import SlidingWindowMemory
from harnessx.processors.control.cost_guard import CostGuardProcessor
from harnessx.processors.control.loop_detection import LoopDetectionProcessor
from harnessx.tracing.journal import HarnessJournal
from harnessx.workspace.workspace import Workspace
```

---

## Two composition points

HarnessX splits configuration into two independent objects:

- **`HarnessConfig`** — behavior pipeline (tools, workspace, processors). **No model.**
- **`ModelConfig`** — model binding (which provider to call, optional named slots).

```python
# Build the behavior pipeline
harness_config = (HarnessBuilder() | context | coding).build()

# Bind a model and get a runnable Harness
model = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6"))
harness = model.agentic(harness_config)

# Shorthand: provider.agentic(config)
harness = LiteLLMProvider("claude-sonnet-4-6").agentic(harness_config)
```

**Never** pass `model_provider` to `HarnessConfig` — that field does not exist.

---

## Running a task

```python
harness = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6")).agentic(harness_config)

result = await harness.run(BaseTask(
    description="Write a hello-world Flask app",
    success_criteria="contains app.run()",
    max_steps=20,
    token_budget=80_000,
    max_cost_usd=0.50,
))

result.final_output       # str
result.exit_reason        # "done" | "budget_exceeded" | "loop_detected" | "error"
result.trajectory         # StatefulTrajectory
```

---

## Common implementation patterns

### Pattern 1: Custom tool

```python
from harnessx.tools.base import tool
from harnessx.tools.inmemory import InMemoryToolRegistry

@tool(description="Query the internal knowledge base")
async def kb_search(query: str, top_k: int = 5) -> str:
    results = await my_vector_db.search(query, k=top_k)
    return "\n".join(r.text for r in results)

registry = InMemoryToolRegistry()
registry.register(kb_search)

harness_config = HarnessBuilder().slot(tool_registry=registry).build()
harness = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6")).agentic(harness_config)
```

### Pattern 2: Custom processor (single hook)

```python
from harnessx.core.processor import before_tool

@before_tool
async def audit(event):
    await audit_log.write(event.tool_name, event.tool_input)
    yield event

harness_config = (
    HarnessBuilder()
    .add(audit)
    | context
).build()
```

### Pattern 3: Multi-hook processor (register under `"*"`)

One object, multiple hooks, one registration:

```python
from harnessx import MultiHookProcessor

class LifecycleLogger(MultiHookProcessor):
    async def on_before_model(self, event):
        print(f"→ model step={event.step_id}")
        yield event

    async def on_step_end(self, event):
        print(f"← step={event.step_id} tokens={event.cumulative_tokens}")
        yield event

    async def on_task_end(self, event):
        print(f"✓ exit={event.exit_reason} total={event.total_tokens}")
        yield event

harness_config = (
    HarnessBuilder()
    .add(LifecycleLogger())
    | context
).build()
```

### Pattern 4: Custom memory backend

Implement the `BaseMemory` protocol:

```python
from harnessx.processors.memory.strategies.base import BaseMemory
from harnessx.core.events import Message

class PostgresMemory(BaseMemory):
    async def add(self, messages: list[Message]) -> None: ...
    async def retrieve(self, query: str, k: int = 10) -> list[Message]: ...
    async def compress(self, messages: list[Message], budget: int) -> list[Message]:
        return messages[-budget:]
    async def persist(self) -> None: ...
    async def load(self, run_id: str) -> list[Message]: ...

harness_config = (
    HarnessBuilder()
    | make_context(memory=PostgresMemory())
).build()
```

### Pattern 5: Interrupt / resume (human-in-the-loop)

```python
task = BaseTask(
    description="Send a status update to the team Slack",
    interrupt_on=["send_message"],     # pause BEFORE executing this tool
)
result = await harness.run(task)

if result.is_interrupted:
    tc = result.interrupted_at         # ToolCall
    print(f"About to call: {tc.name}({tc.input})")
    if input("Approve? [y/n] ") == "y":
        result2 = await harness.run(task, resume_state=result.resume_state)
```

### Pattern 6: Workspace isolation

```python
from harnessx.workspace.workspace import Workspace
from harnessx.tools.builtin import build_default_tools
from pathlib import Path

ws = Workspace(root=Path("/tmp/agent-work"), agent_id="demo", mode="isolated")
harness_config = (
    HarnessBuilder()
    .slot(workspace=ws, tool_registry=build_default_tools())
    | context
).build()
harness = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6")).agentic(harness_config)
```

### Pattern 7: Collect training data

```python
result = await harness.run(task)
if result.eval_result and result.eval_result.passed:
    records = result.trajectory.to_training_records()
    # records: list[dict] — one per step, OpenAI chat format + reward
```

### Pattern 8: Multi-turn chat

```python
model = ModelConfig(main=LiteLLMProvider("openai/gpt-4o"))
harness = model.agentic((HarnessBuilder() | context).build())

while True:
    msg = input("> ").strip()
    if not msg:
        break
    result = await harness.run(BaseTask(description=msg))
    print(result.final_output)
```

### Pattern 9: Custom context assembly

Use `make_context()` to compose context processors:

```python
from harnessx.bundles.context import make_context
from harnessx.processors.context.strategies.system_prompt.null import NullSystemPromptBuilder
from harnessx.processors.memory.strategies.sliding_window import SlidingWindowMemory

harness_config = (
    HarnessBuilder()
    | make_context(
        system_builder=NullSystemPromptBuilder(),
        memory=SlidingWindowMemory(n=20),
    )
).build()
```

---

## Benchmark task adapter (recipe/)

Third-party benchmark libraries go in `recipe/`, never in `harnessx/`.
Use absolute imports inside recipe files:

```python
# recipe/my_bench/harness.py
from harnessx import BaseTask, HarnessConfig
from harnessx.core.builder import HarnessBuilder

def make_my_bench_harness() -> HarnessConfig:
    from harnessx.bundles import context, coding
    return (HarnessBuilder() | context | coding).build()
```

```python
# recipe/my_bench/runner.py
from harnessx.core.model_config import ModelConfig
from harnessx.providers.litellm_provider import LiteLLMProvider
from .harness import make_my_bench_harness

model = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6"))
harness = model.agentic(make_my_bench_harness())
```

---

## Key files to read when implementing features

| Feature | Read these files |
|---------|-----------------|
| Custom tool | `harnessx/tools/base.py`, `harnessx/tools/inmemory.py` |
| Custom processor (single hook) | `harnessx/core/processor.py` (hook decorators) |
| Multi-hook processor | `harnessx/core/processor.py` (`MultiHookProcessor`) |
| Custom memory | `harnessx/processors/memory/strategies/base.py` |
| Context assembly | `harnessx/bundles/context.py`, `harnessx/processors/context/` |
| ModelConfig | `harnessx/core/model_config.py` |
| HarnessBuilder | `harnessx/core/builder.py` |
| Training data | `harnessx/core/trajectory.py` (`StatefulTrajectory.to_training_records`) |
| Multi-agent | `harnessx/tools/spawn_subagent.py`, `harnessx/workspace/factory.py` |
| Run loop internals | `harnessx/core/runloop.py` |
| CLI extension | `harnessx/cli.py` |
| RL format integration | `recipe/slime/` |

---

## Constraints and rules

1. **Core never imports third-party benchmark libs.** Benchmark adapters belong in
   `recipe/<name>/`. Core (`harnessx/`) may only import from stdlib and its own package.

2. **Model goes in `ModelConfig`, not `HarnessConfig`.** `HarnessConfig` has no
   `model_provider` field. Always build with `ModelConfig(main=...).agentic(harness_config)`.

3. **Tool names are PascalCase** — `Bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep`,
   `WebSearch`, `WebFetch`, `Browser`. Match exactly when referencing in allowlists, prompts,
   or `interrupt_on`.

4. **`finish_reason`** from OpenAI-compatible models is `"stop"`. From Anthropic it is
   `"end_turn"`. The RunLoop handles both — don't add provider-specific checks elsewhere.

5. **`HarnessJournal` writes JSONL traces** on each run. Pass `silent=True` to suppress
   console output. Use `harness_config.copy(tracer=HarnessJournal())` to wire it in.

6. **Model defaults come from model config/env vars.** CLI resolves model config from
   config YAML, `~/.harnessx/model_config.yaml`, then provider env vars
   (`ANTHROPIC_*`, `OPENAI_*`, `LITELLM_*`).

7. **Processors are ordered** within each hook. Earlier processors run first. A processor
   that yields nothing blocks all subsequent processors and the hook action itself.

8. **`state.slots`** is a free-form key-value store for agent-specific runtime state.
   Use `state.set_slot(key, slot_type, content)` and `state.get_slot(key)`.

---

## What NOT to do

- Do not pass `model_provider` to `HarnessConfig` or `HarnessConfig.copy()` — the field
  does not exist. Use `ModelConfig(main=provider).agentic(harness_config)`.
- Do not `from harnessx.presets import ...` — the presets module is empty. Use
  `HarnessBuilder() | context | coding` or `HarnessConfig.from_yaml(yaml_str)`.
- Do not call `harness_config.copy(processors={"key": [proc]})` to *append* processors —
  this replaces the entire hook list. Use `{**harness_config.processors, "key": [...existing, proc]}`.
- Do not register the same `MultiHookProcessor` instance under multiple hook keys — register it
  once under `"*"` and let it dispatch internally via `on_*` methods.
- Do not access `result.task_end.messages` — messages live on `result.trajectory.steps`.
- Do not add `finish_reason == "end_turn"` checks in new code — use
  `finish_reason in ("end_turn", "stop")` to support both Anthropic and OpenAI models.
- Do not write documentation files unless asked.
- Do not add type annotations or docstrings to code you didn't change.
