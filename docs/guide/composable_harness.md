# Composable Harness

HarnessX splits configuration into two independent objects:

- **`HarnessConfig`** — the behavior pipeline (tools, workspace, processors). No model.
- **`ModelConfig`** — the model binding (which provider(s) to call).

Combine them with `model.agentic(harness_config)` to get a runnable `Harness`.

## HarnessConfig slots

```
HarnessConfig
├── tool_registry       — available tools the model can call
├── sandbox_provider    — execution environment for tool calls
├── workspace           — file system boundary for tool execution
├── tracer              — event logging + JSONL export
├── processors          — event middleware hooks (dict[hook_name, list[Processor]])
├── workspace_template  — workspace initialisation template
├── init_workspace      — whether to initialise workspace on run
└── sandbox_hint_id     — stable ID for warm-pool sandbox reuse
```

> **Note**: Memory, evaluation, and context assembly are not direct `HarnessConfig` fields.
> They live inside the **processor pipeline** — `MemoryRetrievalProcessor`,
> `EvaluationProcessor`, `SystemPromptProcessor`, etc. This keeps `HarnessConfig` minimal
> while processors handle all behavioral concerns.

## ModelConfig

Model information lives separately in `ModelConfig`, not in `HarnessConfig`:

```python
from harnessx.core.model_config import ModelConfig
from harnessx.providers.litellm_provider import LiteLLMProvider
from harnessx.providers.anthropic_provider import AnthropicProvider

# Single provider
model = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6"))

# Named slots (judge, compact, …)
model = ModelConfig(main=AnthropicProvider("claude-sonnet-4-6"), judge=AnthropicProvider("claude-opus-4-6"))

# Combine with HarnessConfig
harness = model.agentic(harness_config)
```

`provider.agentic(config)` is shorthand for `ModelConfig(main=provider).agentic(config)`.

## HarnessBuilder (recommended composition API)

`HarnessBuilder` provides a safer composition API than building `HarnessConfig` by hand:
- `MultiHookProcessor` subclasses added via `.add()` auto-register under `"*"`
- `_singleton_group` conflict detection: adding the same group twice raises immediately
- `|` operator merges two builders (capability bundle pattern)

```python
from harnessx import HarnessBuilder
from harnessx.bundles.context import make_context
from harnessx.core.model_config import ModelConfig
from harnessx.providers.litellm_provider import LiteLLMProvider
from harnessx.processors.control.cost_guard import CostGuardProcessor
from harnessx.processors.control.loop_detection import LoopDetectionProcessor

harness_config = (
    HarnessBuilder()
    | make_context()
    .add(CostGuardProcessor(max_usd=1.0))
    .add(LoopDetectionProcessor())
).build()

model = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6"))
harness = model.agentic(harness_config)
```

**Capability bundle** — merge builders with `|`:

```python
from harnessx.processors.control.cost_guard import CostGuardProcessor
from harnessx.processors.tools.tool_whitelist import ToolWhitelistProcessor
from harnessx.processors.observability.otel_proc import OTelProcessor

safety = (
    HarnessBuilder()
    .add(CostGuardProcessor(max_usd=0.5))
    .add(ToolWhitelistProcessor(allow=["Read", "Grep"]))
)
observability = (
    HarnessBuilder()
    .add(OTelProcessor(endpoint="http://otel:4318"))
)

harness_config = (
    HarnessBuilder()
    | make_context()
    | safety
    | observability
).build()

harness = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6")).agentic(harness_config)
```

`.slot()` sets non-processor slots (`tool_registry`, `workspace`, etc.); `.add()` adds processors. See [`docs/concepts/harnesses.md`](../concepts/harnesses.md) for the full three-layer architecture (Processor → Bundle → HarnessConfig).

## Model providers

Any litellm model string works out of the box:

```python
from harnessx.providers.litellm_provider import LiteLLMProvider

# Anthropic
LiteLLMProvider("claude-sonnet-4-6")

# OpenAI
LiteLLMProvider("openai/gpt-4o")

# Custom OpenAI-compatible endpoint — reads OPENAI_API_BASE / OPENAI_API_KEY from env
LiteLLMProvider("openai/my-model")

# Extra HTTP headers (e.g. routing headers for managed inference)
LiteLLMProvider("openai/gpt-4o", extra_headers={"X-Provider-Id": "azure_openai"})

# Pass any litellm kwargs directly
LiteLLMProvider("openai/gpt-4o", temperature=0.2, max_tokens=4096)
```

## Memory backends

Memory is queried once per step (via `MemoryRetrievalProcessor`) and provides long-term,
cross-session recall. It is distinct from in-session context-window safety
handled by `TokenBudgetProcessor` / `CompactionProcessor`.

```python
from harnessx.processors.memory.strategies.sliding_window import SlidingWindowMemory   # default, in-process
from harnessx.processors.memory.strategies.custom import RedisMemory                    # persistent, Redis

from harnessx.bundles.context import make_context

harness_config = (
    HarnessBuilder()
    | make_context(memory=SlidingWindowMemory(n=50))
).build()

harness = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6")).agentic(harness_config)
```

## Context assembly

Context assembly is handled by three atomic processors wired together via `make_context()`:

| Processor | `_order` | Responsibility |
|-----------|----------|----------------|
| `SystemPromptProcessor` | 1 | Build and inject the system prompt |
| `MemoryRetrievalProcessor` | 3 | Retrieve long-term memories (optional) |
| `UserWrapperProcessor` | 5 | Wrap the last user message (XML / CoT) |

Use `make_context()` to compose them:

```python
from harnessx.bundles.context import make_context
from harnessx.processors.context.strategies.system_prompt.null import NullSystemPromptBuilder
from harnessx.processors.memory.strategies.sliding_window import SlidingWindowMemory

# defaults: DefaultSystemPromptBuilder + UserWrapperProcessor
harness_config = (HarnessBuilder() | make_context()).build()

# custom
harness_config = (
    HarnessBuilder()
    | make_context(
        system_builder=NullSystemPromptBuilder(),
        memory=SlidingWindowMemory(n=20),
    )
).build()
```

To show a different set of tools depending on state, pass a `tool_filter`:

```python
from harnessx.processors.tools.strategies.tool_filter import AllowlistToolFilter

harness_config = (
    HarnessBuilder()
    | make_context(tool_filter=AllowlistToolFilter(["Bash", "Read", "Write"]))
).build()
```

## Workspace isolation

A `Workspace` restricts all tool filesystem operations to a directory:

```python
from harnessx.workspace.workspace import Workspace
from harnessx import HarnessConfig
from pathlib import Path

ws = Workspace(root=Path("/tmp/agent-sandbox"), agent_id="main", mode="isolated")
harness_config = HarnessConfig(workspace=ws)
```

Modes:
- `"isolated"` — tool calls cannot escape `root` (raises `WorkspaceEscapeError`)
- `"shared"` — allows access to a parent-shared directory for multi-agent handoff
- `"readonly"` — reads allowed, writes blocked

## Interrupt / resume

Pause the agent at a specific tool call and hand control back to the caller:

```python
task = BaseTask(
    description="Draft an email to the team and send it",
    interrupt_on=["send_email"],   # pause before executing this tool
)
result = await harness.run(task)

if result.is_interrupted:
    # result.interrupted_at is the ToolCall the model was about to make
    # show it to the user, collect approval, then resume:
    user_approved = confirm_with_user(result.interrupted_at)
    if user_approved:
        result2 = await harness.run(task, resume_state=result.resume_state)
```

## Running a task

```python
from harnessx import BaseTask
from harnessx.core.model_config import ModelConfig
from harnessx.providers.litellm_provider import LiteLLMProvider

model = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6"))
harness = model.agentic(harness_config)

result = await harness.run(BaseTask(
    description="Analyse sales_data.csv and produce a summary",
    success_criteria="Contains month-over-month growth figures",
    max_steps=20,
    token_budget=80_000,
    max_cost_usd=0.50,
))

print(result.final_output)          # agent's last response
print(result.exit_reason)           # "done" | "budget_exceeded" | "loop_detected" | "error"
print(result.total_steps)
print(result.total_tokens)
print(result.total_cost_usd)
```

## Multi-turn chat session

```python
model = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6"))
harness = model.agentic(harness_config)

for user_message in conversation:
    result = await harness.run(BaseTask(description=user_message))
    print(result.final_output)
```

Each `harness.run()` call starts a fresh state. To maintain conversation history across turns,
use a `SlidingWindowMemory` (or any persistent memory) — `MemoryRetrievalProcessor` retrieves
it on the next call automatically.
