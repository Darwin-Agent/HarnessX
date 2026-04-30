# Quickstart

## Install

```bash
pip install harnessx
# Development install
pip install -e ".[dev]"
```

After installation, both `hx` and `harnessx` commands are available.

## Configure model access

HarnessX resolves model configuration in this order:

1. `~/.harnessx/model_config.yaml`
2. environment variables (`ANTHROPIC_*`, `OPENAI_*`, `LITELLM_*`)

Common env setup:

```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export OPENAI_API_KEY=sk-...

# OpenAI-compatible local endpoint
export OPENAI_API_BASE=http://127.0.0.1:8061/v1
export OPENAI_API_KEY=EMPTY
```

## CLI

Interactive mode (default):

```bash
hx                                                    # start interactive session
hx "Write a Python script that counts words"         # run one task, then stay interactive
hx -v                                                 # verbose logs
hx --resume <run_id>                                  # resume a previous session
```

Non-interactive mode (print and exit):

```bash
hx -p "Write a fizzbuzz"
```

Harness Lab UI:

```bash
cd frontend && npm install && npm run build && cd ..
hx lab                                                # opens http://localhost:7861
hx lab --port 8080
```

## CLI flags reference

```
hx [OPTIONS] [PROMPT]

  -p, --print          Non-interactive: print response and exit
  --max-steps N        Max agent steps (default: 30)
  -v, --verbose        Show structured logs
  --resume RUN_ID      Resume a previous session
  --version
```

## First Python script

```python
import asyncio
from harnessx import BaseTask, HarnessConfig
from harnessx.core.model_config import ModelConfig
from harnessx.providers.litellm_provider import LiteLLMProvider

async def main():
    model = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6"))
    harness = model.agentic(HarnessConfig())
    result = await harness.run(BaseTask(description="What is 2 + 2?"))
    print(result.final_output)

asyncio.run(main())
```

## Use an example harness

```python
from examples.coding.harness import build_coding
from harnessx.core.model_config import ModelConfig
from harnessx.providers.anthropic_provider import AnthropicProvider

model = ModelConfig(main=AnthropicProvider("claude-sonnet-4-6"))
harness = model.agentic(build_coding().build())
```

## Custom tool + processor

```python
from harnessx import MultiHookProcessor
from harnessx.tools.base import tool
from harnessx.tools.inmemory import InMemoryToolRegistry
from harnessx.bundles import context, coding
from harnessx.core.builder import HarnessBuilder
from harnessx.core.model_config import ModelConfig
from harnessx.providers.litellm_provider import LiteLLMProvider

@tool(description="Query the internal knowledge base")
async def kb_search(query: str) -> str:
    return f"results for: {query}"

registry = InMemoryToolRegistry()
registry.register(kb_search)

class CostGuard(MultiHookProcessor):
    async def on_before_model(self, event):
        if event.cumulative_cost_usd >= 1.0:
            raise RuntimeError("cost limit reached")
        yield event

config = (
    HarnessBuilder()
    .slot(tool_registry=registry)
    .add(CostGuard())
    | context | coding
).build()

harness = ModelConfig(main=LiteLLMProvider("claude-sonnet-4-6")).agentic(config)
```

Further reading:
- [Composable Harness](composable_harness.md)
- [Tools](tools.md)
