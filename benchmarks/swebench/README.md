# recipe/swebench — SWE-Bench

Run SWE-Bench software engineering tasks through HarnessX. The agent is given a GitHub
issue and must produce a git patch that fixes it.

## Prerequisites

**Docker** must be running — SWE-Bench evaluation executes patches inside containers.

```bash
docker info   # verify Docker is running
```

## Install

```bash
pip install 'harnessx'
# pulls in: swebench
```

## Usage

```python
from recipe.swebench import SWEBenchTask, SWEBenchEvaluator
from harnessx import Harness
from harnessx.profiles import BenchRunnerPreset
from harnessx.providers.litellm_provider import LiteLLMProvider
from harnessx.tools.builtin import build_filesystem_tools
from harnessx.workspace.workspace import Workspace
from pathlib import Path

# Each task is one SWE-Bench instance (a GitHub issue + test suite)
task = SWEBenchTask(instance_id="django__django-12345")

# Workspace isolates the agent's file system access to a temp directory
ws = Workspace(root=Path("/tmp/swebench-work"), agent_id=task.instance_id)

config = BenchRunnerPreset.copy(
    model_provider=LiteLLMProvider("claude-sonnet-4-6"),
    evaluator=SWEBenchEvaluator(),
    workspace=ws,
    tool_registry=build_filesystem_tools(ws),
)
result = await Harness(config=config).run(task)
print(result.eval_result.passed)   # True if patch resolves the issue
```

## How evaluation works

1. Agent reads the issue description and explores the codebase with `Bash` / `Read` / `Edit`
2. Agent produces a git diff (written to workspace)
3. `SWEBenchEvaluator` extracts the patch and runs the official SWE-Bench grader inside Docker
4. Returns `passed=True` if all relevant tests pass

## Batch run

```python
import asyncio
from datasets import load_dataset

ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
tasks = [SWEBenchTask(instance_id=row["instance_id"]) for row in ds.select(range(10))]

harness = Harness(config=config)
results = await asyncio.gather(*[harness.run(t) for t in tasks])
passed = sum(r.eval_result.passed for r in results)
print(f"{passed}/{len(tasks)} resolved")
```

## Environment variables

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY
# Docker must be available — no additional env vars needed
```

## Notes

- Each evaluation run pulls/builds a Docker image per instance on first use (~minutes)
- Recommend running on a machine with ≥16 GB RAM and Docker daemon accessible
- Verified split (`SWE-bench_Verified`) is recommended over the full split for faster iteration
