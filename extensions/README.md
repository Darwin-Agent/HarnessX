# Extensions

Optional capabilities for HarnessX live in this directory.

## What Is Included

- `skills/`: reusable task-specific skills
- `plugins/`: plugin examples and custom plugin implementations

## Skills Provenance

Skills in `extensions/skills/` are based on Anthropic's official skills repository:

- <https://github.com/anthropics/skills/tree/main/skills>

They are integrated here with project-specific adaptation where needed (for example, script layout, examples, and local toolchain compatibility).

## Built-in Agent Behavior

For built-in HarnessX agents, `extensions/skills/` is treated as the baseline skills source.

On workspace initialization, HarnessX copies built-in skills:

- from `extensions/skills/<skill>/`
- to `AGENT_HOME/skills/<skill>/` (shared across agents)

Operational notes:

- this copy runs by default
- copy is idempotent (existing skill directories are not overwritten)
- if `workspace.home` is not set, the fallback target is `workspace.root/skills/`

## Example Plugin: `plugins/workflow`

`extensions/plugins/workflow` is a custom procedural-memory plugin example.

It helps an agent:

1. Run structured workflows with tools:
- `flow`: run an ad-hoc multi-step shell workflow
- `flow_resume`: resume a workflow paused by an approval gate
- `flow_exec`: execute a stored workflow YAML by name

2. Recall similar stored workflows at task start:
- scan `workflow_dir` for workflow YAML files
- match by task description
- inject top candidates into the system prompt

3. Internalize complex completed tasks (optional):
- detect high-complexity tasks (for example, by tool-call threshold)
- optionally run lightweight completion judgment
- extract and save reusable workflow YAML in the background

### Good Fit For

- repeated deploy/ops/troubleshooting procedures
- multi-step shell data collection pipelines
- workflows that require explicit human approval checkpoints

## Quick Start

Wire the plugin into your harness:

```python
from harnessx.bundles.control import make_context, reliability
from harnessx.bundles.window_mgmt import make_window_mgmt
from harnessx.core.builder import HarnessBuilder
from extensions.plugins.workflow import WorkflowPlugin

config = (
    HarnessBuilder()
    | make_context()
    | reliability
    | make_window_mgmt()
).plugin(
    WorkflowPlugin(
        workflow_dir="~/.harnessx/workflows",
        guidance=True,
        recall=True,
        internalize=False,  # disable learning in minimal setup
    )
).build()
```

Verify plugin wiring quickly:

```bash
python -m extensions.plugins.workflow.example
```

The example demonstrates tool registration and basic `flow`, `flow_resume`, and `flow_exec` behavior.
