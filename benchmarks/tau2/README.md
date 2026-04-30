# TAU2-Bench

Integration of [tau2-bench](https://github.com/sierra-research/tau2-bench) (Sierra Research) into HarnessX.

TAU2-Bench is a **multi-party simulation evaluation framework** for assessing AI agents in customer-service
scenarios. Unlike simple QA benchmarks, it runs multi-turn dialogues between the agent and an LLM-driven
simulated user. The agent must call domain-specific tools (look up flights, modify orders, etc.) and is
scored on **database state correctness + action correctness + communication quality**.

## Architecture

```
tau2 Orchestrator (drives the simulation)
├── Simulated user        (tau2 built-in, LLM-driven)
├── Environment + tools   (tau2 built-in, domain-specific)
├── Evaluator             (tau2 built-in: DB check, action check, communication check)
└── Agent ← HarnessXAgent (adapter provided by this recipe)
       └── Harness Pipeline (harness.py, built with HarnessBuilder)
              ├── ContextAssemblyProcessor (context assembly + memory management)
              │    ├── NullSystemPromptBuilder (tau2 manages its own system prompt)
              │    └── SlidingWindowMemory
              └── AnthropicProvider / LiteLLMProvider
```

tau2 controls the full simulation loop. This adapter builds a complete Harness pipeline via
**HarnessBuilder**, wiring `ContextAssemblyProcessor` (context assembly, memory, history truncation)
and the model provider so every evaluation pass goes through HarnessX core infrastructure.

## File structure

```
benchmarks/tau2/
├── __init__.py       # exports: HarnessXAgent, Tau2Task, Tau2Evaluator, make_tau2_harness
├── harness.py        # HarnessBuilder config: provider + ContextAssemblyProcessor
├── agent.py          # HalfDuplexAgent adapter — routes LLM calls through the Harness pipeline
├── task.py           # Tau2Task(BaseTask) + Tau2Evaluator
├── stop_guard.py     # StopGuardUserSimulator — strips premature ###STOP### from confirmations
├── test_tau2.py      # evaluation runner (imports / conversion / loading / registry / pipeline / LLM / e2e)
└── README.md
```

## Setup

### Prerequisites

- **Python >= 3.12** (hard requirement from tau2-bench)
- [uv](https://docs.astral.sh/uv/) package manager
- LLM API credentials for both the agent model and the simulated user model

### Step 1 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2 — Clone and install tau2-bench

```bash
git clone https://github.com/sierra-research/tau2-bench ~/tau2-bench
cd ~/tau2-bench
uv sync    # downloads Python 3.12 if needed and installs all dependencies
```

### Step 3 — Install harnessx into the tau2 virtual environment

```bash
uv pip install --python ~/tau2-bench/.venv/bin/python -e /path/to/HarnessX
```

### Step 4 — Set API credentials

```bash
# Agent model endpoint (Anthropic-compatible, required for extended thinking)
export ANTHROPIC_BASE_URL=https://...
export ANTHROPIC_API_KEY=sk-...

# User simulator endpoint (OpenAI-compatible)
export OPENAI_API_BASE=https://...
export OPENAI_API_KEY=sk-...
```

> **Why does this depend on `~/tau2-bench/`?**
> tau2-bench is not on PyPI and must be installed from source. Its evaluation data (task definitions,
> domain databases, policy documents) lives under the repository's `data/` directory and is read at
> runtime. This is the same pattern used by swebench and terminal_bench.

## Running evaluations

All commands must run inside the **tau2-bench virtual environment** because `tau2` is not on PyPI.
Set `HARNESSXDIR` once, then use `uv run --project ~/tau2-bench` so the tau2-bench venv is used
while `PYTHONPATH` makes the HarnessX package importable.

### Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_BASE_URL` | Anthropic-compatible endpoint (used by agent with extended thinking) |
| `ANTHROPIC_API_KEY` | API key for the Anthropic endpoint |
| `OPENAI_API_BASE` | OpenAI-compatible endpoint (used by user simulator and judge) |
| `OPENAI_API_KEY` | API key for the OpenAI endpoint |
| `TAU2_WORKERS` | Number of parallel workers (default: 1) |
| `TAU2_STOP_GUARD` | Set to `1` to enable StopGuard (default: off) |

### retail (114 tasks)

```bash
uv run --project ~/tau2-bench \
  python "$HARNESSXDIR/benchmarks/tau2/test_tau2.py" \
  --domain retail \
  --agent-model anthropic/YOUR_PROVIDER/claude-sonnet-4-5 \
  --user-model  openai/azure_openai/gpt-5.2 \
  --agent-api-base "$ANTHROPIC_BASE_URL" \
  --api-base       "$OPENAI_API_BASE" \
  --extended-thinking --thinking-budget-tokens 62976 \
  --workers 4 \
  --stop-guard \
  --report "$HARNESSXDIR/reports/retail_sonnet45_ET_stop_guard.json"
```

### airline (50 tasks)

```bash
uv run --project ~/tau2-bench \
  python "$HARNESSXDIR/benchmarks/tau2/test_tau2.py" \
  --domain airline \
  --agent-model anthropic/YOUR_PROVIDER/claude-sonnet-4-5 \
  --user-model  openai/azure_openai/gpt-5.2 \
  --agent-api-base "$ANTHROPIC_BASE_URL" \
  --api-base       "$OPENAI_API_BASE" \
  --extended-thinking --thinking-budget-tokens 62976 \
  --workers 4 \
  --stop-guard \
  --report "$HARNESSXDIR/reports/airline_sonnet45_ET_stop_guard.json"
```

### telecom (base split, ~114 tasks)

The telecom domain has 2285 tasks in total. Use the `base` split defined in `split_tasks.json` for
a reproducible evaluation subset.

```bash
uv run --project ~/tau2-bench \
  python "$HARNESSXDIR/benchmarks/tau2/test_tau2.py" \
  --domain telecom \
  --agent-model anthropic/YOUR_PROVIDER/claude-sonnet-4-5 \
  --user-model  openai/azure_openai/gpt-5.2 \
  --agent-api-base "$ANTHROPIC_BASE_URL" \
  --api-base       "$OPENAI_API_BASE" \
  --extended-thinking --thinking-budget-tokens 62976 \
  --workers 4 \
  --task-ids $(python3 -c "import json,os; print(' '.join(json.load(open(os.path.expanduser('~/tau2-bench/data/tau2/domains/telecom/split_tasks.json')))['base']))") \
  --report "$HARNESSXDIR/reports/telecom_sonnet45_ET.json"
```

### Quick smoke test (no LLM required)

```bash
uv run --project ~/tau2-bench \
  python "$HARNESSXDIR/benchmarks/tau2/test_tau2.py" --skip-llm
```

Expected:

```
Step 1: PASS  tau2 + harnessx full import
Step 2: PASS  bidirectional message conversion
Step 3: PASS  task loading per domain
Step 4: PASS  registry + build_agent + isinstance + state init
Step 5: PASS  HarnessBuilder + ContextAssembly + agent integration
Step 6: SKIP  LLM API — --skip-llm
Step 7: SKIP  end-to-end simulation — --skip-llm

Result: 5 passed, 0 failed, 2 skipped  (7 total)
```

## Iterative optimization with tau2-evolver

HarnessX ships a **meta-harness evolver** that automatically improves the agent's
`HarnessConfig` by analyzing failure trajectories and generating new processors and
guidance templates — without touching model weights.

### Single-run evolution

```bash
./recipe/tau2_evolver/run.sh          # edit the file to set models, task IDs, and rounds
```

Each call runs N rounds. Round 0 simulates the baseline; each subsequent round has the
meta-agent (Claude Opus + extended thinking) read the previous round's trajectories,
author a new processor or template, and simulate again. A gating check discards
regressions automatically.

### Iterative badcase-driven evolution

```bash
./recipe/tau2_evolver/badcase_iter.sh [--run-prefix NAME] [--max-iter N]
```

This script implements a focused loop:

1. **Iter 1** — evolve on the full eval set → get config `C1`; collect failures `B1`
2. **Iter k** (k ≥ 2) — evolve only on `B_{k-1}` starting from `C_{k-1}` → get `C_k`; collect `B_k`
3. Stop when `B_k` is empty or `--max-iter` is reached

Each iteration produces one evolved config. The final output is a chain of configs
`[C1, C2, …, Cn]` where each `Ck` is specialized to fix the tasks that `C_{k-1}` could not.

→ Full documentation: [`recipe/tau2_evolver/README.md`](../../recipe/tau2_evolver/README.md)

## Benchmark results

Evaluated with **Sonnet 4.5 + Extended Thinking** (thinking budget: 62976 tokens),
simulated user: GPT-5.2. The *w/o HarnessX* column shows the published tau2 leaderboard
score for the same model ([source](https://taubench.com/#trajectory-visualizer?model=claude-sonnet-4-5_sierra_2026-02-26&domain=telecom)).

| Configuration | Domain | Pass | Total | Pass rate | w/o HarnessX |
|---|---|---:|---:|---:|---:|
| Sonnet 4.5 + ET | **telecom** | 98 | 114 | **86.0%** | 84.9% |
| Sonnet 4.5 + ET | **airline** | 37 | 50 | **74.0%** | 72.0% |
| Sonnet 4.5 + ET + StopGuard | **retail** | 81 | 114 | **71.1%** | 72.4% |

**StopGuard** (`--stop-guard`) addresses a known issue where GPT-based user simulators
send `###STOP###` in the same message as a plan confirmation (`"Yes, please proceed.###STOP###"`).
tau2's orchestrator terminates immediately on `###STOP###`, before the agent executes the write
tool, causing the task to fail despite correct agent behaviour. StopGuard strips premature stops
from affirmative messages, recovering ~8 pp on retail.

## CLI reference

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--domain` | `TAU2_DOMAIN` | `mock` | Evaluation domain |
| `--num-tasks` | `TAU2_NUM_TASKS` | `2` | Number of tasks (ignored if `--task-ids` is set) |
| `--task-ids` | — | — | Explicit list of task IDs (overrides `--num-tasks`) |
| `--agent-model` | `TAU2_AGENT_MODEL` | `anthropic/claude-haiku-4-5-20251001` | Agent LLM (LiteLLM format) |
| `--user-model` | `TAU2_USER_MODEL` | `openai/gpt-4.1` | Simulated user LLM |
| `--api-base` | `TAU2_API_BASE` / `OPENAI_API_BASE` | — | Default API endpoint (user + judge) |
| `--agent-api-base` | `TAU2_AGENT_API_BASE` | — | Dedicated agent endpoint (e.g. Anthropic for ET) |
| `--judge-model` | `TAU2_JUDGE_MODEL` | same as user | LLM for NL assertions judge |
| `--judge-api-base` | `TAU2_JUDGE_API_BASE` | same as `--api-base` | Endpoint for judge |
| `--extended-thinking` | — | off | Enable Extended Thinking (requires `anthropic/` model prefix) |
| `--thinking-budget-tokens` | — | `8000` | Extended Thinking token budget |
| `--workers` | `TAU2_WORKERS` | `1` | Parallel simulation workers |
| `--stop-guard` | `TAU2_STOP_GUARD` | off | Strip premature `###STOP###` from confirmations |
| `--num-trials` | — | `1` | Repeat each task N times; reward is averaged |
| `--report` | — | — | Save evaluation report to a JSON file |
| `--skip-llm` | — | — | Skip LLM-dependent steps (smoke test only) |

## Programmatic usage

### Option A — via tau2 runner (recommended)

```python
from tau2.registry import registry
from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import run_single_task
from tau2.runner.helpers import get_tasks
from tau2.evaluator.evaluator import EvaluationType
from benchmarks.tau2.agent import create_harnessx_agent
import os

registry.register_agent_factory(create_harnessx_agent, "harnessx")

tasks = get_tasks("airline", num_tasks=5)

config = TextRunConfig(
    domain="airline",
    agent="harnessx",
    llm_agent="anthropic/claude-sonnet-4-5-20250929",
    llm_args_agent={"api_base": os.environ["ANTHROPIC_BASE_URL"]},
    llm_user="openai/gpt-5.2",
    llm_args_user={"api_base": os.environ["OPENAI_API_BASE"]},
    num_trials=1,
    max_steps=30,
)

for task in tasks:
    result = run_single_task(config, task, seed=42, evaluation_type=EvaluationType.ALL)
    print(f"{task.id}: reward={result.reward_info.reward:.3f}")
```

### Option B — manual agent construction

```python
import os
from tau2.runner.build import build_environment, build_user
from tau2.runner.helpers import get_tasks
from tau2.runner.simulation import run_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.evaluator.evaluator import EvaluationType
from benchmarks.tau2.agent import HarnessXAgent

env = build_environment("airline")
task = get_tasks("airline", num_tasks=1)[0]

agent = HarnessXAgent(
    tools=env.get_tools(),
    domain_policy=env.get_policy(),
    model="anthropic/claude-sonnet-4-5-20250929",
    api_base=os.environ["ANTHROPIC_BASE_URL"],
    extended_thinking=True,
    thinking_budget_tokens=62976,
)

user = build_user(
    "user_simulator", env, task,
    llm="openai/gpt-5.2",
    llm_args={"api_base": os.environ["OPENAI_API_BASE"]},
)

orchestrator = Orchestrator(
    domain="airline", agent=agent, user=user,
    environment=env, task=task, max_steps=30, seed=42,
)

result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
print(f"reward={result.reward_info.reward}")
```

## Domains

| Domain | Tasks | Description |
|---|---|---|
| `mock` | 10 | Lightweight domain for development and smoke tests |
| `airline` | 50 | Flight booking, cancellation, customer service |
| `retail` | 114 | Order management, returns, product queries |
| `telecom` | 2285 | Account management, troubleshooting (use `base` split for eval) |

## Scoring

TAU2 evaluates on multiple dimensions; the final reward is the product of all dimension scores:

| Dimension | What is checked | Example |
|---|---|---|
| **DB check** | Did the agent correctly modify the database state? | Flight cancelled → DB status = `cancelled` |
| **Action check** | Did the agent call the right tools with the right arguments? | Called `cancel_reservation(id="ABC")` |
| **Communication check** | Did the agent communicate appropriately with the user? | Informed user of cancellation policy and refund amount |

`reward = 1.0` means all dimensions passed.
