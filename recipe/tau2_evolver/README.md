# tau2-evolver

Automated multi-round optimization of a HarnessX agent's `HarnessConfig` against the
tau2-bench evaluation suite. A **meta-agent** (Claude Opus + extended thinking) reads
failure trajectories from the previous round, authors new processors or guidance
templates, and re-evaluates — all without modifying model weights.

## How it works

```
Round 0  ──  simulate baseline config on task set  ──►  trajectories
Round N  ──  meta-agent reads R(N-1) trajectories        ──►  new config.yaml
         ──  simulate new config on task set             ──►  trajectories
         ──  gating: reject if avg_reward drops > tolerance
```

The meta-agent's intervention is always a HarnessX artifact — a `MultiHookProcessor`
injected via `before_model` hooks, a Jinja guidance template, or a configuration
change. The agent model and its weights are never modified.

### Infrastructure-error guard

If more than 50% of tasks in a round hit `infrastructure_error` (proxy outage,
thread-pool crash, model endpoint timeout), the round is marked **poisoned** and
the experiment aborts rather than feeding garbage trajectories to the meta-agent.

### Gating

After each evolved round, the new config is accepted only if:

```
avg_reward(new) ≥ best_avg_reward_so_far − regression_tolerance
```

A rejected config is silently reverted to the previous best; the next round
evolves from that best config instead.

---

## Files

```
recipe/tau2_evolver/
├── run.py              # multi-round evolver (Python entry point)
├── run.sh              # convenience wrapper for a single multi-round run
├── badcase_iter.sh     # iterative badcase-driven evolution (see below)
├── defaults.py         # default model endpoints, concurrency, budgets
├── skills/
│   └── tau2-playbook   # benchmark-specific skill injected into meta-agent
└── runs/               # experiment outputs (git-ignored)
    └── <run_tag>/
        ├── R0/         # baseline round
        │   ├── config.yaml
        │   ├── report.json
        │   └── trajectories/   # one .md per task (meta-agent reads these)
        ├── R1/
        │   ├── config.yaml     # evolved config for this round
        │   ├── evolve/         # meta-agent workspace
        │   │   ├── config.yaml
        │   │   ├── processors/ # authored MultiHookProcessor files
        │   │   └── templates/  # authored Jinja templates
        │   ├── report.json
        │   └── trajectories/
        └── comparison.json     # per-task rewards across all rounds
```

---

## Single multi-round run (`run.sh`)

Edit `run.sh` to set the models, task IDs, round count, and optional
`--from-report` bootstrap, then run from the HarnessX repo root:

```bash
./recipe/tau2_evolver/run.sh
```

Or call `run.py` directly for full control:

```bash
uv run --project ~/tau2-bench \
    python -m recipe.tau2_evolver.run \
    --domain retail \
    --base-config benchmarks/tau2/harness_config_base.yaml \
    --num-rounds 5 \
    --run-tag my_experiment
```

### Bootstrap from an existing report (`--from-report`)

Skip R0 simulation by loading trajectories from a prior evaluation:

```bash
python -m recipe.tau2_evolver.run \
    --base-config runs/prev_run/R4/config.yaml \
    --from-report runs/prev_run/R4/report.json \
    --num-rounds 3 \
    --run-tag continued_run
```

R0 loads the report and applies `--task-ids` filtering if set;
R1 onward runs real simulations with the evolved config.

### Key parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--domain` | `retail` | tau2 domain (`retail`, `airline`, `telecom`) |
| `--base-config` | `benchmarks/tau2/harness_config.yaml` | Starting `HarnessConfig` |
| `--num-rounds` | `3` | Total rounds including R0 |
| `--run-tag` | `<domain>_<timestamp>` | Output directory name under `runs/` |
| `--clean` | off | Wipe `runs/<run-tag>/` before starting |
| `--from-report` | — | Path to existing tau2 JSON report; skips R0 simulation |
| `--task-ids` | — | Comma-separated task IDs; overrides `--max-tasks` |
| `--max-tasks` | all | Maximum tasks to load from the split |
| `--max-concurrency` | `30` | Parallel simulations per round |
| `--num-trials` | `1` | Repeat each task N times; reward is averaged |
| `--regression-tolerance` | `0.02` | Max allowed reward drop before reverting |
| `--evolve-cost` | `50.0` | USD cap for the meta-agent per evolve call |
| `--evolve-wall-clock` | `3600` | Wall-clock cap (seconds) per evolve call |
| `--agent-model` | see `defaults.py` | LLM for the task agent |
| `--agent-api-base` | see `defaults.py` | Endpoint for the task agent |
| `--user-model` | see `defaults.py` | LLM for the user simulator |
| `--meta-model` | `anthropic/…/claude-opus-4-6` | LLM for the meta-agent |

---

## Iterative badcase-driven evolution (`badcase_iter.sh`)

Standard multi-round runs evolve on a fixed task set. This script implements a
**shrinking-frontier** strategy: each iteration focuses exclusively on the tasks
that the previous config could not solve.

### Algorithm

```
Iter 1:  evolve on full task set  →  C1,  collect B1 = {tasks with reward < 1.0}
Iter 2:  evolve on B1 from C1     →  C2,  collect B2 = {tasks in B1 still failing}
Iter k:  evolve on B_{k-1} from C_{k-1}  →  C_k,  B_k = B_{k-1} ∩ {still failing}
Stop:    B_k = ∅  or  k > max-iter
```

Each `Ck` config is specialized to fix the failures that `C_{k-1}` left behind.
The chain `[C1, C2, …, Cn]` together resolves more badcases than any single run
could, because each meta-agent call sees only the hard residual failures — not
the easy wins that earlier configs already handle.

### Internal mapping to `run.py`

Each iteration calls `run.py --num-rounds 2`:

| `run.py` round | Role |
|----------------|------|
| R0 | Load `C_{k-1}`'s report filtered to `B_{k-1}` (no re-simulation) |
| evolve | Meta-agent reads only `B_{k-1}` failure trajectories → authors `C_k` |
| R1 | Simulate `C_k` on `B_{k-1}` → collect `B_k` |

`--from-report` reuses the previous R1 report as R0 input, so no extra simulation
is needed to establish the baseline for each iteration.

### Usage

```bash
# defaults: retail domain, harness_config_base.yaml baseline, max 10 iterations
./recipe/tau2_evolver/badcase_iter.sh

# custom prefix and iteration cap
./recipe/tau2_evolver/badcase_iter.sh --run-prefix retail_full_0429 --max-iter 5

# pass extra flags through to run.py
./recipe/tau2_evolver/badcase_iter.sh --max-concurrency 20 --agent-temperature 0
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--run-prefix` | `badcase_iter` | Prefix for per-iteration `run-tag`s; metadata written to `runs/<prefix>_meta/` |
| `--max-iter` | `10` | Hard upper bound on iterations |
| `--base-config` | `benchmarks/tau2/harness_config_base.yaml` | Baseline config for iter 1 |
| `--domain` | `retail` | tau2 domain |
| any other flag | — | Forwarded to every `run.py` call |

### Outputs

```
runs/<run-prefix>_meta/
├── run.log          # timestamped execution log
├── summary.md       # per-iteration table: tasks / avg_reward / badcase count / fixed count
├── fixes.md         # which task IDs each Ck fixed
├── iter1.log        # raw run.py output for iteration 1
├── iter2.log        # …
└── …

runs/<run-prefix>_iter1/   # standard run.py output tree for iteration 1
runs/<run-prefix>_iter2/   # …
```

`summary.md` example (based on the retail experiment, 2026-04-29,
114 tasks, agent: Qwen3.5-27B):

| iter | run_tag | test set | badcases (B_k) | fixed this iter | cumulative fixed | full-set score (est.) |
|------|---------|----------|---------------|-----------------|-----------------|----------------------|
| iter1 | `retail_iter1` | 114 (full) | 22 | — | 0 | 0.8070 (92/114) |
| iter2 | `retail_iter2` | 22 (B1) | 11 | 11 (4,6,20,31,33,45,49,60,63,93,104) | 11 | ~0.904 (103/114) |
| iter3 | `retail_iter3` | 11 (B2) | 7 | 4 (32,39,80,110) | 15 | ~0.939 (107/114) |
| iter4 | `retail_iter4` | 7 (B3) | 5 | 2 (22,38) | 17 | ~0.956 (109/114) |
| iter5 | `retail_iter5` | 5 (B4) | 5 | 0 | 17 | ~0.956 (109/114) |
| iter6 | `retail_iter6` | 5 (B4) | 4 | 1 (41) | 18 | ~0.965 (110/114) |
| iter7–10 | `retail_iter7–10` | 4 (B6) | 4 | 0 | 18 | ~0.965 (110/114) |

The key columns to watch are **badcases (B_k)** shrinking each round and
**full-set score** climbing. Each `Ck` is only evaluated on `B_{k-1}` (the
previous round's failures), so the per-iteration `avg_reward` is not shown —
it measures performance on a different subset each time and is not comparable
across iterations. Full-set score is estimated as `(original_passing +
cumulative_fixed) / total`, assuming `Ck` does not regress on already-passing
tasks (verify by re-running on the full set if needed).

Outcome: 22 badcases → 18 fixed (81.8%); 4 tasks remain unsolved (79, 100, 105, 112).

---

## What the meta-agent produces

The meta-agent writes its artifacts under `runs/<run_tag>/R{N}/evolve/`:

| Artifact | Description |
|----------|-------------|
| `config.yaml` | The evolved `HarnessConfig` (always required) |
| `processors/<name>.py` | New `MultiHookProcessor` — e.g. `before_model` hook that injects a policy reminder when a task pattern is detected |
| `templates/<name>.j2` | Jinja guidance template extending the system prompt |

The processor pattern used in retail experiments (called **IRMA** — Intervention via
Runtime Message Augmentation) injects context-aware `[POLICY REMINDER]` blocks
before each model call. Examples of interventions that improved retail scores:

- *"STOP. You have not checked all orders yet."* — prevents premature action
- *"Modify address FIRST, then items."* — enforces correct write order
- *"ATTEMPT THE ACTION ANYWAY."* — overrides over-conservative refusals

These are not hard-coded rules; the meta-agent infers them from the failure
trajectories and writes them as runtime processor logic.
