# TB2 Evolver

An automated meta-harness framework for evolving [Terminal Bench 2 (TB2)](https://github.com/laude-institute/terminal-bench-2) agent configurations.
A meta-agent (Claude Opus 4.6) operates in a closed loop: it reads agent trajectories, diagnoses failure root causes, proposes config improvements, and validates them by re-running the benchmark — repeating for N rounds.

**Task-agent model**: Qwen3.5-27B served via SGLang (OpenAI-compatible endpoint).

---

## How It Works

### The Evolution Loop

```
R0 trajectories (baseline)
        │
        ▼
┌──────────────────────────────────────────┐
│  Meta-Agent (Claude Opus 4.6)            │
│  1. Read current-round trajectories      │
│  2. Diagnose root causes of failures     │
│  3. Hypothesize improvements             │
│  4. Write evolved config.yaml            │
│       ├── system prompt (Jinja2 template)│
│       └── processors (.py classes)       │
└──────────────────────────────────────────┘
        │  evolved config
        ▼
  TB2 eval  (harbor run, N concurrent)
        │  new trajectories + score
        ▼
  Append to learnings.md
        │
        └──→ next round R{n+1}
```

### Two Evolution Levers

| Layer | Mechanism | Examples |
|-------|-----------|---------|
| **Instruction** | Jinja2 system prompt template | Timeout guidance, source-file read-only rules, API usage hints |
| **Control** | Python `MultiHookProcessor` subclass | `SqlReadOnlyGuard`, `PackageEnvGuide`, `SolutionCleanupGuard` |

Control-layer processors hook into `on_before_tool` and can inspect, modify, or block tool calls programmatically. Instruction-layer changes only affect what the agent reads in its system prompt.

### Directory Layout

```
recipe/tb2_evolver/
├── run.py                               # Recipe-level evolution loop entry point
├── .env                                 # Model credentials, API endpoints, concurrency
├── .env.example                         # Template for .env
├── tasks_all_tb2.json                   # Full TB2 task list (89 tasks)
├── tasks_sample16_seed42_act15.json     # Evolution training set (15 tasks)
├── scripts/
│   ├── start_evolve_r0baseline.sh   # Launch / resume an evolution run
│   ├── run_full_eval.sh                 # Evaluate any config on the full 89-task set
│   ├── sample_tasks.py                  # Sample N tasks with seed + timeout filter
│   └── copy_task_trajs.py               # Copy trajectory dirs by task list JSON
└── runs/
    └── <run_tag>/
        ├── R0/  R1/  …  Rn/             # Promoted config.yaml per round
        ├── learnings.md                 # Meta-agent cross-round learning journal
        └── _meta_v2/
            ├── R1/  …  Rn/
            │   ├── config.yaml
            │   ├── templates/tb2_system.j2
            │   └── processors/*.py
            └── _meta_scratch/
                └── harness_evolve_state.json
```

---

## Results

### Table 1 — Evolution Effectiveness (full 89-task set)

> Compares the Qwen3.5-27B self-reported score, our HarnessX baseline, and the best evolved config.
> The baseline and evolved config use Qwen3.5-27B as the task-agent.

| Config | Score | Tasks passed |
|--------|:-----:|:------------:|
| [Qwen3.5-27B](https://modelscope.cn/models/Qwen/Qwen3.5-27B) | 41.6% | — |
| HarnessX default config (our baseline) | 38.2% | 34 / 89 |
| **Evolved R4 config** | **47.2%** | **42 / 89** |

The HarnessX default config scores slightly below the Qwen3.5-27B self-reported number, which is expected
given differences in infrastructure and prompt format. The evolved R4 config surpasses
both by +9 pp over our baseline and +5.6 pp over the self-reported score.

> **Note**: Both the HarnessX baseline and Evolved R4 scores are based on a single evaluation
> run (k=1) over the full 89-task set. Single-run variance on 89 tasks is approximately ±4–5 tasks
> (1 σ), so the reported scores are statistically comparable in direction, but exact numbers may not
> fully reproduce due to this inherent variance.

### Table 2 — Generalization: Training Set vs Unseen Tasks

> Splits the 89-task full eval into the 15 tasks the meta-agent trained on
> and the remaining 74 tasks it never saw during evolution.

| Split | Baseline (R0) | Evolved R4 | Gain |
|-------|:-------------:|:----------:|:----:|
| **Training set** (15 tasks the meta-agent saw) | 8 / 15 = 53.3% | 10 / 15 = 66.7% | **+13.3 pp** |
| **Unseen tasks** (74 tasks never seen during evolution) | 26 / 74 = 35.1% | 32 / 74 = 43.2% | **+8.1 pp** |
| **Total** | 34 / 89 = 38.2% | 42 / 89 = 47.2% | **+9.0 pp** |

Both splits improve, confirming that the evolution produced general-purpose improvements
rather than task-specific memorization. The training-set gain (13.3 pp) is 1.6× the
unseen-task gain (8.1 pp), consistent with limited but real transfer.

**Caveat**: results are from a single run per config. Round-to-round variance on 89 tasks
is ≈ ±4–5 tasks (1 σ), so the +8-task net gain is ~1.7 σ (p ≈ 0.09) — suggestive but not
conclusive without repeated eval runs.

---

## Evolution Experiment

**Training set**: 15 tasks sampled with seed=42, agent_timeout ≤ 2000 s (`tasks_sample16_seed42_act15.json`)
**R0 trajectories**: extracted from the full 89-task baseline run (`run_full_eval.sh`) by selecting the 15 training-set tasks — used as the meta-agent's warm-start input for R1
**Meta-agent**: Claude Opus 4.6
**Task-agent**: Qwen3.5-27B via SGLang (local OpenAI-compatible endpoint)
**Rounds**: 5 (R0–R5 covered below)

### Score Progression (first 5 rounds)

| Round | Training set (15) | Changes |
|-------|:-----------------:|---------|
| R0 (baseline) | 8/15 = 53% | Default config |
| R1 | 8/15 = 53% | Jinja2 template; timeout / cleanup / background-service guidance |
| R2 | 7/15 = 47% | Source-file read-only constraint — overly broad, caused regression |
| R3 | **10/15 = 67%** | Fixed R2 regression; refined code-modification semantics; archive extraction rules |
| R4 | **10/15 = 67%** | `send()` vs `sendline()` guidance; GPU checks; `SqlReadOnlyGuard` + `PackageEnvGuide` processors |
| R5 | 9/15 = 60% | Removed phantom `/tests/` path references that were wasting agent steps |

Peak at R3/R4 (67%). R5 slight regression when phantom-path removal inadvertently changed agent behaviour on some tasks.

### What Each Round Changed

**R1** — Converted the static system prompt to a Jinja2 template. Added: (1) use `timeout` parameter for long commands, (2) run background services with `nohup` and verify reachability, (3) always confirm output files exist with `ls` before declaring done.

**R2** — Added a "source files are read-only" constraint and warnings about verifier conflicts. The read-only rule was too broad and blocked tasks that legitimately need to modify source files, causing a 1-task regression.

**R3** — Refined the R2 rules: read-only applies to *input data*, not to source code the task explicitly asks to fix. Added archive extraction semantics (preserve top-level directory, do not flatten). Recovered +3 tasks vs R2.

**R4** — Added terminal/ML-specific sections: `send()` vs `sendline()` distinction for pexpect (sendline appends `\n`, causing unintended command execution), GPU availability check before CUDA inference. Introduced two control-layer processors:
- **`SqlReadOnlyGuard`**: blocks `CREATE INDEX`, `ALTER TABLE`, `INSERT` on task databases
- **`PackageEnvGuide`**: when pip fails with "externally-managed-environment", injects pre-installed package paths (up to 3 times per run)

**R5** — Removed references to `/tests/*.py` files that do not exist in most task containers. These were causing agents to waste steps searching for a non-existent test suite.

### Template Size

| Round | Lines | Note |
|-------|------:|------|
| R1 | 60 | Baseline template |
| R4 | 89 | Peak quality, mostly general guidance |

---

## Conclusions

### Evolution is effective, but the signal is noisy

1. **Net +8 tasks (38.2% → 47.2%)** over the baseline, with gains on both the training set and unseen tasks — not pure memorization.

2. **~3 gains are mechanistically supported** (mcmc-sampling-stan, regex-log, custom-memory-heap-crash): 2–6× token reductions alongside FAIL → PASS indicate systematic behaviour change, not random variance.

3. **R4 is the peak**: R5 and later rounds introduced processor side effects and template bloat that eroded generalization.

### Key Lessons

| Lesson | Detail |
|--------|--------|
| **Control layer > instruction layer** | `PackageEnvGuide` and `SqlReadOnlyGuard` produce verifiable, mechanism-backed effects that prompt text alone cannot guarantee |
| **Constraints must be narrow** | R2's broad read-only rule caused a regression; every constraint needs an explicit trigger condition to avoid breaking unrelated tasks |
| **Regex processors are fragile** | `SolutionCleanupGuard` took two rounds to fix because heredoc content matched the block patterns; use structured parsing instead |
| **Training set of 15 is too small** | σ ≈ ±1.9 tasks per round; single-round scores are unreliable — target ≥ 30 tasks or run each round twice |
| **Single-run evals mislead the meta-agent** | ±4–5 task variance on 89 tasks means a one-point drop may trigger unnecessary recovery rounds |

### Recommended Next Steps

1. **Start the next evolution from R4** — best training/generalization balance, compact 89-line template
2. **Fix the compile-compcert regression** — diagnose which `SqlReadOnlyGuard` pattern fires and add a build-tool whitelist
3. **Expand the training set to 30–50 tasks** to reduce per-round variance
4. **Run each round's eval twice and average** before treating the result as signal
5. **Exclude stuck tasks** (`gpt2-codegolf`, `headless-terminal`) from the training set to remove uninformative noise

---

## Limitations

This evolution experiment is preliminary. All evolution rounds and full-set evaluations use
**k=1** (a single run per config), which carries substantial variance (σ ≈ ±4–5 tasks on 89
tasks, σ ≈ ±1.9 tasks on the 15-task training set). As a result:

- Reported scores should be treated as indicative rather than definitive.
- Reproducing the exact task-pass counts is unlikely; expect ±3–5 tasks of fluctuation.
- Some round-to-round score changes during evolution may reflect noise rather than genuine signal.

Future work will increase the number of evaluation samples per round (k ≥ 3) and expand the
training set to ≥ 30 tasks to reduce variance and make the evolution signal more reliable.

---

## Experiment Setup & Reproduction

### Overview

```
Step 1  Install dependencies
Step 2  Configure .env
Step 3  Run full-set baseline (89 tasks, default config)   →  establish ground truth
Step 4  Sample a training subset                           →  tasks to evolve on
Step 5  Extract training-set trajectories from baseline    →  R0 warm-start for meta-agent
Step 6  Run evolution on the training subset               →  iterate R0 → Rn
Step 7  Evaluate the best evolved config on all 89 tasks   →  measure generalization
```

---

### Step 1 — Install

```bash
# Install uv (if not already available)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or ~/.zshrc

# Clone the repo and create a Python 3.12 virtualenv
git clone <repo-url>
cd HarnessX

uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install harnessx (editable) + benchmark dependencies
uv pip install -e .
uv pip install harbor==0.2.0

# Verify Docker Compose v2 is available (required for local TB2 eval)
docker compose version   # must be v2.x
```

### Step 2 — Configure Environment

```bash
cp recipe/tb2_evolver/.env.example recipe/tb2_evolver/.env
# Edit .env and fill in:
#   ANTHROPIC_API_BASE / ANTHROPIC_API_KEY  — meta-agent endpoint (Claude Opus 4.6)
#   TB2_API_BASE                            — SGLang endpoint serving Qwen3.5-27B
#   TB2_MODEL                               — model ID exposed by your SGLang instance
```

Key variables (see `.env.example` for full reference):

```dotenv
# Meta-agent (Claude Opus 4.6 via Anthropic-compatible API)
ANTHROPIC_API_BASE=https://...
ANTHROPIC_API_KEY=sk-...
META_MODEL=claude-opus-4-6

# Task-agent (Qwen3.5-27B served by SGLang)
TB2_MODEL=qwen3.5-27b      # model ID your SGLang instance exposes
TB2_API_BASE=http://<sglang-host>:<port>/v1
TB2_API_KEY=none

CONCURRENT=8               # parallel tasks per eval round
```

### Step 3 — Run Full-Set Baseline (89 tasks)

Evaluate the **default** HarnessX config (no custom prompt, no processors) on all tasks.
This is the performance floor against which evolved configs are measured.

```bash
cd HarnessX
set -a && source recipe/tb2_evolver/.env && set +a

bash benchmarks/terminal_bench_2/scripts/eval_local_docker.sh \
  --tasks recipe/tb2_evolver/tasks_all_tb2.json \
  --job-name tb2-baseline \
  -n "${CONCURRENT:-8}"
```

Results are written to `.benchmarks/tb2/tb2-baseline/<task>__<hash>/result.json`.

### Step 4 — Sample a Training Subset

Select N tasks, filtering out tasks whose agent timeout exceeds the threshold
(long-running tasks add cost noise without additional evolution signal).

```bash
python recipe/tb2_evolver/scripts/sample_tasks.py \
  --seed 42 \
  --n 16 \
  --max-agent-timeout 2000 \
  --input recipe/tb2_evolver/tasks_all_tb2.json
# → recipe/tb2_evolver/tasks_sample16_seed42_act15.json  (15 tasks after timeout filter)
```

The sampled file becomes the **evolution training set** — all subsequent rounds run and
score only these tasks.

### Step 5 — Extract Training-Set Trajectories from Baseline (R0 warm-start)

The meta-agent needs R0 trajectories before it can produce R1. Rather than re-running
the baseline, copy the relevant directories out of Step 3:

```bash
python recipe/tb2_evolver/scripts/copy_task_trajs.py \
  --tasks recipe/tb2_evolver/tasks_sample16_seed42_act15.json \
  --src  .benchmarks/tb2/tb2-baseline \
  --dst  .benchmarks/tb2-baseline-results/r0-baseline-16
```

`start_evolve_r0baseline.sh` passes this directory to the meta-agent as `--r0-dir`.

### Step 6 — Run Evolution

```bash
# Start a new run (R0 → R{NUM_ROUNDS}, default 6 rounds)
bash recipe/tb2_evolver/scripts/start_evolve_r0baseline.sh

# Override run tag and round count
RUN_TAG=my-exp-01 NUM_ROUNDS=5 \
  bash recipe/tb2_evolver/scripts/start_evolve_r0baseline.sh

# Resume an interrupted run
RESUME=1 RUN_TAG=my-exp-01 \
  bash recipe/tb2_evolver/scripts/start_evolve_r0baseline.sh
```

Each round: (1) meta-agent session ~10–30 min → evolved `config.yaml`; (2) TB2 eval on
15 training tasks; (3) score logged, entry appended to `runs/<run_tag>/learnings.md`.

Promoted configs: `runs/<run_tag>/R{n}/config.yaml`.
Intermediate artifacts: `runs/<run_tag>/_meta_v2/R{n}/` (templates, processors, scratch notes).

### Step 7 — Evaluate the Best Config on All 89 Tasks

Identify the best round from `learnings.md` or `_meta_v2/_meta_scratch/harness_evolve_state.json`,
then run the full evaluation:

```bash
bash recipe/tb2_evolver/scripts/run_full_eval.sh \
  --config recipe/tb2_evolver/runs/my-exp-01/R4/config.yaml \
  --job-name my-exp-01-r4-full

# Resume if interrupted
bash recipe/tb2_evolver/scripts/run_full_eval.sh \
  --config recipe/tb2_evolver/runs/my-exp-01/R4/config.yaml \
  --job-name my-exp-01-r4-full \
  --resume
```

Results land in `.benchmarks/tb2/my-exp-01-r4-full/`.
Compare against the baseline from Step 3 to measure the net gain.
