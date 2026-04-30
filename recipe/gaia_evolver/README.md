# GAIA Evolver

A self-improving benchmark loop for HarnessX: run a target agent on GAIA tasks,
let a meta-agent diagnose the trajectories, and produce a new `HarnessConfig`
for the next round.

This recipe is the primary consumer of `harnessx.meta_harness.MetaAgent`. See
[`harnessx/meta_harness/workspace/SOUL.md`](../../harnessx/meta_harness/workspace/SOUL.md)
for the meta-agent's operating contract and
[`docs/agents.md`](../../docs/agents.md) for the underlying harness API.

---

## Loop shape

```
R0 (baseline)                 R1                            R2 …
┌───────────────┐  evolve()  ┌───────────────┐  evolve()   ┌───────────────┐
│ run tasks     │ ─────────▶ │ run tasks     │ ──────────▶ │ run tasks     │
│ LLMJudge per  │            │ LLMJudge per  │             │ LLMJudge per  │
│ task → .md    │            │ task → .md    │             │ task → .md    │
└──────┬────────┘            └──────┬────────┘             └──────┬────────┘
       │                            │                             │
       ▼ evolve()                   ▼ evolve()                    ▼ gate
  new config.yaml             new config.yaml              final summary
       │                            │
       ▼ gate (best-so-far)         ▼ gate
  kept / reverted             kept / reverted
```

Each round:

1. Every task runs under the current `HarnessConfig`. Per-task `.md`
   trajectories (with judge frontmatter) and a `sessions/` JSONL land under
   `runs/{tag}/R{N}/trajectories/`. Per-task behavioral assessment comes from
   `harnessx.processors.evaluation.LLMJudgeProcessor`, injected by default
   (disable with `--no-judge`). Each trajectory carries a
   `judge_verdict` / `cause` / `missing` / `lesson` frontmatter block that
   the meta-agent uses as grep entry points when diagnosing failure patterns.
2. `MetaAgent.evolve()` reads the trajectories and writes a new
   `config.yaml` under `runs/{tag}/R{N+1}/evolve/`. A byte-identical copy is a
   valid "no-op" outcome.
3. The outer loop gates the round against the **best-so-far** round's score
   (`pass_rate - cost_weight * max(cost_delta, 0)`). Regressions past
   `--regression-tolerance` revert `current_config`; the memo records the
   decision. A `--pass-count-noise-threshold` guard prevents spurious
   rollbacks on small task sets where 1-2 pass flips are stochastic noise.

The last round skips step 2 — no one would consume the extra `evolve()`
output.

---

## Two entry points

| Script | Scope | Use case |
|--------|-------|----------|
| `run.py` | Flat — all tasks in one pool | Quick experiments, single-model runs |
| `run_meta.py` | Per-domain — tasks split by category | Domain-specialized evolution, multi-model (agent + meta-agent) |

---

## CLI — `run.py` (flat evolution)

```bash
python -m recipe.gaia_evolver.run \
    --max-tasks 5 \
    --num-rounds 3 \
    --run-tag my-experiment
```

Common flags (see `--help` for the full set):

| Flag | Meaning | Default |
|---|---|---|
| `--max-tasks N` | How many GAIA tasks per round | 6 |
| `--num-rounds N` | Rounds including R0 baseline | 3 |
| `--model` | Model for the inner (task-doing) agent | `anthropic/YOUR_PROVIDER/claude-sonnet-4-6` |
| `--meta-model` | Model for the meta-agent and judge (stronger tier) | `anthropic/YOUR_PROVIDER/claude-opus-4-6` |
| `--provider-id` | Provider header for routing | `YOUR_PROVIDER_ID` |
| `--api-base` / `--api-key` | Route inner agent to a custom OpenAI-compatible endpoint | unset |
| `--max-cost` | Per-task cost cap (USD) | 5.0 |
| `--max-steps` | Per-task step cap | 20 |
| `--level` | GAIA difficulty level (1/2/3, or 0 = all) | 0 |
| `--concurrency` | Max concurrent trajectories per round | 4 |
| `--data-path` | Path to local GAIA JSON (webthinker schema); `''` falls back to HuggingFace | `data/webthinker_gaia_dev.json` |
| `--attachments-dir` | Dir with per-task attachment files `<task_id>.<ext>` | unset |
| `--no-judge` | Disable LLMJudgeProcessor (verdict fields omitted from frontmatter) | judge enabled |
| `--evolve-cost` / `--evolve-steps` / `--evolve-wall-clock` | Meta-agent budgets per round | $50 / 200 steps / 10000s |
| `--regression-tolerance` | Score drop allowed before revert (0.0 = any regression reverts) | 0.03 |
| `--cost-weight` | Multiplier on cost-regression penalty (0.0 = pass_rate only) | 0.0 |
| `--pass-count-noise-threshold` | Absolute pass-count delta below which regression is treated as noise | 3 |
| `--run-tag` | Output subdir under `runs/`. Defaults to `run_<timestamp>` | auto |
| `--clean` | Wipe `runs/<tag>/` before starting | off |

Auth: put `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_API_BASE`) in a
`.env` at the project root. The runner auto-loads it.

---

## CLI — `run_meta.py` (per-domain evolution)

Splits GAIA tasks by domain category (e.g. Multi-hop, Database, Pure Logic)
and evolves each domain independently. Designed for a two-model setup: a
target agent (e.g. GPT-5) runs the tasks, and a meta-agent (e.g. Claude
Opus 4.6 with extended thinking) evolves the config.

```bash
# List available domains
python -m recipe.gaia_evolver.run_meta --list-domains

# Evolve a single domain
python -m recipe.gaia_evolver.run_meta --domain Multi-hop --num-rounds 4

# Evolve all domains sequentially (default: 8 rounds each)
python -m recipe.gaia_evolver.run_meta

# Smoke test
python -m recipe.gaia_evolver.run_meta \
    --domain Single-source_Fact_Lookup \
    --num-rounds 2 --max-tasks 3 --concurrency 2
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--domain` | `None` (all) | Single domain short name, or omit for all domains |
| `--num-rounds` | `8` | Rounds including R0 baseline |
| `--model` | `gpt-5` | Target agent model |
| `--meta-model` | `anthropic/YOUR_PROVIDER/claude-opus-4-6` | Meta-agent model (prefix `anthropic/` → AnthropicProvider) |
| `--api-base` | `$OPENAI_API_BASE` | Target agent API endpoint |
| `--provider-id` | `azure_openai` | `X-Model-Provider-Id` header for the target agent |
| `--max-steps` | `40` | Max agent steps per task |
| `--max-cost` | `15.0` | Per-task USD cost cap |
| `--concurrency` | `6` | Parallel tasks per round |
| `--evolve-cost` | `50.0` | Meta-agent USD budget per evolve call |
| `--evolve-steps` | `200` | Meta-agent max steps per evolve |
| `--evolve-wall-clock` | `10000` | Meta-agent wall-clock cap (seconds) |
| `--data-path` | `data/webthinker_gaia_dev_classified.json` | Classified task JSON |
| `--max-tasks` | `0` (all) | Limit tasks per domain |
| `--run-tag` | `domain_evolve_v1` | Output subdir under `runs/` |
| `--clean` | off | Wipe run directory before starting |
| `--list-domains` | — | Print domains with task counts and exit |

### Per-domain loop shape

```
Domain: Multi-hop (17 tasks)
  R0 ── run all 17 tasks ── gate (baseline)
  R1 ── evolve() ── run all 17 tasks ── gate (keep/revert)
  R2 ── evolve() ── run all 17 tasks ── gate (keep/revert)
  ...
  Early stop: 2 consecutive noop configs → domain finished
```

Each domain evolves independently with its own `learnings.md` and config
lineage. This lets the meta-agent specialize: database tasks might need
different processors than pure-logic puzzles.

### Gating

After each evolved round, the gate compares against the **best-so-far**
round's pass rate. On regression, the config reverts and the meta-agent
sees the revert in `learnings.md` for the next round.

### Early stop

If the meta-agent produces a byte-identical config twice in a row (2
consecutive "noop" rounds), the domain stops early — no further improvement
expected.

### Output layout

```
runs/{tag}/domain/
├── Multi-hop/
│   ├── R0/
│   │   ├── config.yaml
│   │   ├── trajectories/{task_id}.md
│   │   └── sessions/
│   ├── R1/
│   │   ├── config.yaml
│   │   ├── trajectories/
│   │   ├── sessions/
│   │   └── evolve/          # meta-agent workspace
│   │       ├── config.yaml
│   │       ├── _meta_scratch/
│   │       └── meta_workspace/sessions/
│   ├── configs/              # R0_config.yaml, R1_config.yaml, ...
│   ├── curves.json           # per-round pass rate / cost / tokens / status
│   └── learnings.md          # cross-round memo
├── Database/
│   └── ...
├── Pure_Logic/
│   └── ...
└── summary.json              # aggregate results across all domains
```

### `curves.json` format

Each round appends one entry:

```json
{
  "round": 0,
  "config_hash": "abc123...",
  "evolve_status": "baseline",
  "total_tasks": 17,
  "passed": 12,
  "pass_rate": 0.7059,
  "pass_pct": "70.6%",
  "cost_usd": 83.49,
  "total_tokens": 26164258,
  "level_stats": {"1": [7, 5], "2": [8, 6], "3": [2, 1]}
}
```

### Reference experiment

GPT-5 (agent) + Claude Opus 4.6 (meta-agent), 103 tasks, 9 domains, 8 rounds:

| Metric | Value |
|--------|-------|
| R0 Baseline | 64/103 (62.1%) |
| Best evolved (per-domain) | 86/103 (**83.5%**) |
| Delta | +22 tasks, +21.4pp |
| Total cost | $1,519 |

Domains reaching 100%: Pure Logic (12/12), Multi-constraint (5/5),
Web Archive (4/4), Single-source (22/23 = 95.7%).
Largest absolute gain: Database 0% → 53.8% (+7 tasks).

---

## Output layout

```
recipe/gaia_evolver/runs/{tag}/
├── R0/
│   ├── config.yaml              # the HarnessConfig that ran R0
│   ├── trajectories/
│   │   └── {task_id}.md         # YAML frontmatter (with judge_verdict) + step log
│   └── sessions/                # per-task HarnessJournal JSONL
├── R1/
│   ├── config.yaml              # config produced by R0→R1 evolve()
│   ├── trajectories/ …
│   ├── sessions/ …
│   └── evolve/                  # meta-agent scratch for this transition
│       ├── config.yaml          # ← candidate produced by the meta-agent
│       ├── tools/*.py           # optional new @tool modules
│       ├── processors/*.py      # optional new MultiHookProcessor classes
│       ├── _meta_scratch/       # meta-agent notes, TASK.md brief
│       └── meta_workspace/      # meta-agent HarnessJournal
├── R2/ …
├── learnings.md                 # cross-round memo the meta-agent reads + appends
└── comparison.json              # all rounds + per-round summaries
```

`comparison.json`'s `round_summaries[i]` carries `evolve_status`:

| Status | Meaning |
|---|---|
| `baseline` | R0; no evolve produced this config |
| `ok` | Prior round's evolve produced a new (byte-different) config |
| `noop` | Prior round's evolve produced a byte-identical copy (explicit no-op) |
| `crashed` | Prior round's evolve raised; current config was reused |

---

## What the meta-agent actually does

The meta-agent is an ordinary HarnessX agent whose identity, boundaries,
and skills live under
[`harnessx/meta_harness/workspace/`](../../harnessx/meta_harness/workspace/).

A `MetaAgent` instance is created once per run with persistent config
(model, memo path, budgets, extra skills). Each non-final round calls
`await meta_agent.evolve(...)` with per-round args pointing at:

- `current_config` (the YAML the agent is evolving),
- `trajectories_dir` (R{N}'s outputs),
- `output_dir` (R{N+1}/evolve/),
- `replay_model` / `replay_max_cost_usd` (for synthetic-task replay gate).

The agent follows the loop in `SOUL.md`, consulting:

- `skills/analyze` for the lens x lever x intent framework and retroactive checks,
- `skills/gaia-playbook` (mounted from `recipe/gaia_evolver/skills/`) for
  benchmark-specific guidance,
- `skills/reference` when authoring a new processor / tool / template or
  editing config.yaml,
- `skills/validate` for self-validation CLIs,
- `skills/journal` before it stops to append its round entry.

Its tools are the standard file + shell + web set plus one synchronous
leaf worker (`spawn_reflect_worker`) for digesting large trajectory sets.

The outer loop never inspects what the agent did — it only loads
`output_dir/config.yaml` via `HarnessConfig.from_yaml_file(...).canonicalize()`
and, if that succeeds, adopts it for the next round (subject to the
best-so-far gate).

---

## Relation to the rest of HarnessX

- `harnessx/meta_harness/` — module implementing `MetaAgent` and the
  meta-agent's persona. Benchmark-agnostic; this recipe is one consumer.
- `benchmarks/gaia/` — GAIA dataset loader + evaluator. Pure benchmark
  glue; it does not know about meta-harness.
- `recipe/gaia_evolver/` — orchestration that wires the two together.
