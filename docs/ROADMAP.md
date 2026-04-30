# Roadmap

This document describes the ongoing work and planned directions for HarnessX.
Items marked **Ongoing** are actively being developed; items marked **Planned** are next in line.

---

## 1. Memory Extensions

HarnessX treats memory as a first-class behavior dimension: the `MemoryBundle` and `context`-processor family give agents pluggable, composable recall without touching the model layer.

Our immediate focus is deepening this foundation.

### Ongoing

- **LoCOMo benchmark** — long-context memory evaluation directly inside HarnessX.
  Exercises session-level recall, cross-turn consistency, and compaction fidelity.

### Planned

- **Third-party memory adapters** (in `recipe/` as opt-in integrations):
  - [MemPalace](https://github.com/mem-palace/mem-palace) — structured episodic memory store
  - [SuperMemory](https://supermemory.ai) — cloud-backed semantic memory
  - [OpenVKing](https://github.com/openvking) — vector-knowledge-graph memory
- **Memory quality metrics** — retrieval precision / recall reporting surfaced through HarnessJournal so runs can be compared head-to-head on memory effectiveness.

---

## 2. RL Training & Data Flywheel

HarnessX trajectories are designed to be training-ready:
`trajectory.to_training_records()` for SFT and `trajectory.to_rl_records()` for GRPO-style pipelines already ship in the core.

### Ongoing

- **VERL integration** — connect HarnessX rollouts to [VERL](https://github.com/volcengine/verl) (Volcano Engine RL) so trajectories flow directly into distributed PPO / GRPO training loops.

### Planned

- **Genetic / evolutionary RL** — population-level harness mutation: spawn variant harnesses, evaluate them in parallel, select survivors, repeat. Enables behavior-level evolution without model fine-tuning.
- **Data synthesis pipeline** — use HarnessX as a controlled data generator: configure task distributions, diversity constraints, and quality filters, then export structured SFT or preference datasets automatically.
- **Data flywheel tooling** — close the loop between deployment and training: production trajectories flow back into offline RL pipelines with minimal human labeling.

---

## 3. Agentic Benchmark Integration

HarnessX ships a benchmark adapter layer (`benchmarks/`) that standardizes task loading, environment setup, and result scoring across multiple evaluation suites.

### Ongoing

| Benchmark | Focus area |
|-----------|------------|
| [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) | General assistant, tool use, web reasoning |
| [SWE-bench](https://www.swebench.com) | Real-world GitHub issue resolution |
| [TAU2-Bench](https://github.com/sierra-research/tau2-bench) | Tool-augmented user simulation |
| [EvoClAW](https://github.com/evo-claw) | Evolutionary coding and algorithmic tasks |
| [OSWorld](https://os-world.github.io) | Desktop GUI and OS-level task completion |
| [TerminalBench 2.0](https://github.com/laude-institute/terminal-bench) | Terminal / CLI proficiency |

### Planned

- Unified leaderboard view in the Lab UI: run multiple benchmarks from a single `hx bench` command, compare scores across model configurations side-by-side.
- Scoring registry: each benchmark adapter exports a `score()` function so third-party benchmarks can be plugged in without modifying HarnessX core.

---

## 4. Meta-Harness (Agent Self-Evolution)

**Goal:** let the model observe its own behavior trajectories and use that signal to modify or create new harness configurations — effectively implementing agent self-improvement at the harness level without requiring model fine-tuning.

### Design sketch

1. An **observer harness** runs alongside the primary agent and monitors processor trigger events and outcome scores.
2. A **meta-agent** receives a summary of underperforming patterns (e.g., repeated context truncations, low tool-call precision) and proposes changes to the active `HarnessConfig` (swap a processor, adjust a bundle parameter, add a new context strategy).
3. The proposed config is validated, optionally sandboxed for one evaluation turn, and promoted if it improves on a held-out metric.

This closes the loop: task → trajectory → meta-analysis → harness update → better task performance.

---

## 5. HarnessHUB

A community sharing platform for harnesses: publish a named, versioned `HarnessConfig` bundle (tools + processors + memory + trace settings) and let others pull and run it in one command.

```bash
hx pull coding-agent@v1.2
hx pull deep-research@latest
```

### Planned features

- **One-line pull & run** — `hx pull <name>` fetches the config and its processor dependencies.
- **Version pinning** — harnesses are immutable once published; `@tag` references are stable.
- **Browse in Lab UI** — the Lab workspace ships a HarnessHUB panel for searching and one-click import.
- **Private registries** — teams can host internal hubs for proprietary harness configurations.

---

## Contributing

We welcome benchmarks, memory adapters, and harness recipes. See `docs/guide/composable_harness.md` for the composition model and `benchmarks/terminal_bench_2` as a reference adapter implementation.

Issues and discussion: open a GitLab issue or start a merge request.
