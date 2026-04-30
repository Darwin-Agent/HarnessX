# Benchmarks

HarnessX benchmark adapters — each sub-directory is a self-contained adapter for one evaluation suite.

| Directory | Benchmark | Status |
|-----------|-----------|--------|
| `terminal_bench_2/` | [Terminal Bench 2.0](https://github.com/harbor-framework/terminal-bench-2) | [Available](terminal_bench_2/README.md) |
| `gaia/` | [GAIA](https://huggingface.co/gaia-benchmark) | [Available](gaia/README.md) |
| `swebench/` | [SWE-bench](https://github.com/SWE-bench/SWE-bench) | [Available](swebench/README.md) |
| `tau2/` | [TAU2-Bench](https://github.com/sierra-research/tau2-bench) | [Available](tau2/README.md) |
| `evoclaw/` | [EvoClAW](https://github.com/EvoClaw-Bench/EvoClaw) | Ongoing |
| `locomo/` | [LoCoMo](https://github.com/snap-research/locomo) | Ongoing |
| `osworld/` | [OSWorld](https://github.com/xlang-ai/OSWorld) | Ongoing |

---

## Benchmark descriptions

### [Terminal Bench 2.0](terminal_bench_2/README.md)

[Terminal Bench 2.0](https://github.com/harbor-framework/terminal-bench-2) tests agents on 89 bash/file-system tasks executed inside Harbor-managed containers. Tasks cover compilation, debugging, system administration, cryptography, and ML workloads. Use it to evaluate low-level terminal proficiency and tool-use reliability.

### [GAIA](gaia/README.md)

[GAIA](https://huggingface.co/gaia-benchmark) tests general agent capability across reasoning, tool use, and web information retrieval. Tasks range from simple factual lookups to multi-step questions requiring file parsing, web search, and cross-source synthesis. Use it to evaluate how well a harness handles open-ended, real-world assistant tasks.

### [SWE-bench](swebench/README.md)

[SWE-bench](https://www.swebench.com) tests software engineering capability using real GitHub issues from popular Python repositories. Each task asks the agent to produce a patch that resolves the issue and passes the associated test suite. Use it to evaluate coding and debugging ability in a realistic, large-codebase setting.

### [TAU2-Bench](tau2/README.md)

[TAU2-Bench](https://github.com/sierra-research/tau2-bench) tests tool-augmented agents in user-simulation scenarios. Tasks involve multi-turn interactions with simulated users and structured tool APIs (retail, airline, etc.), evaluating whether the agent can complete user requests correctly while following domain-specific constraints.

---

## Results

### Terminal Bench 2.0

89 terminal/coding tasks inside Harbor-managed containers.

| Agent | Model | Tasks | Pass | Score | Runs per task | Notes |
|-------|-------|-------|------|-------|--------------|-------|
| **HarnessX** | claude-opus-4-6 | 89 | 56 | **63.0%** | 1 (k=1) | OpenSandbox |
| Claude Code | claude-opus-4-6 | 89 | — | **58.0% ± 2.9** | 5 (k=5) | [tbench.ai](https://www.tbench.ai/leaderboard/terminal-bench/2.0) |
| **HarnessX** | claude-haiku-4-5 | 89 | 28 | **31.5%** | 1 (k=1) | OpenSandbox |
| Claude Code 2.0.31 | claude-haiku-4-5 | 89 | — | **27.5% ± 2.8** | 5 (k=5) | [tbench.ai](https://www.tbench.ai/leaderboard/terminal-bench/2.0/claude-code/2.0.31/claude-haiku-4-5-20251001%40anthropic) |

> HarnessX was run once per task (k=1); the official leaderboard uses k=5 and reports mean ± confidence interval. Direct comparison is not strictly valid, but the scores are closely aligned.

### TAU2-Bench

Multi-turn customer-service simulation across three domains. Agent: Sonnet 4.5 + Extended Thinking (budget: 62976 tokens). User simulator: GPT-5.2. The *w/o HarnessX* column is the published [tau2 leaderboard](https://taubench.com) score for the same model.

| Domain | Tasks | Pass | Pass rate | w/o HarnessX | Notes |
|--------|-------|------|-----------|--------------|-------|
| **telecom** | 114 | 98 | **86.0%** | 84.9% | base split |
| **airline** | 50 | 37 | **74.0%** | 72.0% | |
| **retail** | 114 | 81 | **71.1%** | 72.4% | StopGuard enabled |

> StopGuard strips premature `###STOP###` tokens from GPT user-simulator confirmation messages, recovering tasks that would otherwise fail before the agent executes the write tool (~8 pp on retail without it).
