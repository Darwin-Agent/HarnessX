# Architecture

<p align="center">
  <img src="assets/harnessx_architecture.png" alt="HarnessX Architecture" width="800"/>
</p>

HarnessX organizes agent behavior into **9 orthogonal behavioral dimensions**, mapped to the three pillars:

> 🧩 **Compose** = Model · Context · Memory · Tools · Sandbox
> ⚙️ **Adapt** = Evaluate · Control
> 🚀 **Evolve** = Observe · Train

| Slot | Dimension | What It Controls |
|:----:|-----------|------------------|
| ![1](https://img.shields.io/badge/1-MODEL-00d4ff?style=flat-square) | **Model Selection** | Multi-provider routing + role assignment (main / judge / evaluator) — `ModelConfig`, separate from `HarnessConfig` |
| ![2](https://img.shields.io/badge/2-CONTEXT-3b82f6?style=flat-square) | **Context Assembly** | System prompt strategy + history truncation + user message wrapping — `processors/context/` |
| ![3](https://img.shields.io/badge/3-MEMORY-a855f7?style=flat-square) | **Memory Management** | Extract → Store → Retrieve with 5 pluggable strategies incl. Light-Memory — `processors/memory/` |
| ![4](https://img.shields.io/badge/4-TOOLS-22c55e?style=flat-square) | **Tool Ecosystem** | Built-in tools + MCP protocol + Skills + filtering — `processors/tools/` + `tools/` |
| ![5](https://img.shields.io/badge/5-SANDBOX-f59e0b?style=flat-square) | **Execution Environment** | Sandbox isolation: Local / Docker / E2B cloud — `sandbox/` |
| ![6](https://img.shields.io/badge/6-EVALUATE-ef4444?style=flat-square) | **Evaluation & Reward** | LLM Judge / SelfVerify / PRM / Benchmark evaluators — `processors/evaluation/` |
| ![7](https://img.shields.io/badge/7-CONTROL-06b6d4?style=flat-square) | **Control & Safety** | 13 processors: loop detection, cost guard, compaction, sycophancy check — `processors/control/` |
| ![8](https://img.shields.io/badge/8-OBSERVE-8b5cf6?style=flat-square) | **Observability** | HarnessJournal (JSONL) + OpenTelemetry + checkpoints + session recovery — `processors/observability/` |
| ![9](https://img.shields.io/badge/9-TRAIN-ec4899?style=flat-square) | **Training Bridge** | Trajectory → SFT / RL records with token-level annotations — `rl/` |

## Event-Driven Processor Pipeline

All behavior is implemented as **Processors** at 8 hook points:

<p align="center">
  <img src="assets/pipeline_hooks.png" alt="Event-Driven Processor Pipeline" width="800"/>
</p>

Compose with `|` and exhaustive conflict detection:

```python
from harnessx.bundles.coding import make_coding
from harnessx.bundles.reliability import make_reliability

config = (make_coding(working_dir=".") | make_reliability()).build()
# HarnessConflictError if any singleton_group collides — no silent overwrites
```
