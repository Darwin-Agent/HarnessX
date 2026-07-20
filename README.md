<table align="center" border="0" cellspacing="0" cellpadding="0"><tr><td align="center" valign="middle">
<img src="docs/assets/harnessx_logo.png" alt="HarnessX Logo" width="72"/>
</td><td align="left" valign="middle">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/harnessx_wordmark_dark.png"/>
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/harnessx_wordmark.png"/>
  <img alt="HarnessX" src="docs/assets/harnessx_wordmark.png" height="48"/>
</picture>
<br/>
<b>Compose. &nbsp; Adapt. &nbsp; Evolve.</b>
</td></tr></table>

<p align="center">
  <strong>Compose the Harness, define the Agent.<br/>
  From zero-code to full customization — one core, X entry points.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-22c55e?style=flat"/></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3b82f6?style=flat&logo=python&logoColor=white"/>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-a855f7?style=flat"/>
  <img alt="Status" src="https://img.shields.io/badge/Status-Beta-f59e0b?style=flat"/>
  <a href="https://darwin-agent.github.io/HarnessX/"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-HarnessX-ed722e?style=flat"/></a>
</p>

<p align="center">
  <a href="https://darwin-agent.github.io/HarnessX/">Homepage</a> •
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="README_zh.md">中文文档</a>
</p>

---

<a id="overview"></a>
## 🔭 Overview


> **The harness — not just the model — determines agent performance.** The same base model produces dramatically different results depending on how context is managed, how tools are orchestrated, how errors are recovered, and how evaluation signals feed back.

HarnessX is a **harness foundry**: forge any number of agent harnesses from reusable processors and bundles, pair each with any model, and evolve them through training — all without rewriting the agent.

Most frameworks solved model swapping. **Behavior swapping** remains expensive — switching from a coding agent to a research agent, adding memory or guardrails, means rewriting the agent.

HarnessX solves this with one clean separation:

```python
agent = model.agentic(harness)
```

- `ModelConfig` — provider routing, fallback, per-role model assignment
- `HarnessConfig` — the full behavior pipeline (tools, memory, processors, trace, sandbox)

The **X** in Harness**X** stands for e**X**tensible Behavior Composition — compose, adapt, and evolve harnesses without rewriting the agent:

🧩 **Compose** — 9-dimension behavior pipeline; any behavior = Processor, combine with `|` operator.

⚙️ **Adapt** — Harness observes performance and auto-searches optimal harness configurations.

🚀 **Evolve** — every run produces reward-annotated trajectories that feed SFT / RL training.

---

<a id="architecture"></a>
## 🏗️ Architecture

<p align="center">
  <img src="docs/assets/harnessx_architecture.png" alt="HarnessX Architecture" width="800"/>
</p>

→ See **[docs/architecture.md](docs/architecture.md)** for the full 9-dimension behavior pipeline, processor hook points, and composition API.

---

<a id="quick-start"></a>

## 🚀 Quick Start

<details>
<summary>Click to expand</summary>

### Install

**One-click install** (interactive — asks before installing uv, Node.js, and optional IM Gateway):

```bash
curl -sSf https://raw.githubusercontent.com/Darwin-Agent/HarnessX/main/scripts/install.sh | bash
```

**Non-interactive — install everything without prompts:**

```bash
curl -sSf https://raw.githubusercontent.com/Darwin-Agent/HarnessX/main/scripts/install.sh | bash -s -- --all
```

Both commands install uv, Python 3.12, harnessx, and (with Node.js available) the Harness Lab frontend.
After installation, reload your shell or run `source ~/.bashrc` (or `~/.zshrc` on macOS).

<details>
<summary>Manual install with <a href="https://docs.astral.sh/uv/">uv</a></summary>

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
# Build frontend (required for hx lab)
cd frontend && npm install && npm run build && cd ..
```

</details>

### CLI

```bash
export ANTHROPIC_API_KEY=sk-...

hx "Research 2026 AI agent trends and write a structured report"
hx -p "Write a Python fizzbuzz"     # non-interactive, print and exit
hx -c path/to/config.yaml           # load a YAML config
hx --resume <run_id>                # resume a previous session
hx lab                              # open the Lab UI at localhost:8000
```

### IM Gateway

Connect your agent to Feishu, Telegram, Slack, Discord, or DingTalk with a single service.
The gateway ships with a built-in React console for managing channels, sessions, and workspaces.

```bash
hx-gateway start   # start the gateway (configured in ~/.harnessx/gateway.yaml)
```

→ See **[gateway/README.md](gateway/README.md)** for setup, channel configuration, and architecture.

### Python SDK

<details>
<summary>Minimal runnable example</summary>

```python
import asyncio
from harnessx import BaseTask, HarnessConfig
from harnessx.core.model_config import ModelConfig
from harnessx.providers.anthropic_provider import AnthropicProvider

async def main():
    model = ModelConfig(main=AnthropicProvider("claude-sonnet-4-6"))
    harness = model.agentic(HarnessConfig())
    result = await harness.run(BaseTask(description="What is 2 + 2?"))
    print(result.final_output)

asyncio.run(main())
```

</details>

</details>

---

<a id="benchmarks"></a>
## 📊 Benchmarks

HarnessX provides two evolution loops that systematically improve agent performance on any benchmark:

- **Harness Evolution** — a meta-harness analyzes trajectories and automatically searches for better processor combinations, prompt strategies, and tool configurations, *without changing the model*.
- **Model Evolution** — reward-annotated trajectories from harness runs feed RL fine-tuning (via [VERL](https://github.com/volcengine/verl)), improving the model itself.

The two loops compose: evolve the harness first, then evolve the model on top. Below are results on the [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) benchmark. See [`benchmarks/README.md`](benchmarks/README.md) for additional benchmarks and adapter details.

### Harness Evolution (Qwen 3.5 9B)

Starting from a default harness (R0, 33%), the meta-harness discovers better configurations round by round — reaching **47%** by R3, a **+14pp gain** with zero model changes. → Reproduce: [`recipe/gaia_evolver/`](recipe/gaia_evolver/)

<p align="center">
  <img src="docs/assets/Harness_Evolution_Config.png" alt="Harness Evolution Config — Round 0 to Round 3" width="800"/>
</p>

### Harness Evolution (GPT-5)

The same approach scales to frontier models. Overall GAIA accuracy rises from 62% to **84%** after evolution, with gains across all five domains. → Reproduce: [`recipe/gaia_evolver/`](recipe/gaia_evolver/)

<p align="center">
  <img src="docs/assets/Harness_Evolution.png" alt="Harness Evolution — Per-Domain Accuracy" width="700"/>
</p>

### Model-Harness Co-Evolution (Qwen 3.5 9B)

When the two loops run together, the gains compound: harness evolution lifts the baseline from 33.97% to 41.67%; model evolution pushes it further to **55.77%** — a **+64% relative improvement**, all on a 9B model. → Reproduce: [`recipe/verl_harnessX/`](recipe/verl_harnessX/)

<p align="center">
  <img src="docs/assets/HarnessX_Model_Co_evolution.png" alt="Model-Harness Co-Evolution" width="700"/>
</p>

---

<a id="structure"></a>
## 📁 Project Structure

```
HarnessX/
├── harnessx/                  # 🧠 Core framework
│   ├── core/                  #    Harness, Builder, RunLoop, State, Events, Trajectory
│   ├── processors/            #    7 categories × multiple processors
│   │   ├── context/           #    📝 System prompt, history, user wrapper
│   │   ├── control/           #    🛡️ 13 safety & reliability processors
│   │   ├── evaluation/        #    📊 LLM judge, PRM, self-verify
│   │   ├── memory/            #    🧠 Extraction, retrieval, 5 strategies
│   │   ├── multi_model/       #    🔗 Model routing
│   │   ├── observability/     #    🔭 OTel, checkpoints, metrics
│   │   └── tools/             #    🔧 Skill loader, schema adapter, filters
│   ├── providers/             # 🔌 6 model backends + agentic mixin
│   ├── plugins/               # 🧩 Plugin base, discovery, builtins, dimensions
│   │   └── dimensions/
│   │       └── light_memory/  # 🧠 Light-Memory (self-developed)
│   ├── tools/                 # ⚒️ Tool registry, builtins
│   ├── sandbox/               # 📦 Local, Docker, E2B
│   ├── tracing/               # 📡 Journal, OTel, null tracer
│   ├── rl/                    # 🧬 RLConfigSpec, TaskBuilder
│   ├── bundles/               # 📦 Pre-composed capability bundles
│   ├── api/                   # 🌐 FastAPI + SSE for Lab UI
│   └── cli.py                 # ⌨️ CLI entry point (hx)
├── benchmarks/                # 📊 4 integrated + 3 ongoing benchmarks
├── recipe/                    # 🧪 slime (RL training recipe)
├── examples/                  # 📖 coding / research / assistant / custom_processor
├── extensions/                # 🔌 Skills (docx, pdf, pptx, xlsx)
├── frontend/                  # 🖥️ Lab UI (React + TypeScript + Tailwind)
└── tests/                     # ✅ Unit, integration, E2E
```

---

<a id="roadmap"></a>
## 🗺️ Roadmap

> For detailed design notes and motivation behind planned items, see [ROADMAP](docs/ROADMAP.md).

| Phase | Focus | Status |
|:-----:|-------|:------:|
| **1** | Core: 9-dimension behavior pipeline, 13 processors, multi-provider, SFT/RL bridge, 4 benchmarks, Lab UI | ![current](https://img.shields.io/badge/-current-22c55e?style=flat-square) |
| **2** | Meta-opt: Bayesian Optimization, Meta-Harness, auto config search | ![in progress](https://img.shields.io/badge/-in%20progress-f59e0b?style=flat-square) |
| **3** | Self-evolution: closed-loop training, HarnessHUB community marketplace | ![planned](https://img.shields.io/badge/-planned-8b5cf6?style=flat-square) |
| **4** | Memory: multimodal backends, third-party integrations (VERL, SuperMemory, OpenVKing) | ![planned](https://img.shields.io/badge/-planned-8b5cf6?style=flat-square) |

### In-Repo Implementations

- [x] **[Light-Memory](docs/feats/light-memory.md)** — file-based memory with time-decay, daily compression, git versioning (`harnessx/plugins/dimensions/light_memory/`)
- [x] **Slime RL recipe** — SGLang rollout adapter + token annotation + GRPO training pipeline (`recipe/slime/`)
- [x] **MetaHarness** — agent observes its own trajectories and proposes harness config changes; observer harness + meta-agent + sandboxed promotion loop
- [ ] **LoCoMo benchmark** — long-context memory evaluation: session recall, cross-turn consistency, compaction fidelity
- [ ] **Bayesian Optimization** — surrogate model search over the ~10^6-configuration harness space
- [ ] **HarnessHUB** — community platform to publish, version, and pull `HarnessConfig` bundles (`hx pull coding-agent@v1.2`; Lab UI panel; private registries)
- [ ] **Multimodal Memory** — CLIP-based image/video memory backend via the plugin system
- [ ] **Harness Memory Evolution** — closed loop: trajectories → RL fine-tuning → better model → better harness; population-level mutation + data flywheel

### Third-Party Integrations *(opt-in, live in `recipe/`)*

- [x] **[VERL](https://github.com/volcengine/verl)** — connect HarnessX rollouts to distributed PPO / GRPO training loops
- [ ] **[MemPalace](https://github.com/mem-palace/mem-palace)** — structured episodic memory backend
- [ ] **[SuperMemory](https://supermemory.ai)** — cloud-backed semantic memory via the plugin system
- [ ] **[OpenVKing](https://github.com/openvking)** — vector-knowledge-graph memory for entity-rich domains
- [ ] **Memory quality metrics** — retrieval precision / recall surfaced through HarnessJournal
- [ ] **Data synthesis pipeline** — controlled SFT / preference-dataset generation with diversity constraints

---

## 🤝 Contributing

HarnessX is **fully open-source** under the MIT License. Contributions are welcome for:

- 🧩 **New processors** — behavior modules for unexplored dimensions
- 🧠 **New memory backends** — via the plugin system
- 📊 **New benchmark adapters** — `benchmarks/` pattern
- 🧪 **RL training recipes** — `recipe/`
- 🖥️ **Lab UI improvements**

Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

```bibtex
@software{harnessx2026,
  title   = {HarnessX: A Composable, Self-Evolving Agent Harness Foundry},
  author  = {Darwin Agent Team},
  year    = {2026},
  url     = {https://github.com/Darwin-Agent/HarnessX},
  license = {MIT},
}
```

---

<div align="center">
  <strong>HARNESS</strong><strong>X</strong> — <em>Compose. Adapt. Evolve.</em>
  <br/>
  <sub>Built with care by the <strong>Darwin Agent Team</strong></sub>
</div>
