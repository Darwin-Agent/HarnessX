# Comparison

| Capability | HarnessX | LangChain / CrewAI | Claude Code | OpenClaw |
|---|:---:|:---:|:---:|:---:|
| Composable behavior modules | **9-dimension processor pipeline** | Limited | No | Partial |
| Single-module ablation testing | **First-class** | Hard | No | Partial |
| Trajectory → SFT + RL export | **Yes** | No | No | No |
| Meta-optimization (BO + Meta-Harness) | **BO + planned** | No | No | No |
| Benchmark integration | **4 integrated adapters** | Manual | Manual | Manual |
| Sandbox isolation | **Local / Docker / E2B** | Limited | Yes | Docker |
| Multi-provider routing | **Role-based ModelConfig** | Manual | No | Limited |
| Config reproducibility | **Content-addressed YAML** | No | No | No |
| Session recovery | **Journal + wake()** | No | Partial | No |
| RL training bridge | **SGLang + TokenAnnotation** | No | No | No |
| Zero-code Lab UI | **Yes** | No | No | No |
| Self-developed memory system | **Light-Memory** | No | No | No |
| Process Reward Model | **4 strategies** | No | No | No |

> [!NOTE]
> **Where each shines:** LangChain/CrewAI → rapid prototyping & community ecosystem. OpenClaw → production multi-channel infra. Claude Code → best developer UX. HarnessX → composability + trainability + meta-optimization for systematic harness search.
