# verl_harnessX

A multi-turn agentic GRPO (Group Relative Policy Optimization) training framework built on top of [veRL](https://github.com/volcengine/verl). It trains language models to use tools effectively through reinforcement learning with multi-turn conversations.

## Overview

verl_harnessX extends veRL's PPO/GRPO trainer to support **multi-turn agentic rollouts** — the model generates responses, invokes tools (web search, code execution, file reading, etc.), receives tool outputs, and continues reasoning across multiple turns. A reward function evaluates the final answer quality, format correctness, and tool usage efficiency.

Key features:
- **Multi-turn agent loop**: Models interact with tools over multiple conversation turns during rollout
- **Tool suite**: WebSearch, WebFetch, Browser, Bash, CodeInterpreter, Read
- **2-layer timeout architecture**: ToolCall → Tool Internal
- **Hybrid engine**: Actor/Rollout/Reference share GPUs via FSDP + SGLang async rollout
- **Configurable reward**: 0.8 accuracy + 0.1 format + 0.1 tool-call quality
- **Early-commit nudge**: Progressive urgency prompts to force answer commitment as turns deplete

## Project Structure

```
verl_harnessX/
├── main.py                  # Entry point — launches GRPO training via Hydra + Ray
├── config.yaml              # Hydra config (data, model, rollout, reward settings)
├── run_train.sh             # Training launch script with env vars and hyperparams
├── agent_loop.py            # Multi-turn agent loop (HarnessXAgentLoop)
├── agent_loop_config.yaml   # Agent loop registry config
├── dataset.py               # Custom dataset class (TextOnlyDataset) for chat-format prompts
├── reward.py                # Reward function (accuracy + format + tool-call scoring)
├── prompt.py                # System prompt for tool-augmented generation
├── processor/
│   └── early_commit.py      # Turn-budget-aware nudge injection at tool-response boundaries
├── tools/                   # Tool implementations
│   ├── base.py              # Tool decorator and registry
│   ├── web_search.py        # Multi-provider web search (MCP, SerpAPI, Tavily, DDG)
│   ├── web_fetch.py         # Web page fetching (static + browser fallback)
│   ├── browser.py           # Headless Playwright browser pool
│   ├── bash.py              # Shell command execution (sandboxed, timeout-controlled)
│   ├── code.py              # Python code interpreter (sandboxed execution)
│   ├── read.py              # File reader (text, PDF, DOCX, XLSX, PPTX)
│   └── _web_utils.py        # Shared web utilities
├── verl/                    # Vendored veRL framework (with agent rollout extensions)
└── __init__.py
```

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- 8× H100 GPUs (for full training; fewer GPUs possible with adjusted config)
- [veRL](https://github.com/volcengine/verl) dependencies (Ray, SGLang, FSDP, etc.)
- Playwright (for browser-based web fetching)

## Setup

1. **Configure paths** in `run_train.sh`:
   - `DATA_DIR`: Directory containing training/validation parquet files
   - `MODEL_DIR`: Path to the base model (e.g., Qwen3.5-9B)
   - `CKPT_DIR`: Checkpoint save directory
   - `LOG_DIR`: Training log directory

2. **Set API keys** (environment variables or in `run_train.sh`):
   - `OPENAI_API_KEY` / `OPENAI_API_BASE`: For OpenAI-compatible reward model endpoints
   - `ANTHROPIC_API_KEY` / `ANTHROPIC_API_BASE`: For Anthropic reward model endpoints
   - `MCP_SEARCH_URL` / `MCP_SEARCH_KEY`: For MCP-based web search
   - `SERPAPI_API_KEY` (optional): SerpAPI fallback
   - `TAVILY_API_KEY` (optional): Tavily fallback

3. **Install dependencies**:
   ```bash
   pip install -e ./verl
   pip install hydra-core omegaconf ray[default] sglang httpx html2text playwright
   playwright install chromium
   ```

## Usage

### Training

```bash
bash run_train.sh
```

Override any config parameter via Hydra CLI:
```bash
bash run_train.sh trainer.total_epochs=10 data.train_batch_size=32
```

### Validation Only

```bash
bash run_train.sh trainer.val_only=True
```

## Configuration

The training is configured via Hydra (`config.yaml` + CLI overrides). Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data.max_prompt_length` | Max prompt tokens | 8192 |
| `data.max_response_length` | Max response tokens (all turns combined) | 24486 |
| `actor_rollout_ref.rollout.n` | Rollouts per prompt (GRPO group size) | 8 |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | Max agent turns | 12 |
| `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` | Max tool output tokens | 2048 |
| `actor_rollout_ref.rollout.multi_turn.force_answer_remaining_tokens` | Force final answer threshold | 1000 |
| `algorithm.adv_estimator` | Advantage estimator | grpo |

## Architecture

### Timeout Hierarchy

Two nested timeout layers prevent runaway tool calls:

```
L1: VERL_TOOL_CALL_TIMEOUT (12s)  — single tool invocation
 └─ L2: Tool-specific timeouts    — internal per-tool limits
```

### Agent Loop

Each rollout sequence runs through `HarnessXAgentLoop`:

1. The model generates a response (with `<think>` reasoning)
2. Tool calls are parsed and executed against the tool registry
3. Tool outputs are appended as user messages
4. The early-commit processor injects urgency nudges as turns deplete
5. The loop repeats until the model emits `<answer>` tags or hits the turn limit

### Reward Function

The reward combines three components:
- **Accuracy (0.8)**: Correctness of final answer vs. ground truth (with 5% numeric tolerance)
- **Format (0.1)**: Proper tool call syntax, `<think>` blocks, answer formatting
- **Tool Call (0.1)**: Valid tool usage in the response

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VERL_TOOL_CALL_TIMEOUT` | Per-tool-call timeout (seconds) | 12 |
| `VERL_BASH_TIMEOUT_MS` | Bash tool default timeout (ms) | 240000 |
| `VERL_WEBSEARCH_TIMEOUT` | Web search provider timeout (seconds) | 30 |
| `VERL_WEBSEARCH_DDG_TIMEOUT` | DuckDuckGo fallback timeout (seconds) | 20 |
| `VERL_DOWNLOAD_DIR` | Directory for web fetch downloads | /tmp/verl_downloads |
| `HARNESSX_WORK_DIR` | Bash tool working directory root | /tmp/harnessx-work |

## License

This project builds on [veRL](https://github.com/volcengine/verl) (Apache 2.0).
