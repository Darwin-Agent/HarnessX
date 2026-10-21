# Harness and Model Evolution / Harness 与模型进化

This guide documents two distinct workflows. **Legacy Sequential Evolution** first evolves the Harness and then trains model weights; **Integrated Co-Evolution** runs HarnessX agent rollouts inside veRL GRPO training. They use different veRL recipes, dataset names, and execution models. Choose the workflow that matches your goal.

本指南说明两套不同流程：**旧版顺序进化**先进行 Harness 进化，再训练模型权重；**集成式协同进化**是在 veRL GRPO 训练中运行 HarnessX agent rollout。两者使用不同的 veRL recipe、数据文件名和执行方式，请根据目标选择。

## 1. Prerequisites / 前置条件

- Python 3.11+ and HarnessX dependencies. Install from this directory with `uv sync` or `pip install -e .`.
- The legacy GAIA stage needs API access for the selected task-agent/meta-agent models and a local GAIA JSON dataset matching `recipe/gaia_evolver/README.md`.
- Either GRPO route needs a CUDA/PyTorch environment with compatible veRL, Ray, and SGLang dependencies, a local base model, and prepared Parquet datasets. Both checked-in presets use long sequences and substantial GPU memory; tune their configs for your hardware.

- Python 3.11+ 和 HarnessX 依赖；在本目录执行 `uv sync` 或 `pip install -e .`。
- 旧版 GAIA 阶段需要所选 task-agent/meta-agent 模型的 API 权限，以及符合 `recipe/gaia_evolver/README.md` 格式的本地 GAIA JSON。
- 两种 GRPO 方式都需要兼容的 CUDA/PyTorch、veRL、Ray、SGLang 依赖、本地基础模型及准备好的 Parquet 数据。仓库预设序列较长、显存需求较高，请按硬件调整配置。

## 2. Standalone Harness Evolution / 独立 Harness 进化

GAIA Evolver evaluates tasks, asks a meta-agent to propose Harness configuration changes, and gates each round against the best prior score. This is the harness-evolution stage used by the Legacy Sequential workflow; running it alone does not train model weights.

GAIA Evolver 执行任务、由 meta-agent 提议 Harness 配置修改，并依据历史最佳分数决定保留或回退。这是旧版串行流程中的 Harness 进化阶段；单独运行它不会训练模型权重。

```bash
cd /path/to/HarnessX
export ANTHROPIC_API_KEY="<your-key>"
export ANTHROPIC_API_BASE="<optional-provider-endpoint>"
export OPENAI_API_KEY="<optional-key-if-your-selected-model-needs-it>"

python recipe/evolution_pipeline/run_gaia.py \
  --model "<provider/model-for-task-agent>" \
  --meta-model "<provider/model-for-meta-agent>" \
  --provider-id "<provider-id>" \
  --data-path "<local-gaia-json>" \
  --max-tasks 3 --num-rounds 2 --concurrency 1 --run-tag smoke-test
```

Use `python recipe/evolution_pipeline/run_gaia.py --help` for all options.

完整参数见 `python recipe/evolution_pipeline/run_gaia.py --help`。

## 3. Legacy Sequential Evolution / 旧版顺序进化

This two-stage workflow first runs GAIA meta-harness evolution, then trains model weights with the `recipe/verl_harnessX` GRPO recipe. The stages can also be run independently. The optional wrapper runs them in order and starts training only if GAIA succeeds.

这套两阶段流程先运行 GAIA 元 Harness 进化，再用 `recipe/verl_harnessX` GRPO 配方训练模型权重。两个阶段也可以独立运行；可选 wrapper 按顺序执行，只有 GAIA 成功后才启动训练。

```bash
cd /path/to/HarnessX

# Stage 1: Harness evolution. Export the API key(s) required by your provider.
export ANTHROPIC_API_KEY="<your-key>"
python recipe/evolution_pipeline/run_gaia.py \
  --model "<provider/model>" --meta-model "<provider/model>" \
  --provider-id "<provider-id>" --data-path "<local-gaia-json>" \
  --max-tasks 3 --num-rounds 2 --concurrency 1

# Stage 2: model GRPO training on separately prepared data.
export DATA_DIR="<directory-containing-train.parquet-and-val.parquet>"
export MODEL_DIR="<local-base-model-directory>"
export CKPT_DIR="<writable-checkpoint-directory>"
bash recipe/verl_harnessX/run_train.sh
```

For one-command orchestration, set `GAIA_DATA_PATH`, `GAIA_MODEL`, `GAIA_META_MODEL`, `GAIA_PROVIDER_ID`, `DATA_DIR`, `MODEL_DIR`, and `CKPT_DIR`, then run `bash recipe/evolution_pipeline/run_sequential.sh`. This wrapper does **not** convert GAIA trajectories into GRPO Parquet or automatically pass the evolved Harness config into model training. The legacy trainer expects `train.parquet` and `val.parquet` inside `DATA_DIR`.

如需一键按序编排，请设置 `GAIA_DATA_PATH`、`GAIA_MODEL`、`GAIA_META_MODEL`、`GAIA_PROVIDER_ID`、`DATA_DIR`、`MODEL_DIR` 和 `CKPT_DIR`，然后运行 `bash recipe/evolution_pipeline/run_sequential.sh`。该 wrapper **不会**把 GAIA 轨迹转换为 GRPO Parquet，也不会自动把进化后的 Harness 配置传给模型训练。旧版 trainer 要求 `DATA_DIR` 下有 `train.parquet` 和 `val.parquet`。

### Search service / 搜索服务

The legacy `recipe/verl_harnessX` trainer includes a **WebSearch** tool. It tries the optional MCP endpoint first, then SerpAPI, Tavily, and public DuckDuckGo fallbacks. Configure only the providers you intend to use; MCP URL/key and provider keys are optional, but at least one reachable backend is needed for live web search. For an MCP service:

旧版 `recipe/verl_harnessX` trainer 包含 **WebSearch** 工具。它按顺序尝试可选的 MCP endpoint、SerpAPI、Tavily，最后回退到公开的 DuckDuckGo 搜索。只配置你要使用的服务即可；MCP URL/key 和服务商 key 都是可选的，但要执行实时网页搜索，至少需要一个可访问的后端。MCP 服务配置示例：

```bash
export MCP_SEARCH_URL="<your-search-mcp-endpoint>"
export MCP_SEARCH_KEY="<your-mcp-key>" # omit if the service requires no auth
# Or configure a provider key; these providers are tried after MCP.
export SERPAPI_API_KEY="<your-serpapi-key>"
export TAVILY_API_KEY="<your-tavily-key>"
```

Set `VERL_WEBSEARCH_TIMEOUT`, `VERL_WEBSEARCH_DDG_TIMEOUT`, `VERL_WEBSEARCH_RETRIES`, and `VERL_WEBSEARCH_RETRY_DELAY` to tune the legacy tool. With no configured paid/MCP provider, the tool still attempts DuckDuckGo; that fallback depends on network access and may be unavailable in restricted environments.

旧版工具可通过 `VERL_WEBSEARCH_TIMEOUT`、`VERL_WEBSEARCH_DDG_TIMEOUT`、`VERL_WEBSEARCH_RETRIES` 和 `VERL_WEBSEARCH_RETRY_DELAY` 调整超时与重试。没有配置付费/MCP 服务时，工具仍会尝试 DuckDuckGo；该回退依赖网络，在受限环境中可能不可用。

## 4. Integrated Co-Evolution Recipe / 集成式协同进化配方

This separate integrated recipe lives under `recipe/verl/`. It runs HarnessX research-agent rollouts inside veRL GRPO training, so model updates use Harness-mediated task/tool interactions. Its entry point is `recipe/verl/launch/run_research_rl.sh`; it does **not** run the GAIA meta-agent stage from Section 3. The Harness task/tool recipe is selected by `HARNESSX_TASK_TYPE` and is not automatically rewritten by GAIA Evolver. This is a distinct integrated rollout/training setup, not an alias for the legacy sequential pipeline.

这套独立集成配方位于 `recipe/verl/`。它在 veRL GRPO 训练中运行 HarnessX research-agent rollout，使模型更新基于 Harness 执行任务/工具交互产生的轨迹。入口是 `recipe/verl/launch/run_research_rl.sh`；它**不会**运行第 3 节的 GAIA 元进化阶段。Harness 任务/工具配方由 `HARNESSX_TASK_TYPE` 选择，GAIA Evolver 不会自动改写它。这是独立的集成 rollout/训练方案，不是旧版顺序流程的别名。

```bash
cd /path/to/HarnessX
export DATA_DIR="<directory-containing-training-parquet-files>"
export MODEL_DIR="<local-base-model-directory>"
export CKPT_DIR="<writable-checkpoint-directory>"

# The checked-in config expects these exact filenames inside DATA_DIR.
test -f "$DATA_DIR/matpo_train_musique_converted.parquet"
test -f "$DATA_DIR/matpo_val_gaia_repeat_8_converted.parquet"

# Set one only when SGLang is not already installed/importable.
export SGLANG_ROOT="<your-sglang-checkout>"
# Or use SGLANG_PYTHON="<your-sglang-python-directory>" instead.

# Optional live-search providers; public Wikipedia/Bing/DDG fallbacks are also tried.
# export SERPAPI_API_KEY="<your-key>"
# export TAVILY_API_KEY="<your-key>"
# Configure model API credentials only if your selected agent/model needs them.
# export OPENAI_API_KEY="<your-key>"

bash recipe/verl/launch/run_research_rl.sh
```

Use CUDA/PyTorch, Ray, and SGLang versions compatible with the vendored veRL code, and tune `recipe/verl/launch/config.yaml` for your GPU count and memory. Checkpoint, rollout, and validation outputs are written under `CKPT_DIR`. The launcher leaves existing Ray/SGLang processes alone by default. Set `STOP_EXISTING_SERVICES=1` only on a dedicated node if you explicitly want it to stop existing services; do not enable it on a shared training host.

请使用与 vendored veRL 兼容的 CUDA/PyTorch、Ray、SGLang 环境，并按 GPU 数量和显存调整 `recipe/verl/launch/config.yaml`。Checkpoint、rollout 和验证输出写入 `CKPT_DIR`。启动脚本默认不停止已有 Ray/SGLang 进程；只有在专用节点且明确希望停止它们时才设置 `STOP_EXISTING_SERVICES=1`，共享训练主机不要启用。

**Search service / 搜索服务:** The integrated recipe exposes HarnessX `web_search_tool` directly when MATPO is disabled, and through its `searcher` sub-agent when MATPO is enabled. It supports optional SerpAPI and Tavily credentials, with public Wikipedia/Bing/DuckDuckGo fallbacks. Configure `SERPAPI_API_KEY` and/or `TAVILY_API_KEY` if desired. `MCP_SEARCH_URL` is not used by this integrated search tool. The separate `recipe/verl_harnessX` trainer has its own MCP-capable WebSearch implementation.

**搜索服务：** MATPO 关闭时，集成配方直接使用 HarnessX `web_search_tool`；MATPO 开启时则通过 `searcher` 子代理调用。该工具支持可选的 SerpAPI、Tavily 凭据，并可回退到公开 Wikipedia/Bing/DuckDuckGo 搜索。可配置 `SERPAPI_API_KEY` 和/或 `TAVILY_API_KEY`。集成搜索工具不使用 `MCP_SEARCH_URL`。独立的 `recipe/verl_harnessX` trainer 使用另一套支持 MCP 的 WebSearch 实现。
