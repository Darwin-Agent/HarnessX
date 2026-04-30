#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================
#  HarnessX GRPO Training Script — 8×H100, Qwen3.5-9B
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # Directory of this script
HARNESSX_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                     # Project root
VERL_ROOT="$HARNESSX_ROOT/verl_harnessX/verl"                    # veRL framework root
DATA_DIR="${DATA_DIR:-/path/to/data}"                             # Training/validation data directory
MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3_5-9B}"                     # Base model path (Qwen3.5-9B)
CKPT_DIR="${CKPT_DIR:-/path/to/checkpoints/gaia_qwen35_9b}"       # Checkpoint save directory

export PYTHONPATH="$HARNESSX_ROOT:$VERL_ROOT:${PYTHONPATH:-}"    # Python module search path: project root + veRL
export PYTHONUNBUFFERED=1                                         # Disable Python output buffering for real-time logs

# ---------- Environment ----------
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"       # NCCL communication log level (WARN/INFO/TRACE)
export TOKENIZERS_PARALLELISM=false           # Disable tokenizer multithreading to avoid fork deadlocks

# ---------- API Keys ----------
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"      # OpenAI-compatible API key
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-your-api-key-here}"  # Anthropic API key

# ---------- API Endpoints ----------
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://your-api-endpoint/v1}"   # OpenAI-compatible API endpoint
export ANTHROPIC_API_BASE="${ANTHROPIC_API_BASE:-http://your-api-endpoint}" # Anthropic API endpoint

# ---------- MCP Search Service ----------
export MCP_SEARCH_URL="${MCP_SEARCH_URL:-https://your-mcp-search-endpoint/mcp}"  # MCP search service endpoint (WebSearch primary engine)
export MCP_SEARCH_KEY="${MCP_SEARCH_KEY:-your-mcp-key-here}"                      # MCP authentication key

# ---------- Timeout Architecture ----------
# 2-layer timeout: L1 ToolCall → L2 Tool Internal

# ---------- Tool Settings ----------
# Concurrency control
export VERL_TOOL_CONCURRENCY_PER_WORKER="${VERL_TOOL_CONCURRENCY_PER_WORKER:-18}"  # Per-worker tool concurrency semaphore slots
export VERL_TOOL_CALL_TIMEOUT="${VERL_TOOL_CALL_TIMEOUT:-12}"                      # L1: Per tool call hard timeout (seconds); cancelled via asyncio on expiry

# Browser tool (Playwright headless Chromium)
export VERL_BROWSER_POOL_SIZE="${VERL_BROWSER_POOL_SIZE:-2}"       # Per-worker browser page pool size
export VERL_BROWSER_PAGE_TIMEOUT="${VERL_BROWSER_PAGE_TIMEOUT:-2000}"   # L2: Page navigation timeout (ms)
export VERL_BROWSER_CLICK_TIMEOUT="${VERL_BROWSER_CLICK_TIMEOUT:-2000}" # L2: Element click wait timeout (ms)

# WebFetch tool (httpx static + Playwright browser fallback)
export VERL_WEBFETCH_STATIC_TIMEOUT="${VERL_WEBFETCH_STATIC_TIMEOUT:-5}"    # L2: Static httpx request timeout (seconds)
export VERL_WEBFETCH_BROWSER_TIMEOUT="${VERL_WEBFETCH_BROWSER_TIMEOUT:-5}"  # L2: Browser fallback timeout (seconds); used when static content < 200 chars

# WebSearch tool (MCP primary + DuckDuckGo fallback)
export VERL_WEBSEARCH_TIMEOUT="${VERL_WEBSEARCH_TIMEOUT:-5}"      # L2: Primary engine (MCP/SerpAPI/Tavily) timeout (seconds)
export VERL_WEBSEARCH_DDG_TIMEOUT="${VERL_WEBSEARCH_DDG_TIMEOUT:-5}" # L2: DuckDuckGo fallback timeout (seconds)

# Bash tool (shell command execution)
export VERL_BASH_TIMEOUT_MS="${VERL_BASH_TIMEOUT_MS:-6000}"      # L2: Command execution timeout (ms)

# CodeInterpreter tool (Python sandboxed execution)
export VERL_CODE_TIMEOUT="${VERL_CODE_TIMEOUT:-5}"                # L2: Code execution timeout (seconds)
export VERL_CODE_CONCURRENCY="${VERL_CODE_CONCURRENCY:-8}"        # Max concurrent Python processes (shared across all workers)

# ---------- WandB ----------
export WANDB_API_KEY="${WANDB_API_KEY:-}"       # Weights & Biases API key
export WANDB_MODE="${WANDB_MODE:-offline}"      # WandB mode: offline=local logging, online=cloud upload

# ---------- Log directory ----------
LOG_DIR="${LOG_DIR:-/path/to/logs/qwen35_9b_harness}"               # Training log output directory
mkdir -p "$LOG_DIR" "$CKPT_DIR"

# ============================================================
#  Launch — Override config.yaml defaults via Hydra CLI
# ============================================================
python3 "$SCRIPT_DIR/main.py" \
    --config-path="$SCRIPT_DIR" \
    --config-name=config \
    \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="['$DATA_DIR/val.parquet']" \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=24486 \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path="$MODEL_DIR" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32678 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32678 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32678 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32678 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=30 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.critic_warmup=0 \
    trainer.project_name=harness_grpo \
    trainer.experiment_name=qwen35_9b_grpo \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.logger="['console']" \
    trainer.rollout_data_dir="$CKPT_DIR/rollout_saved" \
    trainer.validation_data_dir="$CKPT_DIR/val_saved" \
    \
    "$@" \
    2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# ============================================================
#  Parameter Quick Reference
# ============================================================
# data.train_files          — Training parquet file path
# data.val_files            — Validation parquet file path list
# data.train_batch_size     — Samples per training step (×n=8 rollouts = 512 sequences)
# data.max_prompt_length    — Max prompt token count; truncated if exceeded
# data.max_response_length  — Max generation tokens (across all multi-turn conversation turns)
# data.prompt_key           — Prompt column name in parquet
# data.reward_fn_key        — Column identifying the reward function in parquet
# data.return_raw_chat      — Return raw chat messages (not tokenized)
# data.return_multi_modal_inputs — Whether to return multi-modal inputs (text-only=False)
# data.filter_overlong_prompts   — Filter out samples exceeding max_prompt_length
# data.shuffle              — Shuffle training data each epoch
#
# actor_rollout_ref.hybrid_engine      — Hybrid engine: actor/rollout/ref share GPUs
# actor_rollout_ref.model.path         — Model weights path
# actor_rollout_ref.model.use_remove_padding — Remove padding for acceleration (FlashAttention)
# actor_rollout_ref.actor.optim.lr     — Actor learning rate
# actor_rollout_ref.actor.ppo_mini_batch_size  — GRPO update mini-batch size
# actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu — Per-GPU gradient accumulation micro-batch (1=min VRAM)
# actor_rollout_ref.actor.use_kl_loss  — Whether to add KL term to loss (GRPO typically disabled)
# actor_rollout_ref.actor.kl_loss_coef — KL loss coefficient
# actor_rollout_ref.actor.entropy_coeff — Entropy regularization coefficient (0=no extra exploration)
# actor_rollout_ref.actor.fsdp_config.param_offload     — FSDP parameter offload to CPU
# actor_rollout_ref.actor.fsdp_config.optimizer_offload  — FSDP optimizer state offload to CPU
#
# actor_rollout_ref.rollout.name       — Inference backend (sglang)
# actor_rollout_ref.rollout.mode       — async=overlap inference with training
# actor_rollout_ref.rollout.n          — Rollouts per prompt (GRPO group comparison)
# actor_rollout_ref.rollout.temperature — Sampling temperature (1.0=standard)
# actor_rollout_ref.rollout.top_p      — Nucleus sampling cutoff probability
# actor_rollout_ref.rollout.tensor_model_parallel_size — Tensor parallelism degree (1=single GPU, sufficient for 9B)
# actor_rollout_ref.rollout.gpu_memory_utilization     — SGLang KV cache VRAM fraction
# actor_rollout_ref.rollout.max_num_batched_tokens     — SGLang max tokens per batch
# actor_rollout_ref.rollout.enforce_eager  — False=allow CUDA Graph acceleration
# actor_rollout_ref.rollout.free_cache_engine — Release KV cache after rollout for training
# actor_rollout_ref.rollout.enable_chunked_prefill — Chunked prefill to reduce peak VRAM
# actor_rollout_ref.rollout.log_prob_* — Log probability computation batch/token config
#
# actor_rollout_ref.ref.*   — Reference model config (for KL divergence; same structure as rollout)
#
# algorithm.adv_estimator   — Advantage estimator: grpo (Group Relative Policy Optimization)
# algorithm.norm_adv_by_std_in_grpo — Normalize advantage by standard deviation
# algorithm.kl_ctrl.kl_coef — KL penalty coefficient (0=no constraint on ref model deviation)
#
# trainer.n_gpus_per_node   — GPUs per node
# trainer.nnodes            — Number of nodes (1=single machine)
# trainer.total_epochs      — Total training epochs
# trainer.save_freq         — Save checkpoint every N epochs
# trainer.test_freq         — Run validation every N epochs
# trainer.val_before_train  — Run validation before training (baseline)
# trainer.val_only          — True=validation only, no training
# trainer.critic_warmup     — Critic warmup steps (GRPO has no critic, set to 0)
# trainer.project_name      — WandB project name
# trainer.experiment_name   — WandB experiment name
# trainer.default_local_dir — Checkpoint save directory
# trainer.logger            — Logging backend ['console']
# trainer.rollout_data_dir  — Rollout debug data save directory
# trainer.validation_data_dir — Validation data save directory
