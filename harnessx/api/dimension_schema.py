# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

_CTX = "harnessx.processors.context"
_MEM = "harnessx.processors.memory"
_CTRL = "harnessx.processors.control"
_EVAL = "harnessx.processors.evaluation"
_OBS = "harnessx.processors.observability"
_TOOLS = "harnessx.processors.tools"
_MULTI = "harnessx.processors.multi_model"
_STRAT = f"{_CTX}.strategies"
_MEM_ST = f"{_MEM}.strategies"
_EVAL_ST = f"{_EVAL}.strategies.evaluators"

DIMENSION_SCHEMA: list[dict] = [
    # ── Context processors (multi-select) ────────────────────────────────────
    {
        "key": "context",
        "label": "Context",
        "description": "Processors that assemble and wrap the model's input context",
        "icon": "file_text",
        "multi_select": True,
        "options": [
            {
                "key": "system_prompt",
                "label": "System Prompt",
                "description": "Build a task-level system prompt each turn",
                "processors": [
                    {"_target_": f"{_CTX}.system_prompt.SystemPromptProcessor"},
                ],
            },
            {
                "key": "user_wrapper",
                "label": "User Wrapper",
                "description": "Wrap user message with a structured format",
                "processors": [
                    {"_target_": f"{_CTX}.user_wrapper.UserWrapperProcessor"},
                ],
            },
            {
                "key": "env_injection",
                "label": "Env Injection",
                "description": "Inject deterministic environment facts block",
                "processors": [
                    {"_target_": f"{_CTX}.env_context_injector.EnvironmentContextInjector"},
                ],
            },
            {
                "key": "compaction",
                "label": "Compaction",
                "description": "Summarise and compact context when approaching token limit",
                "processors": [
                    {
                        "_target_": f"{_CTRL}.compaction.CompactionProcessor",
                        "token_threshold": 80_000,
                    },
                    {"_target_": f"{_CTRL}.token_budget.TokenBudgetProcessor"},
                ],
                "params": [
                    {
                        "label": "Token threshold",
                        "type": "int",
                        "min": 20_000,
                        "max": 200_000,
                        "default": 80_000,
                        "step": 10_000,
                        "targets": [
                            {
                                "processor_target": f"{_CTRL}.compaction.CompactionProcessor",
                                "path": "token_threshold",
                            },
                        ],
                    },
                ],
            },
        ],
    },
    # ── Memory (single-select strategy) ──────────────────────────────────────
    {
        "key": "memory",
        "label": "Memory",
        "description": "Long-term recall stored and retrieved across steps",
        "icon": "brain",
        "options": [
            {
                "key": "none",
                "label": "None",
                "description": "No long-term memory",
                "processors": [],
            },
            {
                "key": "sliding_window",
                "label": "Sliding Window",
                "description": "Keep the most recent N memory entries",
                "processors": [
                    {
                        "_target_": f"{_MEM}.memory_extraction.MemoryExtractionProcessor",
                        "memory": {
                            "_target_": f"{_MEM_ST}.sliding_window.SlidingWindowMemory",
                            "n": 40,
                        },
                        "threshold": 80_000,
                    },
                    {
                        "_target_": f"{_MEM}.memory_retrieval.MemoryRetrievalProcessor",
                        "memory": {
                            "_target_": f"{_MEM_ST}.sliding_window.SlidingWindowMemory",
                            "n": 40,
                        },
                        "top_k": 10,
                    },
                ],
                "params": [
                    {
                        "label": "Window size",
                        "type": "int",
                        "min": 5,
                        "max": 200,
                        "default": 40,
                        "step": 5,
                        "targets": [
                            {
                                "processor_target": f"{_MEM}.memory_extraction.MemoryExtractionProcessor",
                                "path": "memory.n",
                            },
                            {
                                "processor_target": f"{_MEM}.memory_retrieval.MemoryRetrievalProcessor",
                                "path": "memory.n",
                            },
                        ],
                    },
                ],
            },
        ],
    },
    # ── Control processors (multi-select) ────────────────────────────────────
    {
        "key": "control",
        "label": "Control",
        "description": "Individual reliability and correction processors",
        "icon": "shield_check",
        "multi_select": True,
        "options": [
            {
                "key": "loop_detection",
                "label": "Loop Detection",
                "description": "Halt on repeating step fingerprints",
                "processors": [
                    {"_target_": f"{_CTRL}.loop_detection.LoopDetectionProcessor"},
                ],
            },
            {
                "key": "tool_call_correction",
                "label": "Tool Call Correction",
                "description": "Heuristic parameter fixups before dispatch",
                "processors": [
                    {"_target_": f"{_CTRL}.tool_call_correction.ToolCallCorrectionLayer"},
                ],
            },
            {
                "key": "parse_retry",
                "label": "Parse Retry",
                "description": "Retry model on structurally invalid tool calls",
                "processors": [
                    {"_target_": f"{_CTRL}.parse_retry.ParseRetryProcessor"},
                ],
            },
            {
                "key": "tool_failure_guard",
                "label": "Tool Failure Guard",
                "description": "Stop runaway retry loops after N failures",
                "processors": [
                    {"_target_": f"{_CTRL}.tool_failure_guard.ToolFailureGuard"},
                ],
            },
            {
                "key": "repeated_edit_detector",
                "label": "Repeated Edit Detector",
                "description": "Inject hint after N edits to the same file",
                "processors": [
                    {"_target_": f"{_CTRL}.repeated_edit_detector.RepeatedFileEditDetector"},
                ],
            },
            {
                "key": "bg_install_guard",
                "label": "BG Install Guard",
                "description": "Block background package install commands",
                "processors": [
                    {"_target_": f"{_CTRL}.bg_install_guard.BgInstallGuard"},
                ],
            },
            {
                "key": "self_verify",
                "label": "Self-Verify",
                "description": "Verification pass required before task exit",
                "processors": [
                    {"_target_": f"{_CTRL}.self_verify.SelfVerifyProcessor"},
                ],
                "conflicts": [
                    {
                        "if_processor": f"{_EVAL_ST}.self_verify.SelfVerifyEvaluator",
                        "message": "Self-Verify processor duplicates evaluation=self_verify; one is redundant.",
                        "severity": "warning",
                    },
                ],
            },
            {
                "key": "cost_guard",
                "label": "Cost Cap",
                "description": "Hard USD spend limit per run",
                "processors": [
                    {
                        "_target_": f"{_CTRL}.cost_guard.CostGuardProcessor",
                        "max_usd": 5.0,
                    },
                ],
                "params": [
                    {
                        "label": "Max USD",
                        "type": "float",
                        "min": 0.1,
                        "max": 50.0,
                        "default": 5.0,
                        "step": 0.5,
                        "targets": [
                            {
                                "processor_target": f"{_CTRL}.cost_guard.CostGuardProcessor",
                                "path": "max_usd",
                            },
                        ],
                    },
                ],
            },
        ],
    },
    # ── Evaluation (single-select) ────────────────────────────────────────────
    {
        "key": "evaluation",
        "label": "Evaluation",
        "description": "Automatic pass/fail verdict after each run",
        "icon": "check_circle",
        "options": [
            {
                "key": "none",
                "label": "None",
                "description": "No automatic evaluation",
                "processors": [],
            },
            {
                "key": "self_verify",
                "label": "Self-Verify",
                "description": "Token-overlap check against success criteria",
                "processors": [
                    {
                        "_target_": f"{_EVAL}.evaluation.EvaluationProcessor",
                        "evaluator": {
                            "_target_": f"{_EVAL_ST}.self_verify.SelfVerifyEvaluator",
                        },
                    },
                ],
            },
            {
                "key": "llm_judge",
                "label": "LLM Judge",
                "description": "Semantic verdict via a separate judge model call",
                "processors": [
                    {
                        "_target_": f"{_EVAL}.evaluation.EvaluationProcessor",
                        "evaluator": {
                            "_target_": f"{_EVAL_ST}.llm_judge.LLMJudgeEvaluator",
                        },
                    },
                ],
                "conflicts": [
                    {
                        "if_processor": f"{_MULTI}.sycophancy_detector.SycophancyDetector",
                        "message": "LLM Judge needs a second model — add Multi-Model: Contrarian.",
                        "severity": "error",
                        "negate": True,  # conflict fires when the processor is NOT present
                    },
                ],
            },
        ],
    },
    # ── Observability processors (multi-select) ───────────────────────────────
    {
        "key": "observability",
        "label": "Observability",
        "description": "Telemetry and checkpoint persistence",
        "icon": "activity",
        "multi_select": True,
        "options": [
            {
                "key": "otel",
                "label": "OpenTelemetry",
                "description": "Emit spans and traces to OTLP collector",
                "processors": [
                    {"_target_": f"{_OBS}.otel_proc.OTelProcessor"},
                ],
            },
            {
                "key": "checkpoint",
                "label": "Checkpoint",
                "description": "SQLite step checkpoints for crash recovery",
                "processors": [
                    {"_target_": f"{_OBS}.checkpoint.CheckpointProcessor"},
                ],
            },
        ],
    },
    # ── Multi-model (single-select) ───────────────────────────────────────────
    {
        "key": "multi_model",
        "label": "Multi-Model",
        "description": "Secondary model routing strategy",
        "icon": "git_fork",
        "options": [
            {
                "key": "single",
                "label": "Single",
                "description": "One model for all calls",
                "processors": [],
            },
            {
                "key": "contrarian",
                "label": "Contrarian",
                "description": "Second model critiques and debates the first",
                "processors": [
                    {"_target_": f"{_CTRL}.sycophancy_detector.SycophancyDetector"},
                ],
            },
            {
                "key": "query_router",
                "label": "Query Router",
                "description": "Classify task complexity and route between small/main models",
                "processors": [
                    {"_target_": f"{_MULTI}.model_router.ModelRouterProcessor"},
                ],
                "params": [
                    {
                        "label": "Confidence threshold",
                        "type": "float",
                        "min": 0.0,
                        "max": 1.0,
                        "default": 0.7,
                        "step": 0.05,
                        "targets": [
                            {
                                "processor_target": f"{_MULTI}.model_router.ModelRouterProcessor",
                                "path": "confidence_threshold",
                            },
                        ],
                    },
                    {
                        "label": "Router token budget",
                        "type": "int",
                        "min": 128,
                        "max": 4096,
                        "default": 512,
                        "step": 64,
                        "targets": [
                            {
                                "processor_target": f"{_MULTI}.model_router.ModelRouterProcessor",
                                "path": "router_token_budget",
                            },
                        ],
                    },
                ],
            },
        ],
    },
    # ── Tools (multi-select) ──────────────────────────────────────────────────
    {
        "key": "tools",
        "label": "Tools",
        "description": "Skill auto-loading",
        "icon": "wrench",
        "multi_select": True,
        "options": [
            {
                "key": "skill_loader",
                "label": "Skill Loading",
                "description": "Automatically discover and install skills",
                "processors": [
                    {"_target_": f"{_TOOLS}.skill_loader.ProgressiveSkillLoader"},
                ],
            },
        ],
    },
]
