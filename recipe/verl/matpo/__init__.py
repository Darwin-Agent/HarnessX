# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""MATPO — Multi-Agent Token-level Policy Optimization for verl backend.

Subagent tokens participate in loss computation. Main agent's advantage
is inherited by subagents. Padding ensures batch divisibility.

MATPO is not a separate algorithm — it's integrated into GRPO.
When subagent rows are present in the batch (is_subagent=True),
GRPO's compute_advantage automatically delegates to compute_matpo_advantage.
"""
