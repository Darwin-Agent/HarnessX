# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""verl RL backend recipe — uses HarnessX RunLoop for trajectory generation.

Entry points for verl's AgentLoopBase:
    recipe.verl.agent_loop.HarnessXAgentLoop  (registered as "harnessx_full")

The agent loop internally calls:
    recipe.verl.harness_rollout.generate()  — same pattern as slime backend
    recipe.verl.harness_rollout.reward_func()

Directory layout mirrors recipe/slime/:
    spec.py          — VerlConfigSpec (extends RLConfigSpec)
    registry.py      — task routing: load_harness_config(sample)
    harness.py       — make_verl_harness(spec, provider, task)
    harness_rollout.py — generate() / reward_func()
    research/        — GAIA research task (config, tools, evaluator, rewards, prompt)
    formats/         — VerlRLFormat (trajectory → verl token format)
    processor/       — EarlyCommitProcessor, EvaluationProcessor
    dependencies/    — pinned upstream metadata, licenses, and auditable patch
    launch/          — launch scripts and config.yaml
"""
