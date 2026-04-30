"""recipe/slime — HarnessX + Slime RL training integration.

Implements Slime's custom generate() / reward_func() interfaces using
HarnessX tool execution infrastructure.

Entry points for Slime (--custom-generate-function-path / --custom-rm-path):
    --custom-generate-function-path recipe.slime.harness_rollout.generate
    --custom-rm-path                recipe.slime.harness_rollout.reward_func

Task registry (add new tasks here, no harness_rollout.py changes needed):
    recipe/slime/registry.py  → HARNESS_CONFIGS dict + load_harness_config()

Task domains:
    recipe/slime/math/  → MathTaskBuilder, MathBoxedEvaluator, RetoolCompatPRM, …
"""
