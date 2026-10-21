# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""verl entry point — GRPO training with HarnessX agent loop."""
from __future__ import annotations

import hydra
import ray
from omegaconf import OmegaConf
from pathlib import Path

from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


class HarnessXTaskRunner(TaskRunner):
    """TaskRunner — applies patches in Ray worker before model init."""

    def run(self, config):
        from recipe.verl.patches import apply_all
        apply_all()

        config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        OmegaConf.set_struct(config, False)
        super().run(config)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
    auto_set_device(config)

    import os
    OmegaConf.set_struct(config, False)

    # Propagate PYTHONPATH to Ray workers
    pythonpath = os.environ.get("PYTHONPATH", "")
    ray_env_vars = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env.env_vars", default=None)
    if ray_env_vars is None:
        OmegaConf.update(config, "ray_kwargs.ray_init.runtime_env.env_vars", {}, merge=True)
    if pythonpath:
        OmegaConf.update(config, "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH", pythonpath, merge=True)
    # Propagate HARNESSX_* env vars to Ray workers (tools need them)
    ray_env_allowlist = {"SERPAPI_API_KEY", "TAVILY_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE", "ANTHROPIC_API_KEY", "ANTHROPIC_API_BASE"}
    for k, v in os.environ.items():
        if k.startswith("HARNESSX_") or k.startswith("MCP_") or k.startswith("VERL_") or k in ray_env_allowlist:
            OmegaConf.update(config, f"ray_kwargs.ray_init.runtime_env.env_vars.{k}", v, merge=True)

    # Merge with verl defaults to fill missing keys
    import verl
    verl_defaults = Path(verl.__file__).parent / "trainer" / "config" / "_generated_ppo_trainer.yaml"
    if verl_defaults.exists():
        base = OmegaConf.load(verl_defaults)
        OmegaConf.set_struct(base, False)
        config = OmegaConf.merge(base, config)

    OmegaConf.set_struct(config, False)

    # Inject reward config if missing
    cfg_dict = OmegaConf.to_container(config, resolve=True)
    if "reward" not in cfg_dict or cfg_dict["reward"] is None:
        cfg_dict["reward"] = {
            "num_workers": 8,
            "custom_reward_function": cfg_dict.get("custom_reward_function"),
            "reward_manager": {"source": "register", "name": "naive",
                               "module": {"path": None, "name": "custom_reward_manager"}},
            "reward_model": {"enable": False},
        }
    config = OmegaConf.create(cfg_dict)
    OmegaConf.set_struct(config, False)

    task_runner_class = ray.remote(num_cpus=1)(HarnessXTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
