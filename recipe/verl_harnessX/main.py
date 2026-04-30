"""
Custom entry point for verl_harnessX GRPO training.
"""

import hydra
import ray

from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device
from verl.experimental.reward_loop import migrate_legacy_reward_impl


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)

    task_runner_class = ray.remote(num_cpus=1)(TaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
