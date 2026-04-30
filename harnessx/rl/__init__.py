"""harnessx.rl — RL training abstractions.

Builds on core primitives (BaseTask, Processor, Trajectory) to provide
RL-specific configuration, task types, reward models, and harness assembly.

Usage::

    from harnessx.rl.task import RLTask, ProcessRewardModel, NullPRM
    from harnessx.rl.config import RLConfigSpec
    from harnessx.rl.builder import build_rl_harness_config
"""

from .task import (
    EnhancedToolSuccessPRM,
    NullPRM,
    ProcessRewardModel,
    RLEvaluator,
    RLTask,
    TaskBuilder,
)
from .config import RLConfigSpec
from .builder import build_rl_harness_config
from harnessx.plugins.dimensions.rl import RLControlPlugin

__all__ = [
    "RLTask",
    "TaskBuilder",
    "RLEvaluator",
    "ProcessRewardModel",
    "NullPRM",
    "EnhancedToolSuccessPRM",
    "RLConfigSpec",
    "build_rl_harness_config",
    "RLControlPlugin",
]
