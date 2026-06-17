"""HarnessX — composable, trainable Agent Harness."""

# Apply defensive patch to anyio's asyncio CancelScope as early as possible.
# See harnessx/_anyio_patch.py for the why. Import for side effects only.
from . import _anyio_patch as _anyio_patch  # noqa: F401

from .core.events import (
    BeforeModelEvent,
    StepStartEvent,
    EvalResult,
    Event,
    Message,
    ModelResponseEvent,
    SegmentBoundaryEvent,
    StepEndEvent,
    TaskEndEvent,
    TaskStartEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    ToolSchema,
    Usage,
    make_run_id,
)
from .core.harness import BaseTask, Harness, HarnessConfig, HarnessResult
from .core.model_config import ModelConfig
from .core.processor import (
    Processor,
    ProcessorChain,
    MultiHookProcessor,
    after_model,
    after_tool,
    before_model,
    before_tool,
    step_start,
    on_step_end,
    on_task_end,
    pipe,
)
from .core.runloop import (
    BudgetExceededError,
    HarnessError,
    LoopDetectedError,
    ModelParseError,
)
from .core.state import State, StateSlot, PendingSubagent

from .processors.context.strategies.system_prompt import (
    DefaultSystemPromptBuilder,
    NullSystemPromptBuilder,
    TemplateSystemPromptBuilder,
)
from .processors.context.strategies.user_wrapper import (
    ChainOfThoughtWrapper,
    XMLFormatWrapper,
)

# Workspace
from .workspace.workspace import Workspace, WorkspaceEscapeError, WorkspaceWriteError
from .workspace.factory import build_spawn_tool

# Sandbox — pluggable execution environment
from .sandbox.base import Mount, Sandbox, SandboxProvider, get_current_sandbox
from .sandbox.local import LocalSandbox, LocalSandboxProvider

# Sub-agent spawning tool
from .tools.spawn_subagent import SPAWN_TOOL_NAME
from .tools.base import ToolConflictError

# Built-in web tools + default registry factories
from .tools.builtin import (
    web_search_tool,
    web_fetch_tool,
    browser_tool,
    build_web_tools,
    build_default_tools,
)

# Skills + WorkspaceInitializer
from .workspace.skill_index import SkillIndex, SkillMeta
from .workspace.initializer import WorkspaceInitializer

# Trajectory types
from .core.trajectory import (
    FullStateSnapshot,
    StateDelta,
    SlotOperation,
    StateSlotSnapshot,
    StatefulTrajectory,
    TrajectoryStep,
    TokenAnnotation,
)


# Application logger (loguru)
from .logging import logger, configure_logging
from .tracing.journal import HarnessJournal

# Context tool filters
from .processors.tools.strategies.tool_filter import (
    BaseToolFilter,
    AllowlistToolFilter,
    BlocklistToolFilter,
    TagToolFilter,
    CompositeToolFilter,
)

# Memory backends (commonly needed at the top level)
from .processors.memory.strategies.base import BaseMemory, compress_by_token_budget
from .processors.memory.strategies.sliding_window import SlidingWindowMemory
from .processors.memory.strategies.custom import InMemoryMemory, RedisMemory
from .processors.memory.strategies.summarization import SummarizationMemory
# Third-party adapters (OpenVking, SuperMemory, etc.) live in recipe/ at the
# project root — they are NOT part of the core package to avoid coupling to
# external frameworks. See recipe/openvking/, recipe/supermemory/, recipe/slime/.

# Evaluators
from .processors.evaluation.strategies.evaluators.self_verify import SelfVerifyEvaluator
from .processors.evaluation.strategies.evaluators.llm_judge import LLMJudgeEvaluator

# Process Reward Models
from .processors.evaluation.strategies.evaluators.prm import (
    ProcessRewardModel,
    TerminalPRM,
    DiscountedPRM,
    ToolSuccessPRM,
    LLMJudgePRM,
)


# Processors (commonly wired in via HarnessConfig.processors)
from .processors import (
    TokenBudgetProcessor,
    CostGuardProcessor,
    ParseRetryProcessor,
    ToolWhitelistProcessor,
    LoopDetectionProcessor,
    CheckpointProcessor,
    OTelProcessor,
    EvaluationProcessor,
)

# Utility functions from core (serialization + context management)
from .core.events import message_to_dict, dict_to_message, trim_messages_to_budget

# Aliases for progressive disclosure / shorter names
CoTUserWrapper = ChainOfThoughtWrapper
XMLUserWrapper = XMLFormatWrapper

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Events
    "Event",
    "StepStartEvent",
    "BeforeModelEvent",
    "ModelResponseEvent",
    "SegmentBoundaryEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "StepEndEvent",
    "TaskEndEvent",
    "TaskStartEvent",
    "Message",
    "ToolSchema",
    "ToolCall",
    "Usage",
    "EvalResult",
    "make_run_id",
    # Core
    "BaseTask",
    "Harness",
    "HarnessConfig",
    "HarnessResult",
    "ModelConfig",
    "Processor",
    "ProcessorChain",
    "MultiHookProcessor",
    "pipe",
    "step_start",
    "before_model",
    "after_model",
    "before_tool",
    "after_tool",
    "on_step_end",
    "on_task_end",
    "HarnessError",
    "BudgetExceededError",
    "LoopDetectedError",
    "ModelParseError",
    "State",
    "StateSlot",
    "PendingSubagent",
    # Context support utilities
    "DefaultSystemPromptBuilder",
    "NullSystemPromptBuilder",
    "TemplateSystemPromptBuilder",
    "ChainOfThoughtWrapper",
    "XMLFormatWrapper",
    "CoTUserWrapper",
    "XMLUserWrapper",
    # Workspace
    "Workspace",
    "WorkspaceEscapeError",
    "WorkspaceWriteError",
    "build_spawn_tool",
    "SPAWN_TOOL_NAME",
    # Tool errors
    "ToolConflictError",
    # Sandbox
    "Mount",
    "Sandbox",
    "SandboxProvider",
    "get_current_sandbox",
    "LocalSandbox",
    "LocalSandboxProvider",
    "web_search_tool",
    "web_fetch_tool",
    "browser_tool",
    "build_web_tools",
    "build_default_tools",
    "SkillIndex",
    "SkillMeta",
    "WorkspaceInitializer",
    # Trajectory
    "FullStateSnapshot",
    "StateDelta",
    "SlotOperation",
    "StateSlotSnapshot",
    "StatefulTrajectory",
    "TrajectoryStep",
    # Token-level annotation
    "TokenAnnotation",
    # Logger
    "logger",
    "configure_logging",
    # Tracing
    "HarnessJournal",
    # Tool filters
    "BaseToolFilter",
    "AllowlistToolFilter",
    "BlocklistToolFilter",
    "TagToolFilter",
    "CompositeToolFilter",
    # Memory backends
    "BaseMemory",
    "compress_by_token_budget",
    "SlidingWindowMemory",
    "InMemoryMemory",
    "RedisMemory",
    "SummarizationMemory",
    # Evaluators
    "SelfVerifyEvaluator",
    "LLMJudgeEvaluator",
    # Process Reward Models
    "ProcessRewardModel",
    "TerminalPRM",
    "DiscountedPRM",
    "ToolSuccessPRM",
    "LLMJudgePRM",
    # Processors
    "TokenBudgetProcessor",
    "CostGuardProcessor",
    "ParseRetryProcessor",
    "ToolWhitelistProcessor",
    "LoopDetectionProcessor",
    "CheckpointProcessor",
    "OTelProcessor",
    "EvaluationProcessor",
    # Core utilities
    "message_to_dict",
    "dict_to_message",
    "trim_messages_to_budget",
]
