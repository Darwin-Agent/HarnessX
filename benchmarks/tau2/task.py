"""Tau2Task and Tau2Evaluator — wraps tau2-bench as an HarnessX task."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from harnessx.core.harness import BaseTask
from harnessx.core.events import EvalResult

if TYPE_CHECKING:
    from harnessx.core.state import State

logger = logging.getLogger(__name__)

# Available tau2 domains
DOMAINS = ("airline", "retail", "telecom", "banking_knowledge", "mock")


@dataclass
class Tau2Task(BaseTask):
    """Wraps a tau2-bench task for HarnessX.

    The actual simulation is driven by tau2's orchestrator (user simulator,
    tool execution, evaluation).  This task holds configuration; the
    Tau2Evaluator runs the full tau2 pipeline.

    Requires: Python >= 3.12, tau2-bench installed via uv.
    """

    domain: str = "retail"
    tau2_task_id: str = ""
    task_split: str | None = None
    num_trials: int = 1

    # LLM configuration
    agent_model: str = "gpt-4.1"
    agent_api_base: str | None = None
    agent_api_key: str = "EMPTY"
    user_llm: str = "gpt-4.1"
    user_llm_args: dict = field(default_factory=dict)

    # tau2 simulation parameters
    tau2_max_steps: int = 100
    tau2_max_errors: int = 5
    tau2_seed: int | None = 42

    # Cached tau2 task object (not serialized)
    _tau2_task: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.tau2_task_id:
            raise ValueError("Tau2Task requires a tau2_task_id")
        if not self.description:
            self.description = f"tau2-bench {self.domain} task: {self.tau2_task_id}"

    @classmethod
    def from_domain(
        cls,
        domain: str,
        task_id: str | None = None,
        task_split: str | None = None,
        **kwargs: Any,
    ) -> "Tau2Task | list[Tau2Task]":
        """Load task(s) from a tau2 domain.

        Args:
            domain: One of 'airline', 'retail', 'telecom', 'banking_knowledge', 'mock'.
            task_id: Specific task ID, or None for all tasks in the split.
            task_split: Task split name (default: 'base').
            **kwargs: Passed to Tau2Task constructor (agent_model, user_llm, etc.).

        Returns:
            A single Tau2Task if task_id is given, else a list of Tau2Task.
        """
        try:
            from tau2.runner.helpers import get_tasks
        except ImportError as e:
            raise ImportError(
                "Tau2Task.from_domain requires tau2-bench. "
                "Install: git clone https://github.com/sierra-research/tau2-bench && "
                "cd tau2-bench && uv sync"
            ) from e

        tasks = get_tasks(
            domain,
            task_ids=[task_id] if task_id else None,
            task_split_name=task_split,
        )

        if task_id:
            tau2_task = tasks[0]
            desc = f"tau2-bench {domain} task {tau2_task.id}"
            if hasattr(tau2_task, "user_scenario") and tau2_task.user_scenario:
                instructions = getattr(tau2_task.user_scenario, "instructions", "")
                if instructions:
                    desc += f": {instructions[:200]}"
            return cls(
                description=desc,
                domain=domain,
                tau2_task_id=tau2_task.id,
                task_split=task_split,
                _tau2_task=tau2_task,
                **kwargs,
            )

        return [
            cls(
                description=f"tau2-bench {domain} task {t.id}",
                domain=domain,
                tau2_task_id=t.id,
                task_split=task_split,
                _tau2_task=t,
                **kwargs,
            )
            for t in tasks
        ]

    @classmethod
    def list_tasks(
        cls,
        domain: str,
        task_split: str | None = None,
    ) -> list[str]:
        """List available task IDs for a domain."""
        try:
            from tau2.runner.helpers import get_tasks
        except ImportError as e:
            raise ImportError("Requires tau2-bench. See README for installation.") from e
        tasks = get_tasks(domain, task_split_name=task_split)
        return [t.id for t in tasks]


class Tau2Evaluator:
    """Evaluator that runs the full tau2 simulation pipeline.

    Unlike simpler evaluators that check state.messages for an answer,
    this evaluator drives the entire tau2 orchestration: user simulator,
    tool execution, and multi-faceted evaluation (DB checks, action checks,
    communication checks).

    reward=1.0 means all evaluation criteria passed.
    """

    async def evaluate(self, task: BaseTask, state: "State") -> EvalResult:  # noqa: ARG002
        assert isinstance(task, Tau2Task)

        try:
            from tau2.registry import registry
            from tau2.runner.batch import run_single_task
            from tau2.runner.helpers import get_tasks
            from tau2.data_model.simulation import TextRunConfig
            from tau2.evaluator.evaluator import EvaluationType
            from .agent import create_harnessx_agent

            # Register our agent factory (idempotent)
            _register_agent_once(registry, create_harnessx_agent)

            # Load the tau2 task if not cached
            tau2_task = task._tau2_task
            if tau2_task is None:
                tasks = get_tasks(
                    task.domain,
                    task_ids=[task.tau2_task_id],
                    task_split_name=task.task_split,
                )
                if not tasks:
                    return EvalResult(
                        passed=False,
                        score=0.0,
                        reason=f"tau2 task {task.tau2_task_id!r} not found in domain {task.domain!r}",
                        reward=0.0,
                    )
                tau2_task = tasks[0]

            # Build tau2 run config
            llm_args_agent: dict[str, Any] = {}
            if task.agent_api_base:
                llm_args_agent["api_base"] = task.agent_api_base
            if task.agent_api_key and task.agent_api_key != "EMPTY":
                llm_args_agent["api_key"] = task.agent_api_key

            config = TextRunConfig(
                domain=task.domain,
                agent="harnessx",
                user="user_simulator",
                llm_agent=task.agent_model,
                llm_args_agent=llm_args_agent,
                llm_user=task.user_llm,
                llm_args_user=task.user_llm_args,
                max_steps=task.tau2_max_steps,
                max_errors=task.tau2_max_errors,
            )

            # Run simulation
            result = run_single_task(
                config,
                tau2_task,
                seed=task.tau2_seed,
                evaluation_type=EvaluationType.ALL,
            )

            # Map reward_info to EvalResult
            if result.reward_info is None:
                return EvalResult(
                    passed=False,
                    score=0.0,
                    reason="tau2 simulation produced no reward_info",
                    reward=0.0,
                )

            reward = result.reward_info.reward
            reason = _build_reason(result)

            return EvalResult(
                passed=reward >= 1.0,
                score=reward,
                reason=reason,
                reward=reward,
            )

        except ImportError:
            logger.warning("tau2 not installed; returning reward=0.0")
            return EvalResult(
                passed=False,
                score=0.0,
                reason="tau2 not installed. See benchmarks/tau2/README.md for installation.",
                reward=0.0,
            )
        except Exception as e:
            logger.warning("Tau2Evaluator error: %s", e, exc_info=True)
            return EvalResult(
                passed=False,
                score=0.0,
                reason=f"tau2 simulation error: {e}",
                reward=0.0,
            )


# ─── Helpers ─────────────────────────────────────────────────────────────────

_AGENT_REGISTERED = False


def _register_agent_once(registry: Any, factory: Any) -> None:
    """Register the harnessx agent factory in tau2's registry, once."""
    global _AGENT_REGISTERED
    if _AGENT_REGISTERED:
        return
    try:
        registry.register_agent_factory(factory, "harnessx")
        _AGENT_REGISTERED = True
    except (ValueError, KeyError):
        # Already registered
        _AGENT_REGISTERED = True


def _build_reason(result: Any) -> str:
    """Build a human-readable reason string from tau2's SimulationRun."""
    reward = result.reward_info.reward
    parts = [f"reward={reward:.3f}"]

    ri = result.reward_info

    if ri.db_check is not None:
        parts.append(f"db={'pass' if ri.db_check.db_match else 'fail'}")

    if ri.action_checks:
        passed = sum(1 for c in ri.action_checks if c.action_match)
        parts.append(f"actions={passed}/{len(ri.action_checks)}")

    if ri.communicate_checks:
        passed = sum(1 for c in ri.communicate_checks if c.met)
        parts.append(f"comm={passed}/{len(ri.communicate_checks)}")

    if hasattr(result, "termination_reason") and result.termination_reason:
        parts.append(f"term={result.termination_reason}")

    return ", ".join(parts)
