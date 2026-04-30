"""SWEBenchTask — wraps a SWE-bench instance as an HarnessX task."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from harnessx.core.harness import BaseTask
from harnessx.core.events import EvalResult

if TYPE_CHECKING:
    from harnessx.core.state import State

logger = logging.getLogger(__name__)


@dataclass
class SWEBenchTask(BaseTask):
    """
    Wraps a SWE-bench instance as an HarnessX task.

    Requires: docker daemon running + pip install swebench

    The agent receives the issue description as task.description.
    Use DockerWorkspace to isolate file system changes per instance.

    Evaluator: SWEBenchEvaluator runs swebench's official harness to grade the patch.
    """

    instance_id: str = ""
    repo: str = ""
    base_commit: str = ""
    _instance_data: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("SWEBenchTask requires an instance_id")
        if not self.description:
            self._load_instance()

    def _load_instance(self) -> None:
        """Load instance problem statement from swebench dataset."""
        try:
            from swebench.harness.utils import load_swebench_dataset  # type: ignore

            dataset = load_swebench_dataset("princeton-nlp/SWE-bench_Lite", split="test")
            for item in dataset:
                if item["instance_id"] == self.instance_id:
                    self._instance_data = dict(item)
                    self.description = item.get("problem_statement", f"SWE-bench: {self.instance_id}")
                    self.repo = self.repo or item.get("repo", "")
                    self.base_commit = self.base_commit or item.get("base_commit", "")
                    return
            logger.warning("SWE-bench instance %r not found in dataset", self.instance_id)
            self.description = f"Fix GitHub issue in {self.repo}: {self.instance_id}"
        except ImportError:
            logger.warning("swebench not installed; using stub description. Install with: pip install swebench")
            self.description = f"SWE-bench instance: {self.instance_id}"
        except Exception as e:
            logger.warning("Failed to load SWE-bench instance %r: %s", self.instance_id, e)
            self.description = f"SWE-bench instance: {self.instance_id}"


class SWEBenchEvaluator:
    """
    Evaluator using SWE-bench's official grading harness.

    Extracts the agent's git diff from state and runs the SWE-bench grader.
    reward=1.0 if all test cases pass, else 0.0.

    Note: requires Docker and the SWE-bench test harness to be set up.
    """

    def __init__(self, task: SWEBenchTask) -> None:
        self._task = task

    async def evaluate(self, task: BaseTask, state: "State") -> EvalResult:
        assert isinstance(task, SWEBenchTask)
        try:
            from swebench.harness.run_evaluation import run_instance  # type: ignore  # noqa: F401

            patch = self._extract_patch(state)
            if not patch:
                return EvalResult(
                    passed=False,
                    score=0.0,
                    reason="no patch found in agent output",
                    reward=0.0,
                )

            result = await _run_swebench_instance(
                instance_id=task.instance_id,
                patch=patch,
                instance_data=task._instance_data,
            )
            passed = result.get("resolved", False)
            r = 1.0 if passed else 0.0
            return EvalResult(
                passed=passed,
                score=r,
                reason=f"swebench: {'resolved' if passed else 'not resolved'}",
                reward=r,
            )
        except ImportError:
            logger.warning("swebench not installed; returning reward=0.0")
            return EvalResult(passed=False, score=0.0, reason="swebench not installed", reward=0.0)
        except Exception as e:
            logger.warning("SWEBenchEvaluator error: %s", e)
            return EvalResult(passed=False, score=0.0, reason=str(e), reward=0.0)

    def _extract_patch(self, state: "State") -> str:
        """Extract git diff from state (tool results or last assistant message)."""
        for msg in reversed(state.messages):
            if msg.role == "tool" and msg.content:
                content = msg.content
                if content.startswith("diff --git") or content.startswith("---"):
                    return content
        for msg in reversed(state.messages):
            if msg.role == "assistant" and msg.content:
                content = msg.content
                if "diff --git" in content:
                    start = content.index("diff --git")
                    return content[start:]
        return ""


async def _run_swebench_instance(instance_id: str, patch: str, instance_data: dict) -> dict:
    """Run SWE-bench grader in a subprocess to avoid blocking the event loop."""
    import asyncio
    import sys
    import json
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch)
        patch_file = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--instance_id",
            instance_id,
            "--patch_file",
            patch_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        output = stdout.decode()
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            resolved = "resolved: True" in output or '"resolved": true' in output
            return {"resolved": resolved, "raw_output": output[:500]}
    finally:
        os.unlink(patch_file)
