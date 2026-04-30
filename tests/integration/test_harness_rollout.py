# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from harnessx.core.trajectory import (
    StatefulTrajectory,
    TrajectoryStep,
    FullStateSnapshot,
    StateDelta,
    TokenAnnotation,
)
from harnessx.core.events import EvalResult, ToolResultEvent
from harnessx.providers.sglang import (
    SGLangProvider,
    StepCapture,
    build_flat_sequence,
)


# ---------------------------------------------------------------------------
# Slime stubs — injected before importing harness_rollout
# ---------------------------------------------------------------------------


def _install_slime_stubs():
    """Install minimal Slime stubs into sys.modules so harness_rollout can import."""
    # slime package root
    slime_pkg = types.ModuleType("slime")
    sys.modules.setdefault("slime", slime_pkg)

    # slime.utils.types.Sample
    @dataclass
    class Sample:
        prompt: Any = ""
        label: str = ""
        metadata: dict = field(default_factory=dict)
        response: str = ""
        tokens: list = field(default_factory=list)
        loss_mask: list = field(default_factory=list)
        rollout_log_probs: list = field(default_factory=list)
        response_length: int = 0

        class Status:
            ABORTED = "ABORTED"
            TRUNCATED = "TRUNCATED"
            COMPLETED = "COMPLETED"

        status: Any = None

    utils_mod = types.ModuleType("slime.utils")
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.Sample = Sample
    http_utils_mod = types.ModuleType("slime.utils.http_utils")
    http_utils_mod.post = AsyncMock(return_value=None)
    sys.modules.setdefault("slime.utils", utils_mod)
    sys.modules.setdefault("slime.utils.types", types_mod)
    sys.modules.setdefault("slime.utils.http_utils", http_utils_mod)
    slime_pkg.utils = utils_mod

    # slime.rollout.sglang_rollout.GenerateState
    class GenerateState:
        def __init__(self, args):
            self.tokenizer = args.tokenizer if hasattr(args, "tokenizer") else MagicMock()

    rollout_mod = types.ModuleType("slime.rollout")
    sglang_mod = types.ModuleType("slime.rollout.sglang_rollout")
    sglang_mod.GenerateState = GenerateState
    sys.modules.setdefault("slime.rollout", rollout_mod)
    sys.modules.setdefault("slime.rollout.sglang_rollout", sglang_mod)
    rollout_mod.sglang_rollout = sglang_mod

    # slime.rollout.rm_hub.math_dapo_utils.compute_score
    rm_hub = types.ModuleType("slime.rollout.rm_hub")
    math_dapo = types.ModuleType("slime.rollout.rm_hub.math_dapo_utils")
    math_dapo.compute_score = MagicMock(return_value={"score": 1.0, "acc": True, "pred": "42"})
    sys.modules.setdefault("slime.rollout.rm_hub", rm_hub)
    sys.modules.setdefault("slime.rollout.rm_hub.math_dapo_utils", math_dapo)
    rm_hub.math_dapo_utils = math_dapo

    return Sample


_SAMPLE_CLS = _install_slime_stubs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snap(step_id: int) -> FullStateSnapshot:
    return FullStateSnapshot(
        step_id=step_id,
        messages=(),
        slots={},
        cumulative_tokens=0,
        cumulative_cost_usd=0.0,
    )


def _make_delta(step_id: int) -> StateDelta:
    return StateDelta(step_id=step_id, operations=())


def _make_step(
    step_id: int,
    reward: float = -1.0,
    tool_calls: bool = False,
    annotation: TokenAnnotation | None = None,
) -> TrajectoryStep:
    obs = []
    if tool_calls:
        obs.append(
            ToolResultEvent(
                run_id="r",
                step_id=step_id,
                tool_name="code_interpreter",
                tool_call_id="tc1",
                result="output",
                error=None,
            )
        )
    return TrajectoryStep(
        step_id=step_id,
        state_snapshot=_make_snap(step_id),
        state_delta=_make_delta(step_id),
        action=None,
        observation=obs,
        event=None,
        reward=reward,
        token_annotation=annotation,
    )


def _make_traj_with_annotations(
    n_steps: int = 2,
    terminal: float = 1.0,
    tool_calls: bool = True,
) -> StatefulTrajectory:
    """Build a trajectory with TokenAnnotation on every step."""
    traj = StatefulTrajectory(run_id="test")
    for i in range(n_steps):
        ann = TokenAnnotation(
            prompt_ids=list(range(10)),
            response_ids=[100 + i * 10 + j for j in range(5)],
            response_mask=[1, 1, 1, 1, 1],
            response_logprobs=[-0.1 * (j + 1) for j in range(5)],
        )
        traj.add_step(_make_step(i, reward=terminal, tool_calls=tool_calls, annotation=ann))
    return traj


def _make_provider_with_captures(n_steps: int = 2) -> SGLangProvider:
    """Build a mock SGLangProvider with synthetic StepCaptures."""
    provider = MagicMock(spec=SGLangProvider)
    provider.truncated = False
    provider.aborted = False

    # Build monotonically growing input_ids to satisfy flat-sequence invariant
    base = list(range(10))
    caps = []
    for i in range(n_steps):
        output_ids = [100 + i * 10 + j for j in range(5)]
        _tool_ids = [200 + i * 10 + j for j in range(3)]
        if i == 0:
            input_ids = base
        else:
            prev = caps[i - 1]
            input_ids = prev.input_ids + prev.output_token_ids + [200 + (i - 1) * 10 + j for j in range(3)]
        caps.append(
            StepCapture(
                input_ids=input_ids,
                output_token_ids=output_ids,
                output_logprobs=[-0.1] * 5,
                finish_reason="stop",
            )
        )
    provider.step_captures = caps
    return provider


def _make_eval_result(reward: float = 1.0, pred: str = "42") -> EvalResult:
    return EvalResult(passed=reward >= 0, score=reward, reason=pred, reward=reward)


# ---------------------------------------------------------------------------
# Tests — reward_func()
# ---------------------------------------------------------------------------


class TestRewardFunc:
    """Test reward_func() computation and metadata cleanup."""

    def _make_sample_with_traj(
        self,
        terminal: float = 1.0,
        n_steps: int = 2,
        tool_calls: bool = True,
        exit_reason: str = "done",
    ):
        from recipe.slime.harness_rollout import reward_func  # noqa: F401 (imported below)

        sample = _SAMPLE_CLS()
        sample.response = r"\boxed{42}" if terminal < 0 else r"\boxed{42}"
        sample.response_length = 20
        traj = _make_traj_with_annotations(n_steps, terminal, tool_calls)
        sample.metadata = {
            "task_type": "math",
            "trajectory": traj,
            "eval_result": _make_eval_result(reward=terminal),
            "exit_reason": exit_reason,
            "total_steps": n_steps,
            "total_tokens": 100,
        }
        return sample

    @pytest.mark.asyncio
    async def test_reward_func_returns_required_keys(self):
        from recipe.slime.harness_rollout import reward_func

        sample = self._make_sample_with_traj(terminal=1.0)
        result = await reward_func(None, sample)

        required = {
            "score",
            "terminal",
            "prm_adjusted",
            "tool_success_count",
            "tool_error_count",
            "tool_use_rate",
            "total_steps",
            "total_tokens",
            "response_length",
            "exit_reason",
            "trajectory_quality",
        }
        assert required <= result.keys(), f"Missing keys: {required - result.keys()}"

    @pytest.mark.asyncio
    async def test_reward_func_correct_answer(self):
        """terminal=1.0 → score >= 0, trajectory_quality='complete'."""
        from recipe.slime.harness_rollout import reward_func

        sample = self._make_sample_with_traj(terminal=1.0, exit_reason="done")
        result = await reward_func(None, sample)
        assert result["score"] >= 0.0
        assert result["terminal"] == pytest.approx(1.0)
        assert result["trajectory_quality"] == "complete"

    @pytest.mark.asyncio
    async def test_reward_func_wrong_answer_with_tool_turns(self):
        """terminal=-1.0 with tool turns → RetoolCompatPRM applies shaping."""
        from recipe.slime.harness_rollout import reward_func

        sample = self._make_sample_with_traj(terminal=-1.0, n_steps=4, tool_calls=True)
        result = await reward_func(None, sample)
        assert result["terminal"] == pytest.approx(-1.0)
        # RetoolCompatPRM: score < 0 but bounded by -0.6 cap
        assert result["score"] <= 0.0

    @pytest.mark.asyncio
    async def test_reward_func_pops_trajectory_from_metadata(self):
        """Trajectory must be removed from metadata after reward_func runs."""
        from recipe.slime.harness_rollout import reward_func

        sample = self._make_sample_with_traj(terminal=1.0)
        assert "trajectory" in sample.metadata
        assert "eval_result" in sample.metadata
        await reward_func(None, sample)
        assert "trajectory" not in sample.metadata
        assert "eval_result" not in sample.metadata

    @pytest.mark.asyncio
    async def test_reward_func_truncated_trajectory_quality(self):
        from recipe.slime.harness_rollout import reward_func

        sample = self._make_sample_with_traj(terminal=-1.0, exit_reason="budget_exceeded")
        result = await reward_func(None, sample)
        assert result["trajectory_quality"] == "truncated"

    @pytest.mark.asyncio
    async def test_reward_func_aborted_trajectory_quality(self):
        from recipe.slime.harness_rollout import reward_func

        sample = self._make_sample_with_traj(terminal=-1.0, exit_reason="interrupted")
        result = await reward_func(None, sample)
        assert result["trajectory_quality"] == "aborted"

    @pytest.mark.asyncio
    async def test_reward_func_tool_counts(self):
        """tool_success_count incremented for each successful observation."""
        from recipe.slime.harness_rollout import reward_func

        sample = self._make_sample_with_traj(terminal=1.0, n_steps=3, tool_calls=True)
        result = await reward_func(None, sample)
        assert result["tool_success_count"] == 3  # 1 per step, 3 steps
        assert result["tool_error_count"] == 0

    @pytest.mark.asyncio
    async def test_reward_func_math_extra_fields_present(self):
        """math_shaped task: extra_reward_fn returns format_score, pred, has_boxed_answer."""
        from recipe.slime.harness_rollout import reward_func

        sample = self._make_sample_with_traj(terminal=-1.0)
        sample.metadata["task_type"] = "math_shaped"  # math_shaped has extra_reward_fn
        sample.response = r"\boxed{42}"
        result = await reward_func(None, sample)
        assert "format_score" in result
        assert "pred" in result
        assert "has_boxed_answer" in result

    @pytest.mark.asyncio
    async def test_reward_func_missing_trajectory(self):
        """No trajectory in metadata → terminal defaults to -1.0."""
        from recipe.slime.harness_rollout import reward_func

        sample = _SAMPLE_CLS()
        sample.metadata = {"task_type": "math"}
        result = await reward_func(None, sample)
        assert result["terminal"] == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_reward_func_type_error_on_non_sample(self):
        from recipe.slime.harness_rollout import reward_func

        with pytest.raises(TypeError):
            await reward_func(None, {"not": "a sample"})


# ---------------------------------------------------------------------------
# Tests — _fill_sample_from_captures()
# ---------------------------------------------------------------------------


class TestFillSampleFromCaptures:
    """Test _fill_sample_from_captures() directly — token alignment invariants."""

    def _run_fill(self, traj, provider, tokenizer=None):
        from recipe.slime.harness_rollout import _fill_sample_from_captures

        sample = _SAMPLE_CLS()
        if tokenizer is None:
            tokenizer = MagicMock()
            tokenizer.decode = lambda ids, **kw: "decoded"
        result = MagicMock()
        result.task_end.eval_result = _make_eval_result()
        result.task_end.exit_reason = "done"
        result.task_end.total_steps = 2
        result.task_end.total_tokens = 100
        _fill_sample_from_captures(sample, provider, tokenizer, traj=traj, result=result)
        return sample

    def test_primary_path_uses_traj_annotations(self):
        """When traj has annotations, sample is filled from traj.to_rl_records()."""
        traj = _make_traj_with_annotations(n_steps=2)
        provider = MagicMock()
        provider.step_captures = []

        # Patch traj.to_rl_records to return synthetic episode
        traj.to_rl_records = MagicMock(
            return_value={
                "tokens": list(range(20)),
                "loss_mask": [1] * 10 + [0] * 10,
                "rollout_log_probs": [-0.1] * 10 + [0.0] * 10,
                "response_length": 10,
                "response": "answer",
            }
        )

        sample = self._run_fill(traj, provider)
        assert sample.tokens == list(range(20))
        assert sample.response_length == 10
        traj.to_rl_records.assert_called_once()

    def test_fallback_path_when_no_annotations(self):
        """Without token annotations, build_flat_sequence() fallback is used."""
        traj = _make_traj_with_annotations(n_steps=2)
        # Strip annotations
        for step in traj.steps:
            step.token_annotation = None

        provider = _make_provider_with_captures(n_steps=2)

        tokenizer = MagicMock()
        tokenizer.decode = lambda ids, **kw: " ".join(map(str, ids))

        sample = self._run_fill(traj, provider, tokenizer=tokenizer)

        # Fallback invariant: len(loss_mask) == len(rollout_log_probs) == response_length
        assert len(sample.loss_mask) == len(sample.rollout_log_probs) == sample.response_length

    def test_loss_mask_length_invariant_primary_path(self):
        """len(loss_mask) == len(rollout_log_probs) == response_length (primary path)."""
        traj = _make_traj_with_annotations(n_steps=2)
        resp_len = 15
        traj.to_rl_records = MagicMock(
            return_value={
                "tokens": list(range(25)),
                "loss_mask": [1] * resp_len,
                "rollout_log_probs": [-0.1] * resp_len,
                "response_length": resp_len,
                "response": "x",
            }
        )
        provider = MagicMock()
        provider.step_captures = []

        sample = self._run_fill(traj, provider)
        assert len(sample.loss_mask) == len(sample.rollout_log_probs) == sample.response_length == resp_len

    def test_metadata_saved_for_reward_func(self):
        """trajectory and eval_result saved in metadata for reward_func()."""
        traj = _make_traj_with_annotations(n_steps=2)
        traj.to_rl_records = MagicMock(
            return_value={
                "tokens": [],
                "loss_mask": [],
                "rollout_log_probs": [],
                "response_length": 0,
                "response": "",
            }
        )
        provider = MagicMock()
        provider.step_captures = []

        sample = self._run_fill(traj, provider)
        assert "trajectory" in sample.metadata
        assert "eval_result" in sample.metadata
        assert sample.metadata["exit_reason"] == "done"


# ---------------------------------------------------------------------------
# Tests — build_flat_sequence() flat-sequence invariant
# ---------------------------------------------------------------------------


class TestFlatSequenceInvariant:
    """Verify build_flat_sequence() satisfies the structural invariant:

    caps[t+1].input_ids == caps[t].input_ids + caps[t].output_token_ids + tool_ids[t]
    """

    def _build_provider(self, n_steps: int, base_prompt_len: int = 10) -> SGLangProvider:
        """Build a real SGLangProvider with synthetic captures satisfying the invariant."""
        provider = MagicMock(spec=SGLangProvider)
        provider.truncated = False
        provider.aborted = False

        base = list(range(base_prompt_len))
        caps = []
        for i in range(n_steps):
            output_ids = [1000 + i * 100 + j for j in range(8)]
            _tool_ids = [2000 + i * 100 + j for j in range(4)] if i < n_steps - 1 else []

            if i == 0:
                input_ids = base[:]
            else:
                prev = caps[i - 1]
                # Invariant: next input = prev input + prev output + tool tokens
                prev_tool_ids = [2000 + (i - 1) * 100 + j for j in range(4)]
                input_ids = prev.input_ids + prev.output_token_ids + prev_tool_ids

            caps.append(
                StepCapture(
                    input_ids=input_ids,
                    output_token_ids=output_ids,
                    output_logprobs=[-0.05] * len(output_ids),
                    finish_reason="stop",
                )
            )
        provider.step_captures = caps
        return provider

    def test_single_turn(self):
        provider = self._build_provider(n_steps=1)
        flat = build_flat_sequence(provider)
        assert flat.prompt_ids == provider.step_captures[0].input_ids
        assert flat.response_ids == provider.step_captures[0].output_token_ids
        assert flat.loss_mask == [1] * len(flat.response_ids)

    def test_multi_turn_response_ids_include_tool_tokens(self):
        """Multi-turn: response_ids = model tokens (turn 0) + tool tokens + model tokens (turn 1)."""
        provider = self._build_provider(n_steps=2)
        flat = build_flat_sequence(provider)

        cap0 = provider.step_captures[0]
        cap1 = provider.step_captures[1]

        # Tool tokens are the gap between cap1.input_ids and (cap0.input_ids + cap0.output_token_ids)
        expected_tool_ids = cap1.input_ids[len(cap0.input_ids) + len(cap0.output_token_ids) :]
        expected_response = cap0.output_token_ids + expected_tool_ids + cap1.output_token_ids

        assert flat.response_ids == expected_response

    def test_loss_mask_model_tokens_only(self):
        """loss_mask=1 for model-generated tokens, 0 for tool result tokens."""
        provider = self._build_provider(n_steps=2)
        flat = build_flat_sequence(provider)

        cap0 = provider.step_captures[0]
        cap1 = provider.step_captures[1]
        tool_len = len(cap1.input_ids) - len(cap0.input_ids) - len(cap0.output_token_ids)
        model_len_0 = len(cap0.output_token_ids)
        model_len_1 = len(cap1.output_token_ids)

        expected_mask = [1] * model_len_0 + [0] * tool_len + [1] * model_len_1
        assert flat.loss_mask == expected_mask

    def test_rollout_logprobs_length_matches_response(self):
        provider = self._build_provider(n_steps=2)
        flat = build_flat_sequence(provider)
        assert len(flat.rollout_logprobs) == len(flat.response_ids) == len(flat.loss_mask)

    def test_three_turns(self):
        """Invariant holds across 3+ turns."""
        provider = self._build_provider(n_steps=3)
        flat = build_flat_sequence(provider)
        assert len(flat.loss_mask) == len(flat.response_ids) == len(flat.rollout_logprobs)
        assert flat.prompt_ids == provider.step_captures[0].input_ids
