# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""AEGIS — Evidence-Graph Harness Evolution.

3-role adversarial MAS (Planner + Evolver × N + Critic) replacing
``harnessx.meta_harness``. See
``idea_talk/ideas/2026-05-08-aegis-v1-design.md`` for the design.

Public API:
- :class:`AegisAgent` — drop-in replacement for ``MetaAgent``
- :func:`compute_changeset` — re-export from meta_harness for recipe compat
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harnessx.meta_harness.agent import compute_changeset

from .orchestrator import AegisOrchestrator


@dataclass
class AegisAgent:
    """Drop-in replacement for ``MetaAgent``.

    :meth:`evolve` returns the shipped candidate's **applied HarnessConfig
    YAML** path (e.g. ``R1/applied/C-R1-01/config.yaml``) when a ship
    occurred; else returns the original config path (no-op). The applied
    YAML is
    written by the Evolver during Stage 2 and validated (load +
    canonicalize) before Stage 4 gates it. Recipe callers load it via
    ``HarnessConfig.from_yaml_file(path)``.
    """

    num_evolvers: int = 4
    budget_per_round_usd: float = 20.0
    max_ask_more: int = 2
    max_concurrency: int = 4
    model_config: object | None = None
    # Benchmark model config used by the Stage 4 replay gate. This is the
    # model the Harness runs tasks against — NOT the meta-model above. If
    # None, the replay gate skips (permanently — every round). Recipe callers
    # should pass this explicitly (e.g. the GAIA ``model_config``).
    replay_model: object | None = None
    # Forwarded to AegisOrchestrator; Stage 5 auto-revert runs ACROSS rounds in
    # the pilot driver, not inside ``run_round``, so this is currently unused
    # but wired for forward-compat.
    auto_revert_enabled: bool = True

    async def evolve(
        self,
        current_config,
        trajectories_dir,
        output_dir,
        *,
        pass_flags_by_task: dict | None = None,
        round_n: int = 1,
        raw_sessions_dir=None,
        **kwargs,
    ) -> Path:
        """Run one round of AEGIS evolution.

        ``pass_flags_by_task`` maps task_id (str) to a list of booleans where
        each bool marks whether one run of that task passed. Expected shape:
        ``dict[str, list[bool]]``. As a caller convenience, a scalar bool
        ``{"tid": True}`` is coerced to ``{"tid": [True]}``. Any other value
        type raises ``TypeError`` so the footgun surfaces early instead of
        silently producing degenerate per-task patterns.
        """
        # Validate pass_flags_by_task shape: dict[str, list[bool]].
        # Coerce scalar bool to [bool] for caller convenience.
        pass_flags_by_task = pass_flags_by_task or {}
        coerced: dict[str, list[bool]] = {}
        for tid, flags in pass_flags_by_task.items():
            if isinstance(flags, bool):
                coerced[tid] = [flags]
            elif isinstance(flags, (list, tuple)):
                coerced[tid] = [bool(x) for x in flags]
            else:
                raise TypeError(
                    f"pass_flags_by_task[{tid!r}] must be bool or list[bool]; "
                    f"got {type(flags).__name__}"
                )
        pass_flags_by_task = coerced

        trajectories_dir = Path(trajectories_dir)
        # output_dir is legacy — callers historically pointed it at either
        # <run_dir>/R<n>/evolve or <run_dir>/R<n>. We only need run_dir
        # (orchestrator derives all per-round paths from that). Walk up the
        # tree if the caller pointed us at a subdirectory.
        output_dir = Path(output_dir)
        run_dir = output_dir
        if run_dir.name in ("evolve", "evolver"):
            run_dir = run_dir.parent
        if run_dir.name.startswith("R") and run_dir.name[1:].isdigit():
            run_dir = run_dir.parent

        if isinstance(current_config, Path):
            current_config_path = current_config
        else:
            # Materialise a temp YAML so the orchestrator has a path.
            current_config_path = run_dir / "_pending_input_config.yaml"
            current_config_path.write_text(current_config.to_yaml())

        orch = AegisOrchestrator(
            run_dir=run_dir,
            num_evolvers=self.num_evolvers,
            model_config=self.model_config,
            max_ask_more=self.max_ask_more,
            max_concurrency=self.max_concurrency,
            budget_per_round_usd=self.budget_per_round_usd,
            replay_model=self.replay_model,
            auto_revert_enabled=self.auto_revert_enabled,
        )

        run_round_result = await orch.run_round(
            round_n=round_n,
            raw_sessions_dir=raw_sessions_dir or trajectories_dir,
            pass_flags_by_task=pass_flags_by_task,
            current_config_path=current_config_path,
        )

        shipped_cids = run_round_result.get("shipped_cids") or (
            [run_round_result["shipped_cid"]] if run_round_result.get("shipped_cid") else []
        )
        round_dir = run_dir / f"R{round_n}"
        merged = run_round_result.get("merged_applied_path")
        if merged and Path(merged).exists():
            return Path(merged)
        if shipped_cids:
            # Fallback: no merged file (e.g. compose errored) — use first cid's
            # applied yaml so we at least ship one change.
            applied_yaml = round_dir / "applied" / shipped_cids[0] / "config.yaml"
            if applied_yaml.exists():
                return applied_yaml
        return current_config_path


__all__ = ["AegisAgent", "compute_changeset"]
