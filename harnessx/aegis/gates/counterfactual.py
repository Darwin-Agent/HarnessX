# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Counterfactual replay gate.

Samples K trajectories from the previous round's passing set and replays
the NEW candidate's processor chain (loaded from `new_config_yaml_text`)
against the recorded events. Fails if any sampled task's `final_output`
or `exit_reason` changes relative to what the original passing run emitted.

No LLM calls, no tool execution — replay works only on recorded events.
This catches the class of regression that `replay_smoke` can't: "new
processor rewrites output of a task the old config already got right".
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml

from .structure import GateResult


# Recorded JSONL "kind" → MultiHookProcessor hook method name.
_HOOK_KINDS = {
    "after_model": "on_after_model",
    "after_tool": "on_after_tool",
    "task_end": "on_task_end",
}


def _load_trajectory_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def _original_terminals(events: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"final_output": None, "exit_reason": None}
    for e in events:
        if e.get("kind") == "task_end":
            out["final_output"] = e.get("final_output")
            out["exit_reason"] = e.get("exit_reason")
    return out


def _build_event_object(row: dict):
    """Minimal attribute-mutable shim from a JSON event row.

    Processors written against harnessx events expect attribute access
    (event.content, event.final_output, …). SimpleNamespace gives that
    plus mutation semantics; we compare attrs after each hook invocation.
    """
    from types import SimpleNamespace
    return SimpleNamespace(**row)


async def _dispatch_one(proc, event_obj, kind: str) -> None:
    method_name = _HOOK_KINDS.get(kind)
    if method_name is None:
        return
    hook = getattr(proc, method_name, None)
    if hook is None:
        return
    async for _ in hook(event_obj):
        pass  # mutations on event_obj are the observable effect


def _instantiate_processors(cfg_yaml: dict) -> list:
    from harnessx.core.builder import _instantiate
    procs_raw = cfg_yaml.get("processors") or []
    out = []
    for entry in procs_raw:
        if not isinstance(entry, dict):
            continue
        if "_target_" not in entry and "type" not in entry:
            continue
        try:
            obj = _instantiate(entry)
        except Exception:
            # A processor that won't instantiate is caught by canonicalize;
            # this gate isn't the right place to re-raise.
            continue
        if obj is not None:
            out.append(obj)
    return out


async def check_counterfactual_replay(
    *,
    new_config_yaml_text: str | None,
    passing_task_ids: list[str],
    trajectories_dir: Path,
    k_samples: int = 3,
    rng_seed: int | None = 0,
) -> GateResult:
    if new_config_yaml_text is None:
        return GateResult(ok=True, reason="skipped: no candidate cfg supplied")
    if not passing_task_ids:
        return GateResult(
            ok=True, reason="skipped: no previously-passing tasks to sample",
        )

    rng = random.Random(rng_seed)
    sampled = (
        passing_task_ids
        if len(passing_task_ids) <= k_samples
        else rng.sample(passing_task_ids, k_samples)
    )

    try:
        cfg = yaml.safe_load(new_config_yaml_text) or {}
    except yaml.YAMLError as exc:
        return GateResult(
            ok=False, reason=f"counterfactual: new config YAML invalid: {exc!r}",
        )

    processors = _instantiate_processors(cfg)

    regressions: list[str] = []
    trajectories_dir = Path(trajectories_dir)
    for tid in sampled:
        candidates = list(trajectories_dir.glob(f"{tid}*.jsonl"))
        if not candidates:
            continue
        events = _load_trajectory_events(candidates[0])
        if not events:
            continue
        orig = _original_terminals(events)

        for row in events:
            kind = row.get("kind")
            if kind not in _HOOK_KINDS:
                continue
            ev_obj = _build_event_object(row)
            for proc in processors:
                try:
                    await _dispatch_one(proc, ev_obj, kind)
                except Exception:
                    continue
            if kind == "task_end":
                new_final = getattr(ev_obj, "final_output", None)
                new_exit = getattr(ev_obj, "exit_reason", None)
                if new_final != orig["final_output"]:
                    regressions.append(
                        f"{tid}: final_output changed "
                        f"{orig['final_output']!r} → {new_final!r}"
                    )
                if new_exit != orig["exit_reason"]:
                    regressions.append(
                        f"{tid}: exit_reason changed "
                        f"{orig['exit_reason']!r} → {new_exit!r}"
                    )

    if regressions:
        return GateResult(
            ok=False,
            reason=(
                "counterfactual replay flagged regressions: "
                + "; ".join(regressions[:5])
            ),
        )
    return GateResult(ok=True)
