# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Compose multiple shipped candidates' applied configs onto a base.

Each candidate's applied YAML is a FULL HarnessConfig derived from the
round's base config + the candidate's bucket-specific change. When
multiple candidates ship in the same round, we can't just take "last
wins" — each candidate's unchanged fields would silently overwrite
earlier candidates' changes.

This helper diffs each candidate's bucket-relevant fields against the
frozen parent and applies only those changes to the running base:

  prompt     — take candidate's `template_path` for the SystemPromptProcessor
  tools      — merge candidate's `tool_registry.custom` vs parent: drop
               entries parent had but candidate removed, append entries
               candidate added
  config     — replace matching processor kwargs (matched by `_target_`)
  processor  — apply candidate's processor diff vs parent: drop processors
               parent had but candidate removed, append processors candidate
               added (matched by `_target_`)

Multi-ship is only valid when each shipped candidate's bucket is
DISJOINT from the others. Stage 4 enforces that constraint before
calling this function.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import yaml


def _is_system_prompt(proc: dict) -> bool:
    tgt = str(proc.get("_target_", ""))
    return tgt.endswith(".SystemPromptProcessor") or "SystemPromptProcessor" in tgt


def _apply_prompt(base: dict, candidate: dict, parent: dict) -> None:
    """Copy the candidate's SystemPromptProcessor template_path into base.

    ``parent`` is unused here — prompt bucket only swaps a single template_path.
    """
    del parent  # signature uniform across appliers
    cand_tp = None
    for p in candidate.get("processors", []) or []:
        if isinstance(p, dict) and _is_system_prompt(p):
            sb = p.get("system_builder") or {}
            if isinstance(sb, dict):
                cand_tp = sb.get("template_path")
                if cand_tp:
                    break
    if not cand_tp:
        return
    for bp in base.get("processors", []) or []:
        if isinstance(bp, dict) and _is_system_prompt(bp):
            sb = bp.setdefault("system_builder", {})
            if isinstance(sb, dict):
                sb["template_path"] = cand_tp


def _apply_tools(base: dict, candidate: dict, parent: dict) -> None:
    """Apply candidate's tool_registry.custom diff vs parent onto base.

    Drops entries parent had but candidate removed; appends entries candidate
    added. Ensures a candidate that intends to *replace* a custom tool (by
    dropping the old entry and adding a new one in its own config.yaml) does
    not end up with both sitting side-by-side in merged.yaml.
    """
    cand_tr = candidate.get("tool_registry") or {}
    parent_tr = parent.get("tool_registry") or {}
    if not isinstance(cand_tr, dict) or not isinstance(parent_tr, dict):
        return
    cand_custom = cand_tr.get("custom") or []
    parent_custom = parent_tr.get("custom") or []
    if not isinstance(cand_custom, list) or not isinstance(parent_custom, list):
        return

    dropped = [e for e in parent_custom if e not in cand_custom]
    added = [e for e in cand_custom if e not in parent_custom]
    if not dropped and not added:
        return

    base_tr = base.setdefault("tool_registry", {})
    if not isinstance(base_tr, dict):
        base_tr = {}
        base["tool_registry"] = base_tr
    base_custom = base_tr.setdefault("custom", [])
    if not isinstance(base_custom, list):
        base_custom = []
        base_tr["custom"] = base_custom

    if dropped:
        base_custom[:] = [e for e in base_custom if e not in dropped]
    for entry in added:
        if entry not in base_custom:
            base_custom.append(entry)


def _apply_config(base: dict, candidate: dict, parent: dict) -> None:
    """Replace matching processor kwargs (matched by _target_) in base.

    Only touches kwargs whose values differ between base and candidate.
    ``parent`` is unused — config bucket only changes kwargs on entries that
    already exist in both parent and base.
    """
    del parent  # signature uniform across appliers
    base_procs = base.get("processors") or []
    cand_procs = candidate.get("processors") or []
    base_by_target: dict[str, dict] = {}
    for bp in base_procs:
        if isinstance(bp, dict):
            tgt = bp.get("_target_")
            if tgt:
                base_by_target[tgt] = bp
    for cp in cand_procs:
        if not isinstance(cp, dict):
            continue
        tgt = cp.get("_target_")
        if not tgt or tgt not in base_by_target:
            continue
        bp = base_by_target[tgt]
        for k, v in cp.items():
            if k in ("_target_", "_code_hash", "_hook_"):
                continue
            if bp.get(k) != v:
                bp[k] = v


def _apply_processor(base: dict, candidate: dict, parent: dict) -> None:
    """Apply candidate's processor diff vs parent onto base.

    Matches by ``_target_`` string. Drops entries parent had but candidate
    removed (the candidate's intent to *replace* an existing processor is
    expressed by omitting it from its config.yaml); appends entries candidate
    added that aren't already present in base.
    """
    parent_targets = {
        p.get("_target_")
        for p in (parent.get("processors") or [])
        if isinstance(p, dict) and p.get("_target_")
    }
    cand_procs = candidate.get("processors") or []
    cand_targets = {
        p.get("_target_")
        for p in cand_procs
        if isinstance(p, dict) and p.get("_target_")
    }

    removed_targets = parent_targets - cand_targets
    added_targets = cand_targets - parent_targets

    base_procs = base.setdefault("processors", [])
    if not isinstance(base_procs, list):
        return

    if removed_targets:
        base_procs[:] = [
            p for p in base_procs
            if not (isinstance(p, dict) and p.get("_target_") in removed_targets)
        ]

    existing = {
        p.get("_target_") for p in base_procs if isinstance(p, dict)
    }
    for cp in cand_procs:
        if not isinstance(cp, dict):
            continue
        tgt = cp.get("_target_")
        if tgt in added_targets and tgt not in existing:
            base_procs.append(cp)
            existing.add(tgt)


_BUCKET_APPLIERS = {
    "prompt": _apply_prompt,
    "tools": _apply_tools,
    "config": _apply_config,
    "processor": _apply_processor,
}


def compose_shipped_configs(
    base_config_path: Path,
    shipped: Iterable[tuple[str, str, Path]],
    output_path: Path,
) -> Path:
    """Merge shipped candidates' bucket-specific changes onto the base.

    Args:
        base_config_path: the round's starting config (pre-ship).
        shipped: iterable of ``(candidate_id, bucket, applied_yaml_path)``
            tuples, in rank order. Each candidate's `bucket` field names
            which fields of the base to overlay.
        output_path: where to write the merged result.

    Returns:
        The output_path.

    v0.9.3: ``bucket`` may be a str (legacy) OR a list of str (cross-
    bucket bundle). When a list is given, each bucket's applier is
    invoked in the declared order so the candidate's diff is merged in
    full. Stage 4 no longer enforces bucket-disjointness — merge
    conflicts are handled per-applier (later writes overwrite earlier
    for the same key).
    """
    base = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    # Freeze a parent snapshot so each applier can diff candidate vs parent
    # instead of candidate vs running base (which earlier appliers may have
    # mutated). Without this, a processor candidate that intends to replace
    # v1 with v2 by omitting v1 from its config.yaml would leave v1 in the
    # merged output alongside v2.
    parent = copy.deepcopy(base)
    for cid, bucket, applied_path in shipped:
        cand = yaml.safe_load(applied_path.read_text(encoding="utf-8")) or {}
        # Normalize bucket to a list.
        if isinstance(bucket, list):
            bucket_list = [str(b) for b in bucket if b]
        elif isinstance(bucket, str) and bucket:
            bucket_list = [bucket]
        else:
            continue
        for b in bucket_list:
            apply_fn = _BUCKET_APPLIERS.get(b)
            if apply_fn is None:
                continue
            apply_fn(base, cand, parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(base, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path
