# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Materialization helper: validate the Evolver's applied YAML.

The Evolver writes the applied HarnessConfig itself (see build_evolver_harness).
This module only verifies the result: the file exists, loads via HarnessConfig,
and canonicalizes. If the Evolver wrote garbage, ApplyError is raised and
Stage 4's canonicalize gate catches it.

Additionally, for prompt-bucket candidates, the applied YAML MUST reference a
``template_path`` that lives INSIDE the per-candidate ``applied_scratch_dir``
— otherwise the change is a silent no-op (R_{n+1} would run against the
unchanged shared template).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ApplyError(RuntimeError):
    """Raised when the Evolver's applied YAML is missing or malformed."""


@dataclass
class ApplyResult:
    applied_path: Path
    canonicalized: bool


def _is_inside(p: Path, root: Path) -> bool:
    """Return True if ``p`` is the same path as ``root`` or lives under it.

    Uses :meth:`Path.is_relative_to` (Py3.9+) with a ``relative_to``
    fallback. Beats string-prefix matching, which would misfire on
    sibling-prefix paths like ``/applied/C-R1-01`` vs
    ``/applied/C-R1-011/...``.
    """
    try:
        return p.is_relative_to(root)
    except AttributeError:
        # Py<3.9 fallback
        try:
            p.relative_to(root)
            return True
        except ValueError:
            return False


def _collect_template_paths(cfg) -> set[str]:
    """Return all ``template_path`` strings referenced by the applied cfg.

    Walks both the serialized ``processors`` list (dicts and nested dicts)
    and the runtime ``_rt_procs`` list (builder attribute access).
    """
    paths: set[str] = set()

    def _visit(node) -> None:
        if isinstance(node, dict):
            v = node.get("template_path")
            if isinstance(v, str) and v:
                paths.add(v)
            for sub in node.values():
                _visit(sub)
        elif isinstance(node, list):
            for sub in node:
                _visit(sub)

    processors = getattr(cfg, "processors", None)
    if processors is not None:
        _visit(processors)
    for p in getattr(cfg, "_rt_procs", None) or []:
        builder = getattr(p, "system_builder", None) or getattr(p, "builder", None) or p
        tpath = getattr(builder, "template_path", None)
        if isinstance(tpath, str) and tpath:
            paths.add(tpath)
    return paths


def validate_applied_config(
    applied_config_path: Path,
    *,
    expected_bucket: str | None = None,
    scratch_dir: Path | None = None,
) -> ApplyResult:
    """Verify the Evolver produced a loadable + canonicalizable applied YAML.

    Returns ApplyResult(applied_path, canonicalized=True) on success.
    Raises ApplyError if the file is missing, unreadable, or can't canonicalize.

    When ``expected_bucket == "prompt"`` AND ``scratch_dir`` is provided, also
    enforce that at least one ``template_path`` in the applied cfg points
    INSIDE ``scratch_dir``. This blocks the silent no-op where an Evolver
    writes a prompt-bucket candidate whose YAML still references the shared
    harnessx template (leaving R_{n+1} functionally unchanged).
    """
    from harnessx.core.harness import HarnessConfig
    p = Path(applied_config_path)
    if not p.exists():
        raise ApplyError(f"Evolver did not write {p}")
    try:
        cfg = HarnessConfig.from_yaml_file(p)
    except Exception as exc:
        raise ApplyError(f"Applied YAML at {p} failed to load: {exc}") from exc
    try:
        cfg.canonicalize()
    except Exception as exc:
        raise ApplyError(f"Applied YAML at {p} failed to canonicalize: {exc}") from exc

    if expected_bucket == "prompt" and scratch_dir is not None:
        scratch_abs = Path(scratch_dir).resolve()
        tpaths = _collect_template_paths(cfg)
        inside = [
            tp for tp in tpaths
            if _is_inside(Path(tp).resolve(), scratch_abs)
        ]
        if not inside:
            raise ApplyError(
                "prompt-bucket candidate must reference a scratch-dir "
                f"template_path; found: {sorted(tpaths)!r} (scratch_dir="
                f"{scratch_abs!s})"
            )
    return ApplyResult(applied_path=p, canonicalized=True)
