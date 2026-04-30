# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Post-flight validation + self-check CLIs for evolve rounds.

Everything lives here:

- ``StrictValidationError`` — exception type carrying a ``findings_path``
  pointer to the markdown artifact the failing check wrote.
- Individual check functions (``check_canonicalize`` /
  ``run_processor_dry_fire`` / ``run_tool_dry_fire`` /
  ``run_contract_check`` / ``run_literals_check`` / ``check_novelty``).
  Each writes a findings file under ``scratch_dir`` when it has anything
  to report; strict-mode flavours raise ``StrictValidationError``.
- ``EvolveValidator`` — orchestrator class used by ``MetaAgent.evolve``.
  Sequences three phases: **validity** (canonicalize → synthetic-task
  replay), **policy** (novelty + evidence, only when the structural
  diff is non-empty), **advisory** (literals, never blocks).
- Unified CLI: ``python -m harnessx.meta_harness.validate_workflow
  <subcommand> [args]``. Subcommands: ``canonicalize``, ``dry_fire``,
  ``contract``, ``literals``.

The meta-agent reads the ``validate`` skill in its workspace for the
CLI contract; the orchestrator calls the Python API directly via
``EvolveValidator.run``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .replay import run_replay_gate_strict

if TYPE_CHECKING:
    from ..core.harness import HarnessConfig
    from ..core.model_config import ModelConfig

logger = logging.getLogger(__name__)


# ─── Shared types ─────────────────────────────────────────────────────────


class StrictValidationError(Exception):
    """Raised by checks in strict mode when blocking findings are present.

    Carries the path to the artifact file the check wrote so the
    orchestrator can surface it in the final ``RuntimeError`` pointing
    the caller (recipe) at concrete evidence. ``kind`` is a short
    category used for logging / routing (e.g. ``"contract"``,
    ``"processor_dryfire"``, ``"tool_dryfire"``,
    ``"task_specific_literals"``, ``"replay_gate"``,
    ``"reverted_hypothesis_reused"``, ``"reverted_signature_reused"``).
    """

    def __init__(self, kind: str, message: str, findings_path: Path):
        super().__init__(message)
        self.kind = kind
        self.findings_path = findings_path


@dataclass(frozen=True)
class ValidateReport:
    """Result of one check inside the orchestrator workflow."""

    phase: str  # "validity" | "policy" | "advisory"
    check: str  # "canonicalize" | "replay" | "novelty" | "evidence" | "literals"
    ok: bool
    reason: str = ""
    findings_path: Path | None = None


@dataclass(frozen=True)
class ValidateOutcome:
    """Bundle of workflow outputs the caller needs downstream."""

    canon_cfg: "HarnessConfig"
    diff: dict[str, Any]
    reports: tuple[ValidateReport, ...]


# ─── canonicalize ─────────────────────────────────────────────────────────


def _eager_check_system_prompt_builders(cfg: "HarnessConfig") -> int:
    """Force any lazy template load to happen NOW.

    Returns the number of ``TemplateSystemPromptBuilder`` instances
    successfully verified. Raises ``ValueError`` on the first defect
    found (missing / empty / broken-Jinja template file).
    """
    checked = 0
    rt_procs = getattr(cfg, "_rt_procs", None) or []

    try:
        from ..processors.context.strategies.system_prompt.template import (
            TemplateSystemPromptBuilder,
        )
        from ..processors.context.system_prompt import SystemPromptProcessor
    except ImportError:
        SystemPromptProcessor = None  # type: ignore[assignment]
        TemplateSystemPromptBuilder = None  # type: ignore[assignment]

    for p in rt_procs:
        if SystemPromptProcessor is not None:
            if not isinstance(p, SystemPromptProcessor):
                continue
        else:
            if not hasattr(p, "system_builder"):
                continue
        sb = getattr(p, "system_builder", None)
        if sb is None:
            continue
        if TemplateSystemPromptBuilder is not None:
            if not isinstance(sb, TemplateSystemPromptBuilder):
                continue
        else:
            if not hasattr(sb, "template_path"):
                continue
        tpath = getattr(sb, "template_path", None)
        if not tpath:
            raise ValueError("TemplateSystemPromptBuilder has no template_path set")
        try:
            with open(tpath, "r", encoding="utf-8") as f:
                src = f.read()
        except FileNotFoundError as exc:
            raise ValueError(
                f"SystemPromptProcessor.system_builder.template_path "
                f"points to a file that does not exist: {tpath!r}. If "
                "this is a new template authored under "
                "`output_dir/templates/`, make sure you actually wrote "
                "it before referencing it from config.yaml."
            ) from exc
        except OSError as exc:
            raise ValueError(f"Cannot read template_path {tpath!r}: {exc}") from exc

        if not src.strip():
            raise ValueError(
                f"template_path {tpath!r} is empty. An empty system "
                "prompt strips all benchmark guidance from the inner "
                "agent — copy the current template as a starting point, "
                "then edit the copy."
            )

        try:
            from jinja2 import Template, TemplateSyntaxError
        except ImportError:
            return checked

        try:
            Template(src)
        except TemplateSyntaxError as exc:
            raise ValueError(
                f"template_path {tpath!r} has invalid Jinja syntax: "
                f"{exc.message} (line {exc.lineno}). Fix the template "
                "and re-verify."
            ) from exc
        checked += 1
    return checked


def check_canonicalize(config_yaml: Path) -> dict:
    """Run canonicalize + eager template check on ``config_yaml``.

    Returns ``{"ok": True, "checked_templates": N}`` on success, or
    ``{"ok": False, "error": ..., "error_type": ...}`` on failure.
    Never raises — callers inspect ``ok`` and read ``error`` when False.
    """
    from ..core.harness import HarnessConfig

    config_yaml = Path(config_yaml)
    if not config_yaml.is_file():
        return {
            "ok": False,
            "error": f"config.yaml not found at {config_yaml}",
            "error_type": "FileNotFoundError",
        }

    try:
        cfg = HarnessConfig.from_yaml_file(config_yaml).canonicalize()
        checked = _eager_check_system_prompt_builders(cfg)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

    if checked:
        logger.info("[canonicalize] eager-checked %d TemplateSystemPromptBuilder(s)", checked)
    return {"ok": True, "checked_templates": checked}


# ─── dry-fire (processors + tools) ────────────────────────────────────────


def _dummy_input_from_schema(schema: dict) -> dict:
    """Produce a minimal dict that satisfies ``schema.required``."""
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    if not isinstance(props, dict):
        props = {}
    if not isinstance(required, list):
        required = []

    typed_defaults: dict[str, Any] = {
        "integer": 0,
        "number": 0.0,
        "boolean": False,
        "array": [],
        "object": {},
        "string": "",
    }

    dummy: dict = {}
    for name in required:
        if not isinstance(name, str):
            continue
        prop = props.get(name) or {}
        t = prop.get("type") if isinstance(prop, dict) else None
        dummy[name] = typed_defaults.get(t, "")
    return dummy


async def run_processor_dry_fire(
    cfg: "HarnessConfig",
    scratch_dir: Path,
    *,
    strict: bool = False,
) -> dict:
    """Invoke each custom processor's declared ``on_*`` hooks once with a
    minimal dummy event, to surface field-name typos and constructor
    shape drift.

    "Custom" means any processor whose class module does NOT start with
    ``harnessx.`` (i.e. loaded via a ``file://`` entry).
    """
    from ..core.events import (
        BeforeModelEvent,
        ModelResponseEvent,
        StepEndEvent,
        StepStartEvent,
        TaskEndEvent,
        TaskStartEvent,
        ToolCallEvent,
        ToolResultEvent,
    )

    dummy = {
        "on_task_start": TaskStartEvent(run_id="dryrun", step_id=0),
        "on_step_start": StepStartEvent(run_id="dryrun", step_id=0),
        "on_before_model": BeforeModelEvent(run_id="dryrun", step_id=0),
        "on_after_model": ModelResponseEvent(run_id="dryrun", step_id=0),
        "on_before_tool": ToolCallEvent(run_id="dryrun", step_id=0),
        "on_after_tool": ToolResultEvent(run_id="dryrun", step_id=0),
        "on_step_end": StepEndEvent(run_id="dryrun", step_id=0),
        "on_task_end": TaskEndEvent(run_id="dryrun", step_id=0),
    }

    likely_bugs: list[str] = []
    notes: list[str] = []
    seen: set[tuple[str, str]] = set()

    rt_procs = getattr(cfg, "_rt_procs", None) or []
    for p in rt_procs:
        cls = type(p)
        mod = cls.__module__ or ""
        if mod.startswith("harnessx.") or mod == "__main__":
            continue
        key = (mod, cls.__qualname__)
        if key in seen:
            continue
        seen.add(key)

        for hook_name, ev in dummy.items():
            method = getattr(cls, hook_name, None)
            if method is None or not callable(method):
                continue
            qual = getattr(method, "__qualname__", "") or ""
            if qual.startswith("MultiHookProcessor."):
                continue
            try:
                gen = method(p, ev)
                async for _ in gen:
                    break
            except AttributeError as exc:
                msg = str(exc)
                if "object has no attribute" in msg and ("Event" in msg or "Message" in msg):
                    likely_bugs.append(
                        f"- **{cls.__name__}.{hook_name}** — "
                        f"`AttributeError: {msg}` (likely a field-name "
                        f"typo; read `harnessx/core/events.py` and fix)"
                    )
                else:
                    notes.append(f"- {cls.__name__}.{hook_name} — AttributeError: {msg}")
            except TypeError as exc:
                msg = str(exc)
                if "unexpected keyword argument" in msg:
                    likely_bugs.append(
                        f"- **{cls.__name__}.{hook_name}** — "
                        f"`TypeError: {msg}` (constructing an event / "
                        f"message with a stale field; check the "
                        f"dataclass definition)"
                    )
                else:
                    notes.append(f"- {cls.__name__}.{hook_name} — TypeError: {msg}")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"- {cls.__name__}.{hook_name} — {type(exc).__name__}: {exc}")

    if not likely_bugs and not notes:
        return {"likely_bugs": 0, "notes": 0, "artifact": None}

    out = scratch_dir / "DRY_FIRE_WARNINGS.md"
    lines = ["# Dry-Fire Warnings", ""]
    if likely_bugs:
        lines += [
            "## Likely bugs",
            "",
            "These almost certainly fire at runtime in the next round. Fix before the next benchmark pass.",
            "",
            *likely_bugs,
            "",
        ]
    if notes:
        lines += [
            "## Notes (expected for some processors)",
            "",
            "These are often false positives — the hook reads "
            "`event.state` or other fields that are None in a minimal "
            "dummy. Only act if the message names a field that also "
            "doesn't exist on a real event.",
            "",
            *notes,
            "",
        ]
    out.write_text("\n".join(lines), encoding="utf-8")
    if likely_bugs:
        logger.warning("[dry_fire] %d likely bug(s) in custom processors — see %s", len(likely_bugs), out)
    if strict and likely_bugs:
        raise StrictValidationError(
            kind="processor_dryfire",
            message=f"{len(likely_bugs)} likely bug(s) in custom processors — see {out}",
            findings_path=out,
        )
    return {
        "likely_bugs": len(likely_bugs),
        "notes": len(notes),
        "artifact": str(out),
    }


async def run_tool_dry_fire(
    cfg: "HarnessConfig",
    scratch_dir: Path,
    *,
    strict: bool = False,
) -> dict:
    """Call each custom ``@tool`` with dummy args and catch definite bugs
    (ImportError / NameError / SyntaxError in fn body)."""
    from ..tools.base import _execute_tool

    registry = getattr(cfg, "tool_registry", None)
    if registry is None:
        return {"likely_bugs": 0, "notes": 0, "artifact": None}
    tools_map = getattr(registry, "_tools", None)
    if not tools_map:
        return {"likely_bugs": 0, "notes": 0, "artifact": None}

    likely_bugs: list[str] = []
    notes: list[str] = []

    for tool_name, t in tools_map.items():
        fn = getattr(t, "fn", None)
        if fn is None:
            continue
        mod = getattr(fn, "__module__", "") or ""
        if mod.startswith("harnessx.") or mod == "__main__":
            continue

        schema = getattr(t, "input_schema", None) or {}
        dummy_input = _dummy_input_from_schema(schema)

        try:
            result = await asyncio.wait_for(_execute_tool(t, dummy_input), timeout=1.5)
        except asyncio.TimeoutError:
            notes.append(f"- {tool_name} — timed out after 1.5s (likely doing real IO on dummy input; not diagnostic)")
            continue
        except Exception as exc:  # defensive
            notes.append(f"- {tool_name} — harness-level: {type(exc).__name__}: {str(exc)[:160]}")
            continue

        err = (result.error or "").strip()
        if not err:
            continue

        lowered = err.lower()
        if "no module named" in lowered or "importerror" in lowered or "modulenotfound" in lowered:
            likely_bugs.append(
                f"- **{tool_name}** — `ImportError` / `ModuleNotFoundError` "
                f"in fn body: {err[:180]} (hallucinated dep; either add "
                f"the dep, fall back to stdlib, or flag in "
                f"`_meta_scratch/NEEDS_FROM_HUMAN.md` and choose a different approach)"
            )
        elif "nameerror" in lowered:
            likely_bugs.append(
                f"- **{tool_name}** — `NameError` in fn body: {err[:180]} (undefined symbol; typo or missing import)"
            )
        elif "syntaxerror" in lowered:
            likely_bugs.append(f"- **{tool_name}** — `SyntaxError`: {err[:180]}")
        else:
            notes.append(f"- {tool_name} — {err[:180]}")

    if not likely_bugs and not notes:
        return {"likely_bugs": 0, "notes": 0, "artifact": None}

    out = scratch_dir / "DRY_FIRE_TOOL_WARNINGS.md"
    lines = ["# Dry-Fire Warnings (Tools)", ""]
    if likely_bugs:
        lines += [
            "## Likely bugs",
            "",
            "These fire as soon as the tool is called in the next round. "
            "The errors are in the fn body and `canonicalize()` cannot see them. Fix before shipping.",
            "",
            *likely_bugs,
            "",
        ]
    if notes:
        lines += [
            "## Notes (may be false positives)",
            "",
            "Errors recorded when dummy input was fed to a side-effectful "
            "tool. If the tool really hits the network or filesystem, a "
            "failure here is expected. Act only if the message names an "
            "import / symbol that also wouldn't exist in a real call.",
            "",
            *notes,
            "",
        ]
    out.write_text("\n".join(lines), encoding="utf-8")
    if likely_bugs:
        logger.warning("[dry_fire] %d likely bug(s) in custom tools — see %s", len(likely_bugs), out)
    if strict and likely_bugs:
        raise StrictValidationError(
            kind="tool_dryfire",
            message=f"{len(likely_bugs)} likely bug(s) in custom tools — see {out}",
            findings_path=out,
        )
    return {
        "likely_bugs": len(likely_bugs),
        "notes": len(notes),
        "artifact": str(out),
    }


# ─── contract ─────────────────────────────────────────────────────────────


async def run_contract_check(
    cfg: "HarnessConfig",
    scratch_dir: Path,
    *,
    strict: bool = False,
) -> dict:
    """Exercise custom processors' messages-mutating hooks against
    fixtures and record contract violations.

    Writes ``scratch_dir/CONTRACT_VIOLATIONS.md`` on finding. When
    ``strict`` is True, raises ``StrictValidationError`` after writing
    the file.
    """
    from ..core.contract_check import ContractViolation, check_processor_contract
    from ..core.harness import _instantiate_proc

    findings: list[ContractViolation] = []
    seen: set[tuple[str, str]] = set()

    # Collect processors: already-instantiated (_rt_procs) + from config dicts.
    candidates: list = list(getattr(cfg, "_rt_procs", None) or [])
    for p in cfg.processors or []:
        if isinstance(p, dict) and "_target_" in p:
            target = p.get("_target_", "")
            if target.startswith("harnessx."):
                continue
            try:
                inst = _instantiate_proc(p)
                if inst is not None:
                    candidates.append(inst)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[contract] failed to instantiate %s: %r", target, exc)

    for p in candidates:
        cls = type(p)
        mod = cls.__module__ or ""
        if mod.startswith("harnessx.") or mod == "__main__":
            continue
        key = (mod, cls.__qualname__)
        if key in seen:
            continue
        seen.add(key)
        try:
            findings.extend(await check_processor_contract(p))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[contract] check_processor_contract crashed on %s: %r", cls.__name__, exc)

    if not findings:
        return {"violations": 0, "processors": 0, "artifact": None}

    out = scratch_dir / "CONTRACT_VIOLATIONS.md"
    by_proc: dict[str, list[ContractViolation]] = {}
    for v in findings:
        by_proc.setdefault(v.processor, []).append(v)

    lines = [
        "# Contract Violations",
        "",
        "Custom processors in this evolve round violate the HarnessX hook "
        "contract. The target agent will emit `CONTRACT [violation_type] "
        "hook=... processor=...` warnings at runtime, and will abort under "
        "`HARNESSX_CONTRACT_MODE=strict`.",
        "",
        "Contract rules live in "
        "`harnessx/core/processor.py::_validate_messages_contract`; the "
        "`reference` skill carries a quick recap.",
        "",
    ]
    for proc_name, vs in by_proc.items():
        lines.append(f"## {proc_name}")
        lines.append("")
        for v in vs:
            lines.append(f"- **{v.hook}** [{v.violation_type}] fixture=`{v.fixture}` step={v.step_id}: {v.message}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.warning(
        "[contract] %d violation(s) across %d processor(s) — see %s",
        len(findings),
        len(by_proc),
        out,
    )
    if strict:
        raise StrictValidationError(
            kind="contract",
            message=f"{len(findings)} contract violation(s) across {len(by_proc)} processor(s) — see {out}",
            findings_path=out,
        )
    return {
        "violations": len(findings),
        "processors": len(by_proc),
        "artifact": str(out),
    }


# ─── literals ─────────────────────────────────────────────────────────────


# Generic UUID shape only. Benchmark-specific id shapes belong in the
# benchmark's playbook skill. Callers that want a benchmark-specific
# pattern can pass it via ``extra_patterns``.
_DEFAULT_LITERAL_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    (
        "uuid",
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
    ),
]
_LITERAL_STRICT_THRESHOLD = 3


async def run_literals_check(
    output_dir: Path,
    scratch_dir: Path,
    *,
    strict: bool = False,
    extra_patterns: list[tuple[str, "re.Pattern[str]"]] | None = None,
) -> dict:
    """Scan authored tools and processors for task-specific hardcoded
    literals.

    Writes ``scratch_dir/TASK_SPECIFIC_LITERALS.md`` always (even when
    no findings, for consistency with the evolve retry loop's "cite a
    file" contract).

    Returns ``{"findings": int, "threshold": int, "artifact": str | None}``.
    """
    patterns = list(_DEFAULT_LITERAL_PATTERNS)
    if extra_patterns:
        patterns.extend(extra_patterns)

    findings: list[tuple[Path, int, str, str]] = []  # (file, line, kind, excerpt)

    for sub in ("tools", "processors"):
        d = output_dir / sub
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if not (p.is_file() and p.suffix == ".py"):
                continue
            if p.name.startswith("__"):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                for kind, pat in patterns:
                    m = pat.search(line)
                    if m is not None:
                        excerpt = line.strip()[:120]
                        findings.append((p, lineno, kind, excerpt))

    out = scratch_dir / "TASK_SPECIFIC_LITERALS.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["# Task-specific literal scan", ""]
    if not findings:
        lines.append("No task-id-shaped literals found in authored tools or")
        lines.append("processors. This scan is syntactic — absence of matches")
        lines.append("does not prove generality on its own.")
        lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
        return {
            "findings": 0,
            "threshold": _LITERAL_STRICT_THRESHOLD,
            "artifact": str(out),
        }

    lines.append(
        f"Found **{len(findings)}** task-specific literal match(es) across "
        f"authored tools/processors (strict threshold: {_LITERAL_STRICT_THRESHOLD})."
    )
    lines.append("")
    lines.append("Authored tools and processors must help a class of tasks.")
    lines.append("Hardcoded task IDs, UUIDs of specific tasks, or benchmark-")
    lines.append("specific literals are grounds for revert — they pass")
    lines.append("canonicalize and dry-fire but degrade every task whose id")
    lines.append("does not match the one you memorised.")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    for file, lineno, kind, excerpt in findings:
        try:
            rel = file.relative_to(output_dir)
        except ValueError:
            rel = file
        lines.append(f"- `{rel}:{lineno}` [{kind}] `{excerpt}`")
    lines.append("")
    lines.append("## How to fix")
    lines.append("")
    lines.append("Replace the literal with a parameter, a config field, or")
    lines.append("a pattern that matches the task class (a filetype, a URL")
    lines.append("prefix, a judge-missing-capability shape). If the literal")
    lines.append("is legitimately needed for a throwaway experiment, keep")
    lines.append("it under `_meta_scratch/` — only `tools/` and `processors/`")
    lines.append("are scanned.")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")

    if strict and len(findings) >= _LITERAL_STRICT_THRESHOLD:
        raise StrictValidationError(
            kind="task_specific_literals",
            message=f"{len(findings)} task-specific literal(s) in authored tools/processors — see {out}",
            findings_path=out,
        )
    return {
        "findings": len(findings),
        "threshold": _LITERAL_STRICT_THRESHOLD,
        "artifact": str(out),
    }


# ─── novelty ──────────────────────────────────────────────────────────────


_LABEL_SIMILARITY_WARN = 0.85


def _novelty_signature(levers, predicted) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lev = tuple(sorted(str(x) for x in (levers or []) if x))
    pre = tuple(sorted(str(x) for x in (predicted or []) if x))
    return lev, pre


def check_novelty(memo_path: Path, scratch_dir: Path) -> None:
    """Raise ``StrictValidationError`` when the latest journal entry
    re-proposes a reverted hypothesis (by id or by (lever, predicted)
    signature). No-op when the memo is empty or absent.
    """
    memo_path = Path(memo_path)
    if not memo_path.is_file():
        return

    from .journal import read_entries

    entries = read_entries(memo_path)
    if not entries:
        return

    latest = entries[-1]
    hid = latest.hypothesis_id.strip()
    priors = entries[:-1]

    reverted = [e for e in priors if e.gating_outcome == "reverted"]
    if not reverted:
        return

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    findings_path = scratch_dir / "NOVELTY_FAIL.md"

    # 1. Exact id reuse — always blocks.
    if hid:
        exact_match = next((e for e in reverted if e.hypothesis_id.strip() == hid), None)
        if exact_match is not None:
            findings_path.write_text(
                "# Novelty check failed — reverted hypothesis_id reused\n\n"
                f"Round {latest.round} (`{latest.label}`) re-uses the "
                f"hypothesis_id `{hid}`, which was already marked "
                f"`reverted` in round {exact_match.round} "
                f"(`{exact_match.label}`).\n\n"
                "If you genuinely have new evidence, rename the "
                "`hypothesis_id` (e.g. append `_v2`) and explain in prose "
                "what evidence changed since the last try.\n",
                encoding="utf-8",
            )
            raise StrictValidationError(
                kind="reverted_hypothesis_reused",
                message=(
                    f"hypothesis_id {hid!r} was reverted in round {exact_match.round}; "
                    f"round {latest.round} must use a new id or justify re-use"
                ),
                findings_path=findings_path,
            )

    # 2. Signature reuse.
    latest_sig = _novelty_signature(latest.levers, latest.predicted_affected)
    if not latest_sig[0] or not latest_sig[1]:
        return

    sig_match = next(
        (e for e in reverted if _novelty_signature(e.levers, e.predicted_affected) == latest_sig),
        None,
    )
    if sig_match is not None:
        retry_rationale = latest.frontmatter.get("retry_rationale")
        if isinstance(retry_rationale, str) and retry_rationale.strip():
            _novelty_write_rationale_note(
                scratch_dir=scratch_dir,
                latest=latest,
                sig_match=sig_match,
                rationale=retry_rationale.strip(),
            )
        else:
            levers_str = ",".join(latest_sig[0]) or "(none)"
            predicted_str = ",".join(latest_sig[1]) or "(none)"
            findings_path.write_text(
                "# Novelty check failed — reverted signature reused\n\n"
                f"Round {latest.round} (`{latest.label}`, id=`{hid or '?'}`) "
                f"uses the same `(levers, predicted_affected)` signature as "
                f"reverted round {sig_match.round} (`{sig_match.label}`, "
                f"id=`{sig_match.hypothesis_id}`):\n\n"
                f"- levers: `[{levers_str}]`\n"
                f"- predicted_affected: `[{predicted_str}]`\n\n"
                "Renaming the hypothesis_id bypasses exact-id novelty but "
                "not signature novelty — the same lever targeting the same "
                "task cluster that was reverted before needs a new-evidence "
                "rationale.\n\n"
                "If your read of the trajectories genuinely justifies "
                "retrying this cluster, add a `retry_rationale` field to "
                "your journal entry's frontmatter with a one-line summary "
                "of **what evidence is new**:\n\n"
                '```\nretry_rationale: "R{prior_round} tried X; this round '
                'found Y which the prior attempt ignored"\n```\n\n'
                "Otherwise pick a different lever or a different cluster.\n",
                encoding="utf-8",
            )
            raise StrictValidationError(
                kind="reverted_signature_reused",
                message=(
                    f"round {latest.round} shares (levers={latest_sig[0]}, "
                    f"predicted={latest_sig[1]}) with reverted round "
                    f"{sig_match.round}; add `retry_rationale` to frontmatter "
                    "or change the approach"
                ),
                findings_path=findings_path,
            )

    # 3. Soft label-similarity warning — non-blocking.
    similar = _novelty_similar_reverted_labels(latest.label, reverted)
    if similar:
        _novelty_write_soft_similarity_note(scratch_dir=scratch_dir, latest=latest, similar=similar)


def _novelty_similar_reverted_labels(label: str, reverted: list) -> list[tuple[float, object]]:
    label = (label or "").strip().lower()
    if not label:
        return []
    hits: list[tuple[float, object]] = []
    for e in reverted:
        other = (e.label or "").strip().lower()
        if not other:
            continue
        ratio = SequenceMatcher(None, label, other).ratio()
        if ratio >= _LABEL_SIMILARITY_WARN:
            hits.append((ratio, e))
    hits.sort(key=lambda t: t[0], reverse=True)
    return hits


def _novelty_write_rationale_note(*, scratch_dir: Path, latest, sig_match, rationale: str) -> None:
    note = scratch_dir / "NOVELTY_RATIONALE.md"
    note.write_text(
        "# Novelty — signature collision allowed with rationale\n\n"
        f"Round {latest.round} (`{latest.label}`) shares the "
        f"`(levers, predicted_affected)` signature of reverted round "
        f"{sig_match.round} (`{sig_match.label}`), but the entry's "
        "frontmatter provided `retry_rationale`, so the round was not "
        "blocked.\n\n"
        f"**Rationale:** {rationale}\n\n"
        "If the rationale turns out to be unfounded, the gating layer "
        "will revert this round in turn and the cluster will be "
        "marked doubly-reverted — making a third retry even harder "
        "to justify.\n",
        encoding="utf-8",
    )


def _novelty_write_soft_similarity_note(*, scratch_dir: Path, latest, similar) -> None:
    note = scratch_dir / "NOVELTY_SOFT_WARN.md"
    lines = [
        "# Novelty soft warning — similar reverted labels\n",
        "",
        f"Round {latest.round}'s label (`{latest.label}`) is textually "
        "similar to one or more earlier reverted labels. This is **not** "
        "a failure — the signature check passed — but the next round's "
        "CONTEXT reviewer should know the lineage:\n",
        "",
    ]
    for ratio, e in similar:
        lines.append(f"- R{e.round} `{e.hypothesis_id}` — `{e.label}` (similarity {ratio:.2f})")
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─── EvolveValidator — orchestrator class ────────────────────────────────


class EvolveValidator:
    """Runs the post-flight validation pipeline for one evolve round.

    Three phases:

    1. **validity** — ``canonicalize`` + ``compute_changeset`` +
       synthetic-task replay. First failure raises ``RuntimeError``.
    2. **policy** — ``novelty`` + ``evidence``. Only runs when the
       structural diff is non-empty. First failure raises
       ``RuntimeError``.
    3. **advisory** — ``literals``. Always runs; never blocks. Writes
       findings under ``_meta_scratch/`` for next-round review.
    """

    def __init__(
        self,
        *,
        inner_model: "ModelConfig",
        memo_path: Path | None = None,
        replay_model: "ModelConfig | None" = None,
        replay_max_steps: int | None = None,
        replay_max_cost_usd: float | None = None,
        replay_timeout_s: float = 20.0,
        require_evidence: bool = True,
    ) -> None:
        self._inner_model = inner_model
        self._memo_path = memo_path
        self._replay_model = replay_model or inner_model
        self._replay_max_steps = replay_max_steps
        self._replay_max_cost_usd = replay_max_cost_usd
        self._replay_timeout_s = replay_timeout_s
        self._require_evidence = require_evidence

    async def run(
        self,
        *,
        out_yaml: Path,
        current_config: "HarnessConfig | Path",
        scratch_dir: Path,
        output_dir: Path,
        elapsed: float,
        compute_changeset_fn: Any,
    ) -> ValidateOutcome:
        """Execute validity → policy → advisory.

        Returns the canonicalized config + structural diff on success;
        raises ``RuntimeError`` on any validity or policy failure with a
        pointer to the findings file.
        """
        reports: list[ValidateReport] = []

        # Validity
        canon_cfg = self._canonicalize(out_yaml, scratch_dir, elapsed, reports)
        await self._contract(
            canon_cfg=canon_cfg,
            scratch_dir=scratch_dir,
            out_yaml=out_yaml,
            reports=reports,
        )
        diff = self._compute_diff(
            current_config=current_config,
            canon_cfg=canon_cfg,
            scratch_dir=scratch_dir,
            compute_changeset_fn=compute_changeset_fn,
        )
        await self._replay(
            canon_cfg=canon_cfg,
            scratch_dir=scratch_dir,
            out_yaml=out_yaml,
            reports=reports,
        )

        # Policy (only when diff non-empty)
        if diff:
            self._novelty(scratch_dir, out_yaml, reports)
            if self._require_evidence:
                self._evidence(scratch_dir, out_yaml, diff, reports)
        else:
            logger.info("[validate] noop round (empty changeset) — skipping policy phase")

        # Advisory (never blocks)
        await self._literals_advisory(output_dir, scratch_dir, reports)

        return ValidateOutcome(canon_cfg=canon_cfg, diff=diff, reports=tuple(reports))

    # ----- Phase 1: validity -------------------------------------------------

    def _canonicalize(
        self,
        out_yaml: Path,
        scratch_dir: Path,
        elapsed: float,
        reports: list[ValidateReport],
    ) -> "HarnessConfig":
        from ..core.harness import HarnessConfig

        try:
            canon_cfg = HarnessConfig.from_yaml_file(out_yaml).canonicalize()
            _eager_check_system_prompt_builders(canon_cfg)
        except Exception as exc:  # noqa: BLE001 — arbitrary Jinja + import code
            error_md = scratch_dir / "CANONICALIZE_ERROR.md"
            error_md.write_text(
                "# Canonicalize Failure\n\n"
                f"`config.yaml` produced after {elapsed:.1f}s fails "
                f"`HarnessConfig.from_yaml_file(...).canonicalize()`:\n\n"
                f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                "The meta-agent should have run `python -m "
                "harnessx.meta_harness.validate_workflow canonicalize "
                f"{out_yaml}` before ending its turn. See the "
                "`validate` skill for the full self-check suite.\n",
                encoding="utf-8",
            )
            reports.append(
                ValidateReport(
                    phase="validity",
                    check="canonicalize",
                    ok=False,
                    reason=f"{type(exc).__name__}: {exc}",
                    findings_path=error_md,
                )
            )
            raise RuntimeError(
                f"meta-agent produced {out_yaml} but it does not canonicalize: "
                f"{type(exc).__name__}: {exc}. See {error_md}."
            ) from exc

        reports.append(ValidateReport(phase="validity", check="canonicalize", ok=True))
        return canon_cfg

    @staticmethod
    def _compute_diff(
        *,
        current_config: "HarnessConfig | Path",
        canon_cfg: "HarnessConfig",
        scratch_dir: Path,
        compute_changeset_fn: Any,
    ) -> dict[str, Any]:
        from ..core.harness import HarnessConfig

        try:
            if isinstance(current_config, Path):
                before_cfg = HarnessConfig.from_yaml_file(current_config).canonicalize()
            else:
                before_cfg = current_config.canonicalize()
            diff = compute_changeset_fn(before_cfg, canon_cfg)
        except Exception as exc:  # noqa: BLE001 — diff is best-effort
            logger.warning("[validate] changeset computation failed: %s", exc)
            diff = {}
        (scratch_dir / "changeset.json").write_text(
            json.dumps(diff, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return diff

    async def _replay(
        self,
        *,
        canon_cfg: "HarnessConfig",
        scratch_dir: Path,
        out_yaml: Path,
        reports: list[ValidateReport],
    ) -> None:
        try:
            await run_replay_gate_strict(
                canon_cfg,
                scratch_dir,
                replay_model=self._replay_model,
                max_steps=self._replay_max_steps,
                max_cost_usd=self._replay_max_cost_usd,
                timeout_s=self._replay_timeout_s,
                replay_mode="synthetic_task",
            )
        except StrictValidationError as exc:
            reports.append(
                ValidateReport(
                    phase="validity",
                    check="replay",
                    ok=False,
                    reason=str(exc),
                    findings_path=exc.findings_path,
                )
            )
            raise RuntimeError(
                f"meta-agent produced {out_yaml} but the replay gate rejected it: {exc}. See {exc.findings_path}."
            ) from exc

        reports.append(ValidateReport(phase="validity", check="replay", ok=True))

    async def _contract(
        self,
        *,
        canon_cfg: "HarnessConfig",
        scratch_dir: Path,
        out_yaml: Path,
        reports: list[ValidateReport],
    ) -> None:
        """Block the round if custom processors violate the hook-mutation contract."""
        result = await run_contract_check(canon_cfg, scratch_dir, strict=False)
        violations = result.get("violations", 0)
        if violations > 0:
            findings_path = Path(result["artifact"]) if result.get("artifact") else None
            reports.append(
                ValidateReport(
                    phase="validity",
                    check="contract",
                    ok=False,
                    reason=f"{violations} contract violation(s)",
                    findings_path=findings_path,
                )
            )
            raise RuntimeError(
                f"meta-agent produced {out_yaml} but custom processors violate "
                f"the hook-mutation contract ({violations} violation(s)). "
                f"See {findings_path}."
            )
        reports.append(ValidateReport(phase="validity", check="contract", ok=True))

    # ----- Phase 2: policy (only when diff non-empty) ------------------------

    def _novelty(
        self,
        scratch_dir: Path,
        out_yaml: Path,
        reports: list[ValidateReport],
    ) -> None:
        if self._memo_path is None:
            reports.append(
                ValidateReport(
                    phase="policy",
                    check="novelty",
                    ok=True,
                    reason="no memo_path — skipped",
                )
            )
            return
        try:
            check_novelty(self._memo_path, scratch_dir)
        except StrictValidationError as exc:
            reports.append(
                ValidateReport(
                    phase="policy",
                    check="novelty",
                    ok=False,
                    reason=str(exc),
                    findings_path=exc.findings_path,
                )
            )
            raise RuntimeError(
                f"meta-agent produced {out_yaml} but re-proposed a reverted hypothesis: {exc}. See {exc.findings_path}."
            ) from exc

        reports.append(ValidateReport(phase="policy", check="novelty", ok=True))

    def _evidence(
        self,
        scratch_dir: Path,
        out_yaml: Path,
        diff: dict[str, Any],
        reports: list[ValidateReport],
    ) -> None:
        """candidates.md must exist, contain C-N sections, and cross-reference
        with the journal's latest `cited_candidates` frontmatter.
        """
        candidates_path = scratch_dir / "candidates.md"
        findings_path = scratch_dir / "EVIDENCE_FAIL.md"

        def _fail(msg_body: str, short: str) -> None:
            findings_path.write_text(msg_body, encoding="utf-8")
            reports.append(
                ValidateReport(
                    phase="policy",
                    check="evidence",
                    ok=False,
                    reason=short,
                    findings_path=findings_path,
                )
            )
            raise RuntimeError(f"evidence gate: {short}. See {findings_path}.")

        if not candidates_path.is_file():
            _fail(
                "# Evidence Gate Failure\n\n"
                "The config changed this round but `_meta_scratch/candidates.md` "
                "does not exist. The orchestrator requires each non-noop round "
                "to document candidate hypotheses and link them to the config "
                "diff.\n\n"
                "Write a `candidates.md` next round with sections of the form:\n\n"
                "```\n## Candidate C-001\n[lens: ... | lever: ... | intent: ...]\n"
                "<evidence + reasoning + retroactive check>\n```\n\n"
                "See the `analyze` skill for the full schema. Then reference "
                "at least one ID in the journal entry's `cited_candidates` "
                "frontmatter.\n",
                f"config changed but {candidates_path} is missing",
            )

        text = candidates_path.read_text(encoding="utf-8", errors="replace")
        candidate_ids = sorted(set(re.findall(r"^##\s+Candidate\s+(C-\d+)\b", text, flags=re.MULTILINE)))
        if not candidate_ids:
            _fail(
                "# Evidence Gate Failure\n\n"
                f"`{candidates_path}` exists but contains no `## Candidate "
                "C-N` section. The orchestrator expects at least one canonical "
                "candidate ID per non-noop round so the journal entry's "
                "`cited_candidates` frontmatter can link back to specific "
                "evidence.\n\n"
                "Add sections like:\n\n"
                "```\n## Candidate C-001\n[lens: capability-gap | lever: "
                "control | intent: corrective]\n"
                "<evidence + reasoning + retroactive check>\n```\n"
                "See the `analyze` skill for the full schema.\n",
                f"{candidates_path} has no `## Candidate C-N` sections",
            )

        if self._memo_path is None:
            reports.append(ValidateReport(phase="policy", check="evidence", ok=True))
            return
        try:
            from .journal import latest_entry

            latest = latest_entry(self._memo_path)
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning("[validate] could not read latest journal entry: %s", exc)
            reports.append(ValidateReport(phase="policy", check="evidence", ok=True))
            return
        if latest is None:
            reports.append(ValidateReport(phase="policy", check="evidence", ok=True))
            return

        cited_raw = latest.frontmatter.get("cited_candidates")
        cited: set[str] = set()
        if isinstance(cited_raw, list):
            cited = {str(c) for c in cited_raw}
        elif isinstance(cited_raw, str):
            cited = {cited_raw}

        if not cited:
            _fail(
                "# Evidence Gate Failure\n\n"
                f"`{candidates_path}` contains candidates "
                f"{candidate_ids} but the latest journal entry's "
                "`cited_candidates` frontmatter is missing or empty.\n\n"
                "Add a `cited_candidates` line to your journal entry's "
                "frontmatter listing the IDs of the candidate(s) that "
                "motivated this round's config change:\n\n"
                "```\ncited_candidates: [C-001, C-002]\n```\n",
                "journal's latest entry has no `cited_candidates`",
            )

        missing = cited - set(candidate_ids)
        if missing:
            _fail(
                "# Evidence Gate Failure\n\n"
                f"The journal's latest entry cites candidates "
                f"{sorted(missing)} but `{candidates_path}` does not contain "
                f"sections for those IDs. Candidates found in the file: "
                f"{candidate_ids}.\n\n"
                "Either add the missing candidate sections, or correct the "
                "`cited_candidates` list to reference only IDs that exist.\n",
                f"cited_candidates {sorted(missing)} not present in {candidates_path}",
            )

        reports.append(ValidateReport(phase="policy", check="evidence", ok=True))

    # ----- Phase 3: advisory (never blocks) ----------------------------------

    async def _literals_advisory(
        self,
        output_dir: Path,
        scratch_dir: Path,
        reports: list[ValidateReport],
    ) -> None:
        try:
            report = await run_literals_check(output_dir, scratch_dir, strict=False)
        except Exception as exc:  # noqa: BLE001 — advisory must never block
            logger.warning("[validate] literals advisory skipped: %s", exc)
            reports.append(
                ValidateReport(
                    phase="advisory",
                    check="literals",
                    ok=True,
                    reason=f"advisory skipped: {exc}",
                )
            )
            return

        findings = int(report.get("findings", 0))
        artifact = report.get("artifact")
        if findings > 0:
            warn_path = scratch_dir / "LITERALS_WARNING.md"
            warn_path.write_text(
                "# Literals advisory — non-blocking\n\n"
                f"The literals scan found **{findings}** task-specific "
                "literal match(es) in authored tools/processors. This does "
                "not fail the round, but such components rarely survive the "
                "next benchmark run — they only work on the one task the "
                "author memorised.\n\n"
                f"Full findings: `{artifact}`\n\n"
                "Either generalise the offending code (parameterise the "
                "literal, derive it from task input), or note in the "
                "next round's journal why the literal is intentional.\n",
                encoding="utf-8",
            )
            logger.info("[validate] literals advisory: %d finding(s); see %s", findings, warn_path)
        reports.append(
            ValidateReport(
                phase="advisory",
                check="literals",
                ok=True,
                reason=f"{findings} finding(s)",
                findings_path=Path(artifact) if artifact else None,
            )
        )


# ─── CLI dispatcher ──────────────────────────────────────────────────────

_CLI_USAGE = (
    "usage: python -m harnessx.meta_harness.validate_workflow <subcommand> [args]\n"
    "  canonicalize <config.yaml>\n"
    "  dry_fire     <config.yaml> <scratch_dir>\n"
    "  contract     <config.yaml> <scratch_dir>\n"
    "  literals     <output_dir>  <scratch_dir>"
)


def _cli_err(msg: str) -> int:
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    return 2


def _cli_canonicalize(argv: list[str]) -> int:
    if len(argv) != 1:
        return _cli_err("usage: validate_workflow canonicalize <config.yaml>")
    result = check_canonicalize(Path(argv[0]))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def _cli_dry_fire(argv: list[str]) -> int:
    if len(argv) != 2:
        return _cli_err("usage: validate_workflow dry_fire <config.yaml> <scratch_dir>")
    yaml_path = Path(argv[0])
    scratch = Path(argv[1])
    scratch.mkdir(parents=True, exist_ok=True)

    from ..core.harness import HarnessConfig

    try:
        cfg = HarnessConfig.from_yaml_file(yaml_path).canonicalize()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"config failed canonicalize: {type(exc).__name__}: {exc}"}))
        return 1

    async def _run() -> dict:
        proc_report = await run_processor_dry_fire(cfg, scratch, strict=False)
        tool_report = await run_tool_dry_fire(cfg, scratch, strict=False)
        total_likely = proc_report["likely_bugs"] + tool_report["likely_bugs"]
        return {
            "ok": total_likely == 0,
            "processors": proc_report,
            "tools": tool_report,
        }

    result = asyncio.run(_run())
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


def _cli_contract(argv: list[str]) -> int:
    if len(argv) != 2:
        return _cli_err("usage: validate_workflow contract <config.yaml> <scratch_dir>")
    yaml_path = Path(argv[0])
    scratch = Path(argv[1])
    scratch.mkdir(parents=True, exist_ok=True)

    from ..core.harness import HarnessConfig

    try:
        cfg = HarnessConfig.from_yaml_file(yaml_path).canonicalize()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"config failed canonicalize: {type(exc).__name__}: {exc}"}))
        return 1

    report = asyncio.run(run_contract_check(cfg, scratch, strict=False))
    result = {"ok": report["violations"] == 0, **report}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


def _cli_literals(argv: list[str]) -> int:
    if len(argv) != 2:
        return _cli_err("usage: validate_workflow literals <output_dir> <scratch_dir>")
    output_dir = Path(argv[0])
    scratch = Path(argv[1])
    scratch.mkdir(parents=True, exist_ok=True)

    report = asyncio.run(run_literals_check(output_dir, scratch, strict=False))
    ok = report["findings"] < report["threshold"]
    result = {"ok": ok, **report}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if ok else 1


_CLI_DISPATCH = {
    "canonicalize": _cli_canonicalize,
    "dry_fire": _cli_dry_fire,
    "contract": _cli_contract,
    "literals": _cli_literals,
}


def _main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_CLI_USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    handler = _CLI_DISPATCH.get(cmd)
    if handler is None:
        print(
            json.dumps(
                {"ok": False, "error": f"unknown subcommand {cmd!r}. {_CLI_USAGE}"},
                ensure_ascii=False,
            )
        )
        return 2
    return handler(rest)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))


__all__ = [
    "StrictValidationError",
    "ValidateReport",
    "ValidateOutcome",
    "EvolveValidator",
    "check_canonicalize",
    "run_processor_dry_fire",
    "run_tool_dry_fire",
    "run_contract_check",
    "run_literals_check",
    "check_novelty",
]
