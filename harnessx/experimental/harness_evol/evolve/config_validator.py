"""
Programmatic validator for evolved harness configs.

validate_harness_config(path, baseline_config_path=None) → str | None
  None  = passed
  str   = error description (shown to EvolveAgent for fixing)

Seven checks in order:
  1. canonicalize    — YAML parses, HarnessConfig schema valid, Jinja templates OK.
  2. tool_imports    — All tool_registry.custom[] entries are importable Python callables.
  3. leakage         — New processor source files must not branch on task_description
                       content via keyword/regex matching (benchmark overfitting guard).
  4. no_evol_leakage — Config must not reference harness_evol.* processors (evolve-
                       pipeline internals must not leak into the target-agent config).
  5. dry_fire        — All 8 hooks fired on every custom processor with dummy events;
                       catches field-name typos and constructor shape drift.
  6. contract        — Hook contract checks on custom processors (message-mutating
                       yield semantics, required fields, etc.).
  7. param_diff      — (only when baseline_config_path provided) Any processor param
                       that changed vs baseline is reported so the agent can declare it
                       as param_change in the manifest before submitting.

All checks use harnessx.core.* only — no dependency on harnessx.meta_harness.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── 1. canonicalize ────────────────────────────────────────────────────────────

def _check_canonicalize(path: Path) -> dict:
    """Run canonicalize + eager template check. Returns {"ok": True} or {"ok": False, "error": ...}."""
    from harnessx.core.harness import HarnessConfig

    if not path.is_file():
        return {"ok": False, "error": f"config not found: {path}"}

    try:
        cfg = HarnessConfig.from_yaml_file(path).canonicalize()
        _eager_check_system_prompt_builders(cfg)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    return {"ok": True}


def _eager_check_system_prompt_builders(cfg: Any) -> None:
    """Force any lazy template load to happen now; raises ValueError on defect."""
    rt_procs = getattr(cfg, "_rt_procs", None) or []

    try:
        from harnessx.processors.context.strategies.system_prompt.template import (
            TemplateSystemPromptBuilder,
        )
        from harnessx.processors.context.system_prompt import SystemPromptProcessor
    except ImportError:
        return

    for p in rt_procs:
        if not isinstance(p, SystemPromptProcessor):
            continue
        sb = getattr(p, "system_builder", None)
        if sb is None or not isinstance(sb, TemplateSystemPromptBuilder):
            continue
        tpath = getattr(sb, "template_path", None)
        if not tpath:
            raise ValueError("TemplateSystemPromptBuilder has no template_path set")
        try:
            with open(tpath, "r", encoding="utf-8") as f:
                src = f.read()
        except FileNotFoundError as exc:
            raise ValueError(
                f"SystemPromptProcessor.system_builder.template_path points to a "
                f"file that does not exist: {tpath!r}"
            ) from exc
        except OSError as exc:
            raise ValueError(f"Cannot read template_path {tpath!r}: {exc}") from exc

        if not src.strip():
            raise ValueError(f"template_path {tpath!r} is empty.")

        try:
            from jinja2 import Template, TemplateSyntaxError
            try:
                Template(src)
            except TemplateSyntaxError as exc:
                raise ValueError(
                    f"template_path {tpath!r} has invalid Jinja syntax: "
                    f"{exc.message} (line {exc.lineno})"
                ) from exc
        except ImportError:
            pass


# ── 2. tool_imports ────────────────────────────────────────────────────────────

def _check_tool_imports(path: Path) -> list[str]:
    """Verify all tool_registry.custom[] entries are importable Python callables."""
    import importlib
    import yaml

    errors: list[str] = []
    try:
        cfg_dict = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return [f"YAML parse error: {e}"]

    custom_tools = (cfg_dict.get("tool_registry") or {}).get("custom") or []
    for import_path in custom_tools:
        if not isinstance(import_path, str) or "." not in import_path:
            errors.append(f"tool_registry.custom: invalid entry {import_path!r} (expected 'module.function')")
            continue
        module_path, _, attr = import_path.rpartition(".")
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            errors.append(f"tool_registry.custom: cannot import module '{module_path}': {e}")
            continue
        if not hasattr(mod, attr):
            errors.append(
                f"tool_registry.custom: '{module_path}' has no attribute '{attr}' "
                f"(full path: {import_path!r})"
            )
    return errors


# ── 3. leakage ─────────────────────────────────────────────────────────────────

# Structural leakage pattern: reads task_description + does pattern matching on it.
# This is how a processor hard-codes benchmark task knowledge — the exact fingerprint
# of domain_hint_processor.py style overfitting.
_RE_TASK_DESC_ACCESS = re.compile(
    r"\btask_description\b|\bevent\.task_description\b|\btask\.description\b",
)
_RE_KEYWORD_MATCH = re.compile(
    r"\.search\s*\(|\.match\s*\(|\.findall\s*\(|\.fullmatch\s*\("
    r"|\bin\s+task_desc|\bif\s+[^\n]*task_desc",
)


def _resolve_processor_source_files(path: Path) -> list[Path]:
    """
    Return filesystem paths for every non-harnessx, non-benchmarks processor
    referenced in the config YAML.

    Handles two _target_ formats:
      - file:///abs/path/to/file.py::ClassName   (direct file reference)
      - some.module.path.ClassName               (Python module; resolved via importlib)
    """
    import importlib.util
    import yaml

    try:
        cfg_dict = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []

    processors = cfg_dict.get("processors") or []
    if not isinstance(processors, list):
        return []

    found: list[Path] = []
    seen: set[str] = set()

    for entry in processors:
        if not isinstance(entry, dict):
            continue
        target = entry.get("_target_", "") or ""

        # ── file:// absolute path format ──────────────────────────────────────
        if target.startswith("file://"):
            # Strip scheme and split off ::ClassName suffix
            file_part = target[len("file://"):]
            if "::" in file_part:
                file_part = file_part.split("::")[0]
            src = Path(file_part)
            key = str(src)
            if key not in seen and src.is_file():
                seen.add(key)
                found.append(src)
            continue

        # ── Python module path — skip harnessx.* and benchmarks.* ─────────────
        if (
            not target
            or target.startswith("harnessx.")
            or target.startswith("benchmarks.")
        ):
            continue

        module_path = target.rsplit(".", 1)[0] if "." in target else target
        if module_path in seen:
            continue
        seen.add(module_path)

        try:
            spec = importlib.util.find_spec(module_path)
        except (ModuleNotFoundError, ValueError):
            continue
        if spec and spec.origin:
            found.append(Path(spec.origin))

    return found


def _check_leakage(path: Path) -> list[str]:
    """
    Scan new processor source files for task-description leakage.

    Leakage = the processor reads task_description content and does keyword/regex
    matching on it to inject task-specific hints.  This is benchmark overfitting:
    the processor has memorised which TB2 tasks need which tricks instead of
    improving the model's general reasoning capability.

    Rule: a processor file that BOTH accesses task_description AND calls a
    regex/string-search method on it is flagged as leakage.
    """
    errors: list[str] = []

    for src_path in _resolve_processor_source_files(path):
        try:
            source = src_path.read_text(encoding="utf-8")
        except OSError:
            continue

        has_task_desc = bool(_RE_TASK_DESC_ACCESS.search(source))
        has_keyword_match = bool(_RE_KEYWORD_MATCH.search(source))

        if has_task_desc and has_keyword_match:
            errors.append(
                f"LEAKAGE [{src_path.name}]: Processor reads task_description "
                f"and performs keyword/regex matching on it. "
                f"This is benchmark overfitting — the processor memorises which "
                f"tasks need which tricks rather than improving general capability. "
                f"Processors MUST NOT branch on task_description content. "
                f"Fix: use a generic system_prompt injection (param_change on an "
                f"existing prompt field) that applies to ALL tasks equally."
            )

    return errors


# ── 5. dry_fire ────────────────────────────────────────────────────────────────

async def _run_dry_fire(cfg: Any) -> list[str]:
    """Fire all 8 hooks on every custom processor with dummy events. Returns list of bug strings."""
    from harnessx.core.events import (
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
                        f"{cls.__name__}.{hook_name}: AttributeError: {msg} "
                        f"(likely a field-name typo; read harnessx/core/events.py)"
                    )
            except TypeError as exc:
                msg = str(exc)
                if "unexpected keyword argument" in msg:
                    likely_bugs.append(
                        f"{cls.__name__}.{hook_name}: TypeError: {msg} "
                        f"(constructing event with stale field; check dataclass definition)"
                    )
            except Exception:  # noqa: BLE001
                pass

    return likely_bugs


# ── 6. contract ────────────────────────────────────────────────────────────────

async def _run_contract_check(cfg: Any) -> list[str]:
    """Check hook yield-contract on custom processors. Returns list of violation strings."""
    from harnessx.core.contract_check import check_processor_contract
    from harnessx.core.harness import _instantiate_proc

    violations: list[str] = []
    seen: set[tuple[str, str]] = set()

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
            except Exception:  # noqa: BLE001
                pass

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
            findings = await check_processor_contract(p)
            for v in findings:
                violations.append(
                    f"{v.processor}.{v.hook} [{v.violation_type}] fixture={v.fixture}: {v.message}"
                )
        except Exception:  # noqa: BLE001
            pass

    return violations


# ── 7. param_diff ──────────────────────────────────────────────────────────────

def _check_param_diff(new: Path, baseline: Path) -> list[str]:
    """Compare processor params between new config and baseline.

    Reports every param that changed (added, removed, or modified) so the agent
    can declare it as a param_change in the change manifest before submitting.
    Does NOT require the manifest — it just surfaces the raw diff so the agent
    can act on it immediately within the same session.
    """
    import yaml

    _IGNORED_KEYS = {"_target_", "_code_hash"}

    try:
        base_cfg = yaml.safe_load(baseline.read_text(encoding="utf-8")) or {}
        new_cfg = yaml.safe_load(new.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return [f"Could not parse configs for param diff: {e}"]

    def _index(processors: list) -> dict[str, dict]:
        idx: dict[str, dict] = {}
        for p in (processors or []):
            if isinstance(p, dict):
                t = p.get("_target_")
                if t:
                    idx[t] = {k: v for k, v in p.items() if k not in _IGNORED_KEYS}
        return idx

    base_idx = _index(base_cfg.get("processors", []))
    new_idx = _index(new_cfg.get("processors", []))

    diffs: list[str] = []
    for target, base_params in base_idx.items():
        if target not in new_idx:
            continue  # removed processor — separate concern
        new_params = new_idx[target]
        if base_params == new_params:
            continue
        changed_keys = [
            k for k in set(base_params) | set(new_params)
            if base_params.get(k) != new_params.get(k)
        ]
        for k in changed_keys:
            diffs.append(
                f"  {target}: '{k}' changed "
                f"{base_params.get(k)!r} → {new_params.get(k)!r}"
            )

    return diffs


# ── public API ─────────────────────────────────────────────────────────────────

async def validate_harness_config(path: Path, baseline_config_path: Path | None = None) -> str | None:
    """
    Validate an evolved harness config file.

    Returns None on success, or a human-readable error string on failure.
    The error is injected into the agent's context so it can fix the config.
    """
    # ── 1. canonicalize ───────────────────────────────────────────────────────
    result = _check_canonicalize(path)
    if not result.get("ok"):
        return f"Config validation failed (canonicalize):\n  {result.get('error', 'unknown error')}"

    # ── 2. tool_imports ───────────────────────────────────────────────────────
    tool_errors = _check_tool_imports(path)
    if tool_errors:
        return "Tool import validation failed:\n" + "\n".join(f"  - {e}" for e in tool_errors)

    # ── 3. leakage ────────────────────────────────────────────────────────────
    leakage_errors = _check_leakage(path)
    if leakage_errors:
        return (
            "Leakage check failed — config rejected to prevent benchmark overfitting:\n"
            + "\n".join(f"  - {e}" for e in leakage_errors)
        )

    # ── 4. no_evol_leakage ───────────────────────────────────────────────────
    _EVOL_PREFIX = "harnessx.experimental.harness_evol."
    try:
        import yaml as _yaml
        _cfg_text = path.read_text(encoding="utf-8")
        _leaked = [
            line.strip()
            for line in _cfg_text.splitlines()
            if _EVOL_PREFIX in line and "_target_:" in line
        ]
        if _leaked:
            return (
                "Config rejected — evolve-pipeline processor(s) must not appear in the "
                "target-agent config (harness_evol.* is for the evolution harness only):\n"
                + "\n".join(f"  - {l}" for l in _leaked[:5])
            )
    except Exception:
        pass

    # ── 5. dry_fire + 6. contract ─────────────────────────────────────────────
    try:
        from harnessx.core.harness import HarnessConfig
        cfg = HarnessConfig.from_yaml_file(path)
    except Exception as e:
        return f"Config load failed: {e}"

    errors: list[str] = []

    try:
        bugs = await _run_dry_fire(cfg)
        if bugs:
            errors.append("Dry-fire likely bugs:\n" + "\n".join(f"  - {b}" for b in bugs))
    except Exception as e:
        errors.append(f"dry_fire raised: {e}")

    try:
        viols = await _run_contract_check(cfg)
        if viols:
            errors.append("Contract violations:\n" + "\n".join(f"  - {v}" for v in viols))
    except Exception as e:
        errors.append(f"contract check raised: {e}")

    if errors:
        return "Processor checks failed:\n\n" + "\n\n".join(errors)

    # ── 7. param_diff ─────────────────────────────────────────────────────────
    if baseline_config_path is not None and baseline_config_path.is_file():
        diffs = _check_param_diff(path, baseline_config_path)
        if diffs:
            return (
                "PARAM DIFF DETECTED — processor params changed vs baseline.\n"
                "You MUST declare each change as a param_change entry in the manifest "
                "(with target, param_name, old_value, new_value, and all 4 evidence fields) "
                "before calling submit_change_manifest, "
                "OR revert the param to its baseline value if the change was unintentional.\n"
                "Changed params:\n" + "\n".join(diffs)
            )

    return None
