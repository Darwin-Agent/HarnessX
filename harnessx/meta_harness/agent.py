# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Meta-agent: trajectory analysis in, evolved HarnessConfig out.

Public surface:

- ``MetaAgent`` — class-based API. Instantiate once per benchmark run,
  call ``await agent.evolve(...)`` per round.
- ``build_meta_agent_harness_config(...)`` — low-level factory exposed
  for tests and any caller that wants a raw ``HarnessConfig`` without
  the orchestration layer.
- ``compute_changeset(before, after)`` — shallow diff of two canonical
  HarnessConfigs (tools / processors / processor kwargs / templates).
  Consumed by the orchestrator to populate journal frontmatter and the
  CONTEXT.md changeset ribbon.

Business logic this module owns:

1. Building the meta-agent ``HarnessConfig`` (tools, processors,
   persona, sandbox).
2. Writing the per-run brief (``_meta_scratch/TASK.md``) and the
   journal-derived context (``_meta_scratch/CONTEXT.md``).
3. Running one agent turn with a wall-clock deadline.
4. Invoking the post-flight workflow (see ``validate_workflow.py``).

Everything else stays in its own module: ``journal.py`` owns cross-round
memory + attribution, ``replay.py`` owns the synthetic-task smoke gate,
``validate_workflow.py`` bundles the two-phase post-flight workflow
(``EvolveValidator``) and the agent-facing self-check CLIs,
``processors/`` are HarnessConfig plugins, ``workers/`` is the
reflect-worker spawn tool.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Post-flight lives in ``validate_workflow.EvolveValidator`` — imported
# lazily inside ``evolve()`` so callers that only want
# ``build_meta_agent_harness_config`` / ``compute_changeset`` don't pay
# the module-load cost.

if TYPE_CHECKING:
    from ..core.harness import HarnessConfig
    from ..core.model_config import ModelConfig

logger = logging.getLogger(__name__)


_META_WORKSPACE_ROOT = Path(__file__).parent / "workspace"

_DEFAULT_MAX_COST_USD = 5.0
_DEFAULT_WALL_CLOCK_S = 900.0
_DEFAULT_MAX_STEPS = 500


# ---------------------------------------------------------------------------
# Sandbox provider — keeps absolute paths usable while defaulting Bash cwd
# to output_dir so relative commands don't pollute filesystem root.
# ---------------------------------------------------------------------------


class _MetaAgentSandboxProvider:
    def __init__(self, output_dir: Path | None = None) -> None:
        self._cwd = str(Path(output_dir).expanduser().resolve()) if output_dir is not None else None

    async def acquire(self, hint_id=None, workspace=None):  # noqa: ANN001
        from ..sandbox.local import LocalSandbox

        default_cwd = self._cwd

        class _MetaSandbox(LocalSandbox):
            async def exec(self, command, *args, **kwargs):  # noqa: ANN001
                if "cwd" not in kwargs and not args:
                    kwargs["cwd"] = default_cwd
                return await LocalSandbox.exec(self, command, *args, **kwargs)

        return _MetaSandbox(root="/", mode="isolated")

    async def release(self, sandbox) -> None:  # noqa: ANN001
        pass

    async def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# HarnessConfig factory — low-level, stateless.
# ---------------------------------------------------------------------------


def build_meta_agent_harness_config(
    *,
    inner_model: "ModelConfig",
    tracer: Any = None,
    workspace_root: Path | None = None,
    persona_root: Path | None = None,
    extra_skills_dirs: "list[str | Path] | None" = None,
    max_cost_usd: "float | None" = _DEFAULT_MAX_COST_USD,
    loop_detection: bool = True,
    output_dir: Path | None = None,
    allowed_write_roots: tuple[Path, ...] = (),
    allowed_write_files: tuple[Path, ...] = (),
    step_deadline_reminder_step: int | None = 260,
    step_deadline_early_reminder_step: "int | None" = None,
    step_deadline_output_within: int = 30,
    tool_result_noise_filter: bool = False,
    read_scope_blocked_roots: "tuple[str | Path, ...] | None" = None,
    read_scope_allowed_files: "tuple[str | Path, ...] | None" = None,
) -> "HarnessConfig":
    """Assemble the meta-agent's HarnessConfig.

    Tools: Read / Write / Edit / Glob / Grep / Bash / WebSearch / WebFetch /
    spawn_reflect_worker.

    Processors (in registration order): SystemPromptProcessor +
    WriteScopeGate + StepDeadlineReminder + CostGuardProcessor +
    LoopDetectionProcessor + ToolResultNoiseFilter + CompactionProcessor.
    """
    from ..core.builder import HarnessBuilder
    from ..processors.context.strategies.system_prompt.default import (
        DefaultSystemPromptBuilder,
    )
    from ..processors.context.system_prompt import SystemPromptProcessor
    from ..processors.control.compaction import CompactionProcessor
    from ..processors.control.cost_guard import CostGuardProcessor
    from ..processors.control.loop_detection import LoopDetectionProcessor
    from ..tools.builtin import (
        bash_tool,
        edit_tool,
        glob_tool,
        grep_tool,
        read_tool,
        web_fetch_tool,
        web_search_tool,
        write_tool,
    )
    from ..tools.inmemory import InMemoryToolRegistry
    from ..workspace.workspace import Workspace
    from .processors import (
        ContractAutoCheckProcessor,
        StepDeadlineReminderProcessor,
        WriteScopeGateProcessor,
    )
    from .workers import make_spawn_reflect_worker_tool

    tool_reg = InMemoryToolRegistry()
    for t in (
        read_tool,
        write_tool,
        edit_tool,
        glob_tool,
        grep_tool,
        bash_tool,
        web_search_tool,
        web_fetch_tool,
    ):
        tool_reg.register(t)

    resolved_workspace_root = Path(workspace_root) if workspace_root is not None else _META_WORKSPACE_ROOT
    resolved_persona_root = Path(persona_root) if persona_root is not None else resolved_workspace_root
    ws = Workspace(
        root=resolved_workspace_root,
        agent_id="meta-agent",
        mode="shared",
    )

    slot_kwargs: dict[str, Any] = dict(
        tool_registry=tool_reg,
        workspace=ws,
        init_workspace=False,
        sandbox_provider=_MetaAgentSandboxProvider(output_dir=output_dir),
    )
    if tracer is not None:
        slot_kwargs["tracer"] = tracer

    builder = (
        HarnessBuilder()
        .slot(**slot_kwargs)
        .add(
            SystemPromptProcessor(
                DefaultSystemPromptBuilder(
                    max_skills_shown=20,
                    persona_root=resolved_persona_root,
                    extra_skills_dirs=list(extra_skills_dirs) if extra_skills_dirs else None,
                )
            )
        )
        .add(
            WriteScopeGateProcessor(
                allowed_roots=tuple(str(Path(p).resolve()) for p in allowed_write_roots),
                allowed_files=tuple(str(Path(p).resolve()) for p in allowed_write_files),
            )
        )
    )
    if step_deadline_reminder_step is not None:
        builder = builder.add(
            StepDeadlineReminderProcessor(
                reminder_step=step_deadline_reminder_step,
                early_reminder_step=step_deadline_early_reminder_step,
                output_within_steps=step_deadline_output_within,
            )
        )
    if loop_detection:
        builder = builder.add(LoopDetectionProcessor())

    if max_cost_usd is not None:
        builder = builder.add(CostGuardProcessor(max_usd=max_cost_usd))

    if tool_result_noise_filter:
        from .processors.tool_result_noise_filter import ToolResultNoiseFilterProcessor

        builder = builder.add(ToolResultNoiseFilterProcessor())

    builder = builder.add(ContractAutoCheckProcessor())

    if output_dir is not None:
        from .processors.leakage_guard import LeakageGuardProcessor

        builder = builder.add(LeakageGuardProcessor(output_dir=output_dir))

    if read_scope_blocked_roots is not None:
        from .processors.read_scope_gate import ReadScopeGateProcessor

        builder = builder.add(
            ReadScopeGateProcessor(
                blocked_roots=tuple(str(r) for r in read_scope_blocked_roots),
                allowed_files=tuple(str(f) for f in (read_scope_allowed_files or ())),
                hint_message=(
                    "harnessx source is gated. "
                    "Consult the SKILL.md files in your workspace/skills/ for the "
                    "MultiHookProcessor API and hook signatures. "
                    "Only harnessx/core/processor.py is accessible for quick reference."
                ),
            )
        )

    cfg = builder.add(
        CompactionProcessor(
            token_threshold=200000,
            retention_window=4,
            eviction_fraction=0.95,
            summarize_prompt_template=(
                "You are compacting older conversation context for a meta-agent that reads\n"
                "task trajectories and evolves a HarnessConfig (processor pipeline, prompts,\n"
                "tools, verification logic).\n"
                "Return concise Markdown with exactly these sections:\n"
                "1) Decisions\n"
                "2) Facts and Constraints\n"
                "3) Errors and Unresolved Risks\n"
                "4) Pending Actions\n\n"
                "Preserve: failure root causes, chosen hypothesis, config changes made or planned,\n"
                "file paths written, hard constraints, and open blockers.\n"
                "Discard: install/download progress logs (apt/pip/npm) unless they contain errors,\n"
                "repeated status lines, file listings with no decisions attached, and any filler.\n"
                "Be concise but complete — do not omit any decision, blocker, or config change.\n\n"
                "Conversation to summarize:\n{conversation}"
            ),
        )
    ).build()

    # Register the reflect-worker spawn tool AFTER the inner config is built
    # so its child_config_fn closes over the parent HarnessConfig.
    spawn_tool = make_spawn_reflect_worker_tool(
        inner_model=inner_model,
        parent_harness_config=cfg,
    )
    cfg.tool_registry.register(spawn_tool)
    return cfg


# ---------------------------------------------------------------------------
# Changeset — shallow diff of two canonical HarnessConfigs.
# Previously lived in changeset.py; folded in because agent.py is the only
# non-test caller and keeping them together makes the evolve loop easier to
# follow.
# ---------------------------------------------------------------------------


def _tool_label_from_custom_target(target: str) -> str:
    """Extract the tool *symbol name* from a ToolRegistryConfig.custom entry.

    Supported forms (mirrors ``_build_tool_registry_from_config``):

    * ``file:///abs/path.py::symbol`` → ``symbol``
    * ``module.path.symbol``          → ``symbol``

    The tool's *registered* name can differ from its symbol name (via
    ``@tool(name=...)``), but at changeset-diff time we only know the
    import target, so we use the symbol. Different symbols ⇒ different
    labels ⇒ still detected as added/removed. Same symbol re-pointed at
    a different path is rare and would require loading the module to
    disambiguate, which is out of scope for a best-effort diff.
    """
    if not isinstance(target, str) or not target:
        return ""
    if target.startswith("file://"):
        # file:///abs/path.py::symbol → symbol
        _, sep, sym = target.rpartition("::")
        return sym.strip() if sep else ""
    # dotted path: module.path.symbol → symbol
    return target.rsplit(".", 1)[-1]


def _tool_names(cfg: "HarnessConfig") -> set[str]:
    """Extract the set of tool names present in a HarnessConfig.

    Handles three shapes the ``tool_registry`` slot can take:

    * live registry (``InMemoryToolRegistry``) — uses ``list_names()``.
    * :class:`~harnessx.core.config_schema.ToolRegistryConfig` dataclass —
      yields ``set(builtin) | {symbol for each custom target}``. This is
      the shape produced by :meth:`HarnessConfig.from_yaml`, which is
      what :func:`compute_changeset` sees in practice when diffing two
      YAML-round-tripped configs.
    * ``None`` or anything else — empty set.

    Previously this function only handled the live-registry shape, so
    YAML-loaded configs appeared to carry *zero* tools and tool
    additions/removals silently disappeared from the changeset diff.
    """
    reg = getattr(cfg, "tool_registry", None)
    if reg is None:
        return set()

    # Live registry path (InMemoryToolRegistry and friends).
    if hasattr(reg, "list_names"):
        try:
            return set(reg.list_names())
        except Exception:  # noqa: BLE001 — diff is best-effort
            return set()

    # Declarative path (ToolRegistryConfig). Import lazily so this
    # module stays importable without harnessx.core initialised.
    try:
        from harnessx.core.config_schema import ToolRegistryConfig
    except Exception:  # noqa: BLE001
        return set()

    if isinstance(reg, ToolRegistryConfig):
        names: set[str] = set()
        for name in getattr(reg, "builtin", []) or []:
            if isinstance(name, str) and name:
                names.add(name)
        for target in getattr(reg, "custom", []) or []:
            label = _tool_label_from_custom_target(target)
            if label:
                names.add(label)
        return names

    return set()


def _iter_processor_entries(cfg: "HarnessConfig"):
    for entry in getattr(cfg, "processors", None) or []:
        if isinstance(entry, dict):
            target = entry.get("_target_", "") or ""
            yield _label_from_target(target), entry
    for p in getattr(cfg, "_rt_procs", None) or []:
        label = getattr(p, "_singleton_group", "") or type(p).__name__
        entry = {"_target_": type(p).__module__ + "." + type(p).__name__}
        yield label, entry


def _label_from_target(target: str) -> str:
    if "::" in target:
        return target.rsplit("::", 1)[-1]
    return target.rsplit(".", 1)[-1] if target else ""


def _processor_labels(cfg: "HarnessConfig") -> set[str]:
    return {label for label, _ in _iter_processor_entries(cfg) if label}


def _processor_kwargs_fingerprint(entry: dict) -> str:
    kwargs = {k: v for k, v in entry.items() if k not in ("_target_", "_code_hash")}
    try:
        return json.dumps(kwargs, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return repr(sorted(kwargs.items()))


def _processor_kwargs_by_label(cfg: "HarnessConfig") -> dict[str, str]:
    out: dict[str, str] = {}
    for label, entry in _iter_processor_entries(cfg):
        if not label:
            continue
        out[label] = _processor_kwargs_fingerprint(entry)
    return out


def _collect_template_paths(cfg: "HarnessConfig") -> set[str]:
    paths: set[str] = set()

    def _visit(node: Any) -> None:
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


def _template_fingerprints(cfg: "HarnessConfig") -> dict[str, str]:
    fps: dict[str, str] = {}
    for tpath in _collect_template_paths(cfg):
        tp = Path(tpath)
        if not tp.is_file():
            fps[str(tp)] = "<missing>"
            continue
        try:
            data = tp.read_bytes()
        except OSError:
            fps[str(tp)] = "<unreadable>"
            continue
        fps[str(tp)] = hashlib.sha256(data).hexdigest()[:16]
    return fps


def compute_changeset(
    before: "HarnessConfig",
    after: "HarnessConfig",
) -> dict[str, Any]:
    """Return a stable plain-dict diff of two canonical HarnessConfigs.

    Buckets (all optional, omitted when empty):

    - ``tools_added`` / ``tools_removed``
    - ``processors_added`` / ``processors_removed``
    - ``processors_config_changed`` — labels whose kwargs differ
    - ``templates_added`` / ``templates_removed`` / ``templates_changed``
    """
    b_tools = _tool_names(before)
    a_tools = _tool_names(after)
    b_procs = _processor_labels(before)
    a_procs = _processor_labels(after)
    b_proc_kwargs = _processor_kwargs_by_label(before)
    a_proc_kwargs = _processor_kwargs_by_label(after)
    b_tpl = _template_fingerprints(before)
    a_tpl = _template_fingerprints(after)

    tpl_added = sorted(set(a_tpl) - set(b_tpl))
    tpl_removed = sorted(set(b_tpl) - set(a_tpl))
    tpl_changed = sorted(p for p in set(a_tpl) & set(b_tpl) if a_tpl[p] != b_tpl[p])

    shared_labels = set(b_proc_kwargs) & set(a_proc_kwargs)
    procs_cfg_changed = sorted(lbl for lbl in shared_labels if b_proc_kwargs[lbl] != a_proc_kwargs[lbl])

    diff: dict[str, Any] = {}
    if a_tools - b_tools:
        diff["tools_added"] = sorted(a_tools - b_tools)
    if b_tools - a_tools:
        diff["tools_removed"] = sorted(b_tools - a_tools)
    if a_procs - b_procs:
        diff["processors_added"] = sorted(a_procs - b_procs)
    if b_procs - a_procs:
        diff["processors_removed"] = sorted(b_procs - a_procs)
    if procs_cfg_changed:
        diff["processors_config_changed"] = procs_cfg_changed
    if tpl_added:
        diff["templates_added"] = tpl_added
    if tpl_removed:
        diff["templates_removed"] = tpl_removed
    if tpl_changed:
        diff["templates_changed"] = tpl_changed
    return diff


# ---------------------------------------------------------------------------
# MetaAgent — the main public class.
# ---------------------------------------------------------------------------


class MetaAgent:
    """The meta-agent: reads trajectories, ships evolved HarnessConfigs.

    Instantiate once per benchmark run; call ``await agent.evolve(...)``
    per round. Per-instance fields hold values that don't change across
    rounds (model, memo path, budgets, skill dirs); the ``evolve``
    method takes the per-round things (current_config, trajectories_dir,
    output_dir, ...).

    The HarnessConfig is rebuilt per evolve() call — the sandbox
    provider needs ``output_dir`` as its default Bash cwd, and
    ``output_dir`` changes per round. Rebuilding is cheap (~ms).
    """

    def __init__(
        self,
        *,
        inner_model: "ModelConfig",
        memo_path: Path | None = None,
        extra_skills_dirs: "list[str | Path] | None" = None,
        max_cost_usd: "float | None" = _DEFAULT_MAX_COST_USD,
        wall_clock_s: float = _DEFAULT_WALL_CLOCK_S,
        max_steps: int = _DEFAULT_MAX_STEPS,
        allowed_write_roots: tuple[Path, ...] = (),
        allowed_write_files: tuple[Path, ...] = (),
        step_deadline_reminder_step: int | None = 260,
        step_deadline_early_reminder_step: "int | None" = None,
        step_deadline_output_within: int = 30,
        context_recent_window: int = 5,
        extra_harness_kws: "dict | None" = None,
    ) -> None:
        self.inner_model = inner_model
        self.memo_path = Path(memo_path).resolve() if memo_path else None
        self.extra_skills_dirs = list(extra_skills_dirs) if extra_skills_dirs else None
        self.max_cost_usd = float(max_cost_usd) if max_cost_usd is not None else None
        self.wall_clock_s = float(wall_clock_s)
        self.max_steps = int(max_steps)
        self.allowed_write_roots = tuple(allowed_write_roots)
        self.allowed_write_files = tuple(allowed_write_files)
        self.step_deadline_reminder_step = step_deadline_reminder_step
        self.step_deadline_early_reminder_step = step_deadline_early_reminder_step
        self.step_deadline_output_within = int(step_deadline_output_within)
        self.context_recent_window = int(context_recent_window)
        self.extra_harness_kws: dict = extra_harness_kws or {}

    async def evolve(
        self,
        *,
        current_config: "HarnessConfig | Path",
        trajectories_dir: Path,
        output_dir: Path,
        replay_model: "ModelConfig | None" = None,
        replay_max_cost_usd: float | None = 0.5,
        replay_timeout_s: float = 300.0,
        replay_mode: str = "synthetic_task",
    ) -> Path:
        """Run one meta-agent pass. Returns path to ``output_dir/config.yaml``.

        Post-flight checks: canonicalize → novelty → evidence →
        changeset → replay smoke gate.
        """
        from ..core.harness import BaseTask

        trajectories_dir = Path(trajectories_dir).resolve()
        if not trajectories_dir.is_dir():
            raise FileNotFoundError(f"trajectories_dir not found: {trajectories_dir}")

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir = output_dir / "_meta_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        # Materialize current_config to a file the agent can Read.
        if isinstance(current_config, Path):
            current_config_path = Path(current_config).resolve()
        else:
            current_config_path = (scratch_dir / "current.yaml").resolve()
            current_config.to_yaml_file(current_config_path)

        brief_path, context_path = self._prepare_brief_and_context(
            current_config_path=current_config_path,
            trajectories_dir=trajectories_dir,
            output_dir=output_dir,
            scratch_dir=scratch_dir,
        )

        # Build the harness for this round.
        from ..tracing.journal import HarnessJournal

        meta_ws = output_dir / "meta_workspace"
        meta_ws.mkdir(parents=True, exist_ok=True)
        tracer = HarnessJournal(base_dir=str(meta_ws / "sessions"), silent=False)
        # Auto-include output_dir so the meta-agent can write its deliverables.
        effective_write_roots = (output_dir, *self.allowed_write_roots)
        effective_write_files = tuple(self.allowed_write_files)
        if self.memo_path is not None:
            effective_write_files = (self.memo_path, *effective_write_files)

        cfg = build_meta_agent_harness_config(
            inner_model=self.inner_model,
            tracer=tracer,
            workspace_root=meta_ws,
            persona_root=_META_WORKSPACE_ROOT,
            extra_skills_dirs=self.extra_skills_dirs,
            max_cost_usd=self.max_cost_usd,
            output_dir=output_dir,
            allowed_write_roots=effective_write_roots,
            allowed_write_files=effective_write_files,
            step_deadline_reminder_step=self.step_deadline_reminder_step,
            step_deadline_early_reminder_step=self.step_deadline_early_reminder_step,
            step_deadline_output_within=self.step_deadline_output_within,
            **self.extra_harness_kws,
        )
        harness = self.inner_model.agentic(cfg)

        task = BaseTask(
            description=self._render_user_message(brief_path=brief_path),
            max_steps=self.max_steps,
            max_cost_usd=self.max_cost_usd,
        )

        logger.info(
            "[evolve] trajectories=%s output=%s memo=%s budget=%s",
            trajectories_dir,
            output_dir,
            self.memo_path,
            f"${self.max_cost_usd:.2f}" if self.max_cost_usd is not None else "unlimited",
        )

        # --- Run the agent turn -------------------------------------------
        t0 = time.time()
        timed_out = False
        try:
            await asyncio.wait_for(harness.run(task), timeout=self.wall_clock_s)
        except asyncio.TimeoutError:
            timed_out = True
            logger.warning("[evolve] wall_clock timeout after %.0fs", self.wall_clock_s)
        elapsed = time.time() - t0

        if timed_out:
            timeout_md = scratch_dir / "TIMEOUT.md"
            timeout_md.write_text(
                "# Evolve Timeout\n\n"
                f"The meta-agent did not finish within {self.wall_clock_s:.0f}s "
                f"wall-clock. Any files under `{output_dir}` may be partial "
                "writes that never went through the `validate` skill. The "
                "round is rejected here rather than risk promoting an "
                "unvalidated artifact.\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"meta-agent timed out after {self.wall_clock_s:.0f}s; see {timeout_md}")

        # --- Post-flight --------------------------------------------------
        out_yaml = output_dir / "config.yaml"
        if not out_yaml.is_file():
            findings_path = self._write_missing_config_findings(
                scratch_dir=scratch_dir,
                sessions_dir=meta_ws / "sessions",
                current_config_path=current_config_path,
                output_dir=output_dir,
                elapsed=elapsed,
            )
            raise RuntimeError(
                f"meta-agent finished after {elapsed:.1f}s but did not produce {out_yaml}. Inspect {findings_path}."
            )

        from .validate_workflow import EvolveValidator

        validator = EvolveValidator(
            inner_model=self.inner_model,
            memo_path=self.memo_path,
            replay_model=replay_model,
            replay_max_cost_usd=replay_max_cost_usd,
            replay_timeout_s=min(replay_timeout_s, 20.0),
        )
        await validator.run(
            out_yaml=out_yaml,
            current_config=current_config,
            scratch_dir=scratch_dir,
            output_dir=output_dir,
            elapsed=elapsed,
            compute_changeset_fn=compute_changeset,
        )

        logger.info("[evolve] produced %s in %.1fs", out_yaml, elapsed)
        return out_yaml

    # ----------------------- per-round prep -----------------------------

    def _prepare_brief_and_context(
        self,
        *,
        current_config_path: Path,
        trajectories_dir: Path,
        output_dir: Path,
        scratch_dir: Path,
    ) -> tuple[Path, Path | None]:
        """Write TASK.md + (when journal exists) CONTEXT.md. Return both paths."""
        context_path: Path | None = None
        if self.memo_path is not None and self.memo_path.is_file():
            from .journal import build_context, read_entries

            entries = read_entries(self.memo_path)
            if entries:
                next_round = max(e.round for e in entries) + 1
                candidate = build_context(
                    journal_path=self.memo_path,
                    current_round=next_round,
                    output_path=scratch_dir / "CONTEXT.md",
                    recent_window=self.context_recent_window,
                )
                if candidate is not None:
                    context_path = candidate

        brief_path = scratch_dir / "TASK.md"
        brief_path.write_text(
            self._render_task_brief(
                current_config_path=current_config_path,
                trajectories_dir=trajectories_dir,
                output_dir=output_dir,
                context_path=context_path,
            ),
            encoding="utf-8",
        )
        return brief_path, context_path

    def _render_task_brief(
        self,
        *,
        current_config_path: Path,
        trajectories_dir: Path,
        output_dir: Path,
        context_path: Path | None = None,
    ) -> str:
        memo_line = f"- `memo_path`: `{self.memo_path}`" if self.memo_path is not None else "- `memo_path`: (not set)"
        context_section = ""
        if context_path is not None and Path(context_path).is_file():
            context_section = (
                f"- `journal_context`: `{context_path}` — machine-rendered "
                "lever scoreboard, recent hypotheses, reverted bets, per-task "
                "history matrix, recent changesets. Read this after the memo to "
                "see which levers have been tried and how well their predicted-"
                "affected tasks actually flipped\n"
            )
        return (
            "# Evolve Brief\n\n"
            f"- `current_config`: `{current_config_path}`\n"
            f"- `trajectories_dir`: `{trajectories_dir}`\n"
            f"- `output_dir`: `{output_dir}`\n"
            f"{memo_line}\n"
            f"{context_section}"
            f"- budget: {self.max_cost_usd} USD, {self.max_steps} steps, "
            f"{self.wall_clock_s:.0f}s wall-clock\n\n"
            "## Deliverables (all under `output_dir`)\n\n"
            "- `config.yaml` — a HarnessConfig YAML that canonicalizes (required)\n"
            "- `tools/<name>.py` — optional new `@tool` modules\n"
            "- `processors/<name>.py` — optional new `MultiHookProcessor` classes\n"
            "- `templates/<name>.j2` — optional new system-prompt Jinja templates\n"
            "- `_meta_scratch/candidates.md` — when you change the config, list\n"
            "  candidates as `## Candidate C-N` sections (see orchestrator\n"
            "  evidence gate below)\n"
            "- `_meta_scratch/` — your own notes; this brief already lives here\n\n"
            "Read-only context (if present in `_meta_scratch/`):\n\n"
            "- `pareto_archive.json` — per-task pass/fail history across all rounds "
            "(tasks marked stuck/fragile/solved)\n"
            "- `history/OVERVIEW.md` — round-by-round score table\n"
            "- `history/R{i}_per_task.json` — per-task results for round i\n"
            "- `history/R{i}_to_R{j}.diff` — config diff between rounds\n"
            "- `env_probe.md` — which external sites/APIs are reachable "
            "(OK/BLOCKED/TIMEOUT) from this environment\n"
            "- `task_catalog.md` — all task questions being evaluated "
            "(read this to understand WHAT the agent needs to solve)\n\n"
            "## Global optimization constraint (Pareto-style)\n\n"
            "Do not optimize a narrow local win at the expense of global "
            "benchmark health. Prefer candidates that improve failing clusters "
            "while protecting already-passing clusters.\n\n"
            "For each shipped candidate, explicitly state:\n"
            "- `expected_global_gain`: which failing cluster(s) and why this can generalize\n"
            "- `regression_risk`: what could break outside `predicted_affected`\n"
            "- `cost_shift`: expected token/cost movement if the change lands\n\n"
            "A local improvement with likely net global degradation is not "
            "acceptable unless you provide unusually strong evidence and a clear rollback trigger.\n\n"
            "## Self-validation before `end_turn`\n\n"
            "No retry loop. If you end your turn with a broken artifact the\n"
            "round fails. `Read validate` for the CLI commands — at minimum\n"
            "run `canonicalize` on your new config before stopping, and run\n"
            "`dry_fire` / `contract` / `literals` when you authored anything\n"
            "under `tools/`, `processors/`, or `templates/`.\n\n"
            "## Decision contract (required)\n\n"
            "Before `end_turn`, make exactly one explicit decision:\n"
            "1) **Ship change**: write ALL of the following, then run validation:\n"
            "   - `output_dir/config.yaml` (required)\n"
            "   - `_meta_scratch/candidates.md` (REQUIRED when config changes —\n"
            "     at least one `## Candidate C-NNN` section with lens/lever/intent\n"
            "     tag, signal, verified body evidence, retroactive check, and\n"
            "     'Why X not Y' lever argument; the evidence gate hard-fails without it)\n"
            "   - a new `## Round N` section appended to `memo_path` journal with\n"
            "     `cited_candidates` frontmatter referencing your C-NNN IDs\n"
            "   - optional: `tools/<name>.py`, `processors/<name>.py`, `templates/<name>.j2`\n"
            "2) **Explicit no-op**: copy `current_config` byte-for-byte to\n"
            "   `output_dir/config.yaml`, then stop. No candidates.md needed.\n"
            "Analysis-only `end_turn` is invalid and is treated as a failed round.\n\n"
            "## Orchestrator post-flight\n\n"
            "After you stop, the orchestrator runs:\n"
            "1. **canonicalize** (always) — config.yaml must parse and every\n"
            "   template must render.\n"
            "2. **novelty** — a journal hypothesis_id marked `reverted` in\n"
            "   a prior round cannot be re-proposed.\n"
            "3. **evidence** — when the changeset is non-empty, the round\n"
            "   must produce `_meta_scratch/candidates.md` with at least\n"
            "   one `## Candidate C-N` section, and the journal entry's\n"
            "   `cited_candidates` frontmatter must reference ≥1 of those\n"
            "   IDs. Prevents shipping config changes without linked\n"
            "   evidence.\n"
            "4. **replay** — by default runs a synthetic smoke task\n"
            "   (`replay_mode=synthetic_task`): executes one tiny fixed\n"
            "   task through the run loop and fails on exception / timeout /\n"
            "   `exit_reason=error`. Optional `replay_mode=config_only`\n"
            "   keeps bind-only checking.\n"
        )

    @staticmethod
    def _render_user_message(*, brief_path: Path) -> str:
        return (
            "## Evolve Request\n\n"
            f"Your per-run brief lives at `{brief_path}` — `Read` it for paths, "
            "memo, and budgets.\n\n"
            "Follow the loop in SOUL.md. If a benchmark-specific playbook skill "
            "is loaded (e.g. `*-playbook`), use it for guidance on interventions "
            "that tend to move scores on this benchmark. Use the `reference` "
            "skill when authoring a new processor / tool / template or editing "
            "config.yaml. Append to the memo before you stop (see `journal`).\n\n"
            "Ship a real intervention when the evidence supports one. "
            "Under-spending while ignoring live failure signals is a worse "
            "outcome than over-spending on an evidence-backed change — the "
            "caller will auto-revert regressions.\n\n"
            "Do not end with analysis only. Before `end_turn`, choose one path:\n"
            "- write `output_dir/config.yaml` with a real change, or\n"
            "- copy `current_config` to `output_dir/config.yaml` as explicit no-op."
        )

    @staticmethod
    def _read_last_assistant_message(sessions_dir: Path) -> str | None:
        """Best-effort capture of the final assistant content from journal JSONL."""
        if not sessions_dir.is_dir():
            return None
        jsonl_files = sorted(sessions_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in jsonl_files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for raw in reversed(lines):
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") not in {"raw_assistant", "assistant"}:
                    continue
                msg = rec.get("message") or {}
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
            # Only inspect the latest segment first; this is enough for diagnostics.
            break
        return None

    def _write_missing_config_findings(
        self,
        *,
        scratch_dir: Path,
        sessions_dir: Path,
        current_config_path: Path,
        output_dir: Path,
        elapsed: float,
    ) -> Path:
        findings_path = scratch_dir / "DECISION_REQUIRED.md"
        last_assistant = self._read_last_assistant_message(sessions_dir)
        last_assistant_block = (
            f"## Last assistant message excerpt\n\n{last_assistant}\n\n"
            if last_assistant
            else "## Last assistant message excerpt\n\n(Not available)\n\n"
        )
        findings_path.write_text(
            "# Missing `config.yaml` (decision not completed)\n\n"
            f"The meta-agent finished after {elapsed:.1f}s but no "
            f"`{output_dir / 'config.yaml'}` was written.\n\n"
            "This usually means it ended with analysis but did not commit to a final decision.\n\n"
            "## Required decision before `end_turn`\n\n"
            "Choose exactly one:\n"
            "1. **Ship change**: write `output_dir/config.yaml` (+ optional authored files).\n"
            "2. **Explicit no-op**: copy current config byte-for-byte:\n\n"
            f"   `cp {current_config_path} {output_dir / 'config.yaml'}`\n\n"
            "Either choice is valid; missing `config.yaml` is not.\n\n"
            f"{last_assistant_block}",
            encoding="utf-8",
        )
        return findings_path
