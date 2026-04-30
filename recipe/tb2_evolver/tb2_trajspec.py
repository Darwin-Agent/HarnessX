# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""TB2-specific round helpers used by this recipe's evolve loop.

This module is the recipe-local adapter between ``MetaAgent.evolve``
and the TB2 eval harness. The recipe owns its own round loop
(``run.py``), so these helpers live as a plain class + module-level
functions rather than a shared abstraction.

Responsibilities:

- ``resolve_trajectories_for_round`` — produce the trajectories directory
  for a given round by either running TB2 fresh with ``current_config``
  (``run_mode="rerun"``) or reusing an existing directory
  (``run_mode="reuse"``).
- ``promote_round_output`` — copy the meta-agent's new ``config.yaml``
  (and any authored ``processors/``) into the canonical ``run_root/R{n}/``
  layout the TB2 verify command expects.
- ``read_round_pass_rate`` — parse the aggregate pass rate from TB2's
  ``result.json`` for display in the run summary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-task result helpers
# ---------------------------------------------------------------------------


def read_per_task_results(trajectories_dir: Path) -> dict[str, bool]:
    """Return ``{task_name: passed}`` for each completed trial in the dir."""
    results: dict[str, bool] = {}
    td = Path(trajectories_dir).resolve()
    if not td.is_dir():
        return results
    for trial_dir in td.iterdir():
        if not trial_dir.is_dir() or trial_dir.name.startswith("_"):
            continue
        rp = trial_dir / "result.json"
        if not rp.is_file():
            continue
        try:
            obj = json.loads(rp.read_text(encoding="utf-8"))
            task_name = str(obj.get("task_name") or trial_dir.name.split("__")[0])
            vr = obj.get("verifier_result") or {}
            reward = (vr.get("rewards") or {}).get("reward")
            results[task_name] = isinstance(reward, (int, float)) and float(reward) > 0
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# Per-task trajectory .md writer (fast-path for meta-agent)
# ---------------------------------------------------------------------------


def _parse_agent_jsonl(trial_dir: Path) -> tuple[int, dict[str, int]]:
    """Return (step_count, tool_call_counts) from agent oh_runs JSONL.

    Supports HarnessX trace format: ``raw_assistant`` events with
    ``message.tool_calls[].name``.
    """
    steps = 0
    tool_counts: dict[str, int] = defaultdict(int)
    oh_runs = trial_dir / "agent" / "oh_runs"
    if not oh_runs.is_dir():
        return steps, dict(tool_counts)
    # Walk one level of subdirectories (session_id subdirs)
    search_dirs = [oh_runs] + [p for p in oh_runs.iterdir() if p.is_dir()]
    for search_dir in search_dirs:
        for jsonl_path in sorted(search_dir.glob("*.jsonl")):
            if "_trace" in jsonl_path.name or "_state" in jsonl_path.name:
                continue
            try:
                for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    # HarnessX format: raw_assistant with tool_calls
                    if obj.get("type") == "raw_assistant":
                        msg = obj.get("message") or {}
                        for tc in msg.get("tool_calls") or []:
                            name = tc.get("name") or tc.get("function", {}).get("name") or "unknown"
                            tool_counts[str(name)] += 1
                            steps += 1
            except Exception:
                continue
    return steps, dict(tool_counts)


def _write_task_trajectory_mds(trajectories_dir: Path) -> None:
    """Write <task_name>.md sidecar files for each completed trial.

    These give the meta-agent a fast ``Read <task>.md limit=30`` path
    to key signals (reward, verifier tail, step count, tool calls) without
    trawling raw JSON and JSONL files.
    """
    td = Path(trajectories_dir).resolve()
    if not td.is_dir():
        return
    for trial_dir in td.iterdir():
        if not trial_dir.is_dir() or trial_dir.name.startswith("_"):
            continue
        rp = trial_dir / "result.json"
        if not rp.is_file():
            continue
        try:
            obj = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue

        task_name = str(obj.get("task_name") or trial_dir.name.split("__")[0])
        trial_name = str(obj.get("trial_name") or trial_dir.name)

        vr = obj.get("verifier_result") or {}
        reward_raw = (vr.get("rewards") or {}).get("reward")
        reward = float(reward_raw) if isinstance(reward_raw, (int, float)) else None
        passed = isinstance(reward, float) and reward > 0

        exc_info = obj.get("exception_info")
        exception_str = None
        if exc_info:
            if isinstance(exc_info, dict):
                exception_str = exc_info.get("message") or exc_info.get("type") or str(exc_info)
            else:
                exception_str = str(exc_info)[:200]

        # Duration
        duration_s: float | None = None
        try:
            ae = obj.get("agent_execution") or {}
            sa, fa = ae.get("started_at"), ae.get("finished_at")
            if sa and fa:
                from datetime import datetime, timezone

                fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
                s = datetime.strptime(sa, fmt).replace(tzinfo=timezone.utc)
                f = datetime.strptime(fa, fmt).replace(tzinfo=timezone.utc)
                duration_s = round((f - s).total_seconds(), 1)
        except Exception:
            pass

        # Agent steps + tool calls from JSONL
        steps, tool_counts = _parse_agent_jsonl(trial_dir)

        # Verifier tail (last 25 lines of test-stdout.txt)
        verifier_tail = ""
        test_stdout = trial_dir / "verifier" / "test-stdout.txt"
        if test_stdout.is_file():
            try:
                lines = test_stdout.read_text(encoding="utf-8", errors="replace").splitlines()
                verifier_tail = "\n".join(lines[-25:])
            except Exception:
                pass

        # Build YAML frontmatter
        fm_lines = [
            "---",
            f"task: {task_name}",
            f"trial: {trial_name}",
            f"passed: {str(passed).lower()}",
            f"reward: {reward if reward is not None else 'null'}",
            f"steps: {steps if steps else 'null'}",
        ]
        if tool_counts:
            fm_lines.append("tool_calls:")
            for tn, cnt in sorted(tool_counts.items(), key=lambda x: -x[1]):
                fm_lines.append(f"  {tn}: {cnt}")
        if duration_s is not None:
            fm_lines.append(f"duration_s: {duration_s}")
        if exception_str:
            exc_escaped = exception_str.replace("\n", " ").replace('"', "'")
            fm_lines.append(f'exception: "{exc_escaped}"')
        fm_lines.append("---")

        body_parts = ["\n".join(fm_lines)]
        if verifier_tail:
            body_parts.append(f"\n## Verifier output (tail)\n```\n{verifier_tail}\n```\n")
        else:
            body_parts.append("\n*(no verifier output)*\n")

        md_path = td / f"{task_name}.md"
        try:
            md_path.write_text("\n".join(body_parts), encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Round adapter
# ---------------------------------------------------------------------------


class TB2RoundAdapter:
    """Recipe-local helper bundling TB2 trajectory generation + promotion.

    Thin: just packages the eval-script invocation + round-output copy so
    ``run.py``'s for-loop stays readable. Not an ABC — the recipe owns its
    own loop now.
    """

    def __init__(
        self,
        *,
        baseline_config: Path,
        trajectories_root: Path | None = None,
        run_mode: str = "rerun",
        strict_round_trajectories: bool = True,
        repo_root: Path | None = None,
        eval_script: Path | None = None,
        eval_concurrent: int = 2,
        eval_resume: bool = False,
        eval_timeout_s: float | None = None,
        task_names: list[str] | None = None,
        r0_trajectories: Path | None = None,
    ) -> None:
        self.baseline_config = Path(baseline_config).resolve()
        self.trajectories_root = Path(trajectories_root).resolve() if trajectories_root is not None else None
        self.r0_trajectories = Path(r0_trajectories).resolve() if r0_trajectories is not None else None
        self.run_mode = run_mode
        self.strict_round_trajectories = bool(strict_round_trajectories)
        self.repo_root = (
            Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parent.parent.parent
        )
        self.eval_script = (
            Path(eval_script).resolve()
            if eval_script is not None
            else (self.repo_root / "benchmarks/terminal_bench_2/scripts/eval_opensandbox.sh").resolve()
        )
        self.eval_concurrent = int(eval_concurrent)
        self.eval_resume = bool(eval_resume)
        self.eval_timeout_s = float(eval_timeout_s) if eval_timeout_s is not None else None
        self._explicit_task_names: list[str] | None = list(task_names) if task_names else None
        if self.run_mode not in {"reuse", "rerun"}:
            raise ValueError(f"Unsupported TB2 run_mode: {self.run_mode} (expected reuse|rerun)")
        if self.run_mode == "reuse" and self.trajectories_root is None:
            raise ValueError("trajectories_root is required for run_mode='reuse'")
        if not self._explicit_task_names and self.trajectories_root is None:
            raise ValueError("Either task_names or trajectories_root must be provided")
        if self.eval_concurrent < 1:
            raise ValueError(f"eval_concurrent must be >= 1, got {self.eval_concurrent}")

    def initial_config(self) -> Path:
        if not self.baseline_config.is_file():
            raise FileNotFoundError(f"baseline config not found: {self.baseline_config}")
        return self.baseline_config

    async def resolve_trajectories_for_round(
        self,
        *,
        run_root: Path,
        input_round: int,
        current_config: Path,
    ) -> Path:
        if self.run_mode == "rerun":
            if input_round == 0 and self.r0_trajectories is not None:
                traj_dir = self.r0_trajectories
                logger.info("R0: using pre-existing trajectories (skipping eval) → %s", traj_dir)
            else:
                traj_dir = await self._run_tb2_eval_for_round_async(
                    run_root=run_root,
                    input_round=input_round,
                    current_config=Path(current_config).resolve(),
                )
        else:
            traj_dir = _resolve_round_trajectories(
                trajectories_root=self.trajectories_root,
                input_round=input_round,
                strict=self.strict_round_trajectories,
            )

        _write_task_trajectory_mds(traj_dir)
        return traj_dir

    def promote_round_output(
        self,
        *,
        run_root: Path,
        output_round: int,
        rx_output_dir: Path,
        output_config: Path,
    ) -> Path:
        """Copy the evolved config into ``run_root/R{n}/`` canonical layout."""
        return _promote_rx_output(
            run_root=run_root,
            output_round=output_round,
            rx_output_dir=rx_output_dir,
            output_config=output_config,
        )

    def read_round_score(self, trajectories_dir: Path) -> float | None:
        return _read_tb2_pass_rate(trajectories_dir)

    async def _run_tb2_eval_for_round_async(
        self,
        *,
        run_root: Path,
        input_round: int,
        current_config: Path,
    ) -> Path:
        """Run TB2 eval with *current_config* and return the resulting run dir."""
        if not self.eval_script.is_file():
            raise FileNotFoundError(f"TB2 eval script not found: {self.eval_script}")

        task_names = self._task_names()
        if not task_names:
            raise RuntimeError("Unable to derive TB2 task names from subset manifest; cannot rerun trajectories.")

        job_name = f"{run_root.name}-r{input_round}-traj"
        cmd = [
            "bash",
            str(self.eval_script),
            "--job-name",
            job_name,
            "-n",
            str(self.eval_concurrent),
        ]
        if self.eval_resume:
            cmd.append("--resume")
        for t in task_names:
            cmd.extend(["-t", t])

        env = os.environ.copy()
        env["TB2_HARNESS_CONFIG"] = str(current_config)

        await asyncio.wait_for(
            asyncio.to_thread(
                subprocess.run,
                cmd,
                cwd=str(self.repo_root),
                env=env,
                check=True,
            ),
            timeout=self.eval_timeout_s,
        )

        run_dir = (self.repo_root / ".benchmarks/tb2" / job_name).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Expected TB2 eval run dir not found after rerun: {run_dir}")
        return run_dir

    def _task_names(self) -> list[str]:
        if self._explicit_task_names:
            return self._explicit_task_names
        if self.trajectories_root is None:
            return []
        manifest = self.trajectories_root / "_manifest.json"
        if not manifest.is_file():
            return []
        try:
            obj = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return []
        trials = obj.get("trials")
        if not isinstance(trials, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for trial in trials:
            name = str(trial).split("__", 1)[0].strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out


# ---------------------------------------------------------------------------
# Pure helpers (filesystem-only, no network)
# ---------------------------------------------------------------------------


def _read_tb2_pass_rate(trajectories_dir: Path) -> float | None:
    """Return aggregate pass rate from result.json at or above trajectories_dir."""
    for candidate in (trajectories_dir, trajectories_dir.parent, trajectories_dir.parent.parent):
        p = candidate / "result.json"
        if not p.is_file():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            stats = obj.get("stats") or {}
            metrics = stats.get("metrics")
            if isinstance(metrics, list) and metrics:
                val = metrics[0].get("mean") if isinstance(metrics[0], dict) else None
                if isinstance(val, (int, float)):
                    return float(val)
            for eval_data in (stats.get("evals") or {}).values():
                if not isinstance(eval_data, dict):
                    continue
                m = eval_data.get("metrics")
                if isinstance(m, list) and m:
                    val = m[0].get("mean") if isinstance(m[0], dict) else None
                    if isinstance(val, (int, float)):
                        return float(val)
        except Exception:
            continue
    return None


def _resolve_round_trajectories(
    *,
    trajectories_root: Path,
    input_round: int,
    strict: bool,
) -> Path:
    def _has_any_files(d: Path) -> bool:
        return any(p.is_file() for p in d.rglob("*"))

    round_candidates = [
        trajectories_root / f"R{input_round}" / "trajectories",
        trajectories_root / f"R{input_round}",
    ]
    for cand in round_candidates:
        if cand.is_dir() and _has_any_files(cand):
            return cand.resolve()

    if trajectories_root.is_dir() and _has_any_files(trajectories_root):
        return trajectories_root.resolve()

    if strict:
        raise FileNotFoundError(
            f"Missing trajectories for round R{input_round} under {trajectories_root}. "
            "Expected R{n}/trajectories or direct trajectories dir."
        )

    r0_traj = trajectories_root / "R0" / "trajectories"
    if r0_traj.is_dir() and _has_any_files(r0_traj):
        return r0_traj.resolve()

    for p in sorted(trajectories_root.glob("R*/trajectories")):
        if p.is_dir() and _has_any_files(p):
            return p.resolve()
    raise FileNotFoundError(f"Unable to resolve trajectories for round R{input_round} from {trajectories_root}")


def _promote_rx_output(
    *,
    run_root: Path,
    output_round: int,
    rx_output_dir: Path,
    output_config: Path,
) -> Path:
    if not output_config.is_file():
        raise FileNotFoundError(f"output config not found: {output_config}")

    round_dir = run_root / f"R{output_round}"
    round_dir.mkdir(parents=True, exist_ok=True)

    promoted_cfg = round_dir / "config.yaml"
    shutil.copy2(output_config, promoted_cfg)

    round_evolve_dir = round_dir / "evolve"
    round_evolve_dir.mkdir(parents=True, exist_ok=True)
    evolve_dst = round_evolve_dir / "config.yaml"
    if not (evolve_dst.exists() and os.path.samefile(output_config, evolve_dst)):
        shutil.copy2(output_config, evolve_dst)

    src_processors = rx_output_dir / "processors"
    if src_processors.is_dir():
        dst_processors = round_evolve_dir / "processors"
        # Guard: src and dst may be the same path when rx_output_dir == round_evolve_dir
        if dst_processors.exists() and not os.path.samefile(src_processors, dst_processors):
            shutil.rmtree(dst_processors)
        if not (dst_processors.exists() and os.path.samefile(src_processors, dst_processors)):
            shutil.copytree(src_processors, dst_processors)

    return promoted_cfg.resolve()


__all__ = [
    "TB2RoundAdapter",
    "read_per_task_results",
]
