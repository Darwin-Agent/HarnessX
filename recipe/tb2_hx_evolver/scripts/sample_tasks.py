#!/usr/bin/env python3
"""
Sample N tasks from the harbor task cache, filtered by agent timeout.

Steps:
  1. Discover all tasks in the harbor cache (--tasks-cache).
  2. Keep only tasks whose agent timeout_sec <= --max-agent-timeout.
  3. Sample --n tasks with fixed --seed.
  4. Write the result to --output.

Usage:
    # Reproduce tasks_sample16_seed42_lt15m.json
    python recipe/tb2_hx_evolver/scripts/sample_tasks.py \
        --seed 42 --n 16 --max-agent-timeout 900 \
        --output recipe/tb2_hx_evolver/tasks_sample16_seed42_lt15m.json

Does NOT depend on recipe/tb2_evolver/.
"""
from __future__ import annotations

import argparse
import json
import random
import tomllib
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_OUTPUT_DIR = _SCRIPT_DIR.parent          # recipe/tb2_hx_evolver/
_HARBOR_TASKS_CACHE = Path("/root/.cache/harbor/tasks")
_DEFAULT_MAX_AGENT_TIMEOUT = 900.0                # 15 minutes


def _discover_all_tasks(cache_dir: Path) -> list[str]:
    """Return all task names found in the harbor cache."""
    tasks: set[str] = set()
    for hdir in cache_dir.iterdir():
        if not hdir.is_dir():
            continue
        for task_dir in hdir.iterdir():
            if (task_dir / "task.toml").exists():
                tasks.add(task_dir.name)
    return sorted(tasks)


def _get_agent_timeout(task_name: str, cache_dir: Path) -> float | None:
    """Return agent timeout_sec from task.toml, or None if not found."""
    for hdir in cache_dir.iterdir():
        if not hdir.is_dir():
            continue
        toml_path = hdir / task_name / "task.toml"
        if toml_path.exists():
            data = tomllib.loads(toml_path.read_text())
            return data.get("agent", {}).get("timeout_sec")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample tasks from harbor cache filtered by agent timeout."
    )
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed for reproducibility.")
    parser.add_argument("--n", type=int, default=16,
                        help="Number of tasks to sample (default: 16).")
    parser.add_argument("--max-agent-timeout", type=float,
                        default=_DEFAULT_MAX_AGENT_TIMEOUT,
                        help=f"Keep only tasks with agent timeout_sec <= this value "
                             f"(default: {_DEFAULT_MAX_AGENT_TIMEOUT:.0f}s = 15 min).")
    parser.add_argument("--tasks-cache", type=Path, default=_HARBOR_TASKS_CACHE,
                        help=f"Harbor task cache directory (default: {_HARBOR_TASKS_CACHE}).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path. Default: "
                             "<script-dir>/../tasks_sample<n>_seed<seed>_lt<timeout>m.json")
    args = parser.parse_args()

    if not args.tasks_cache.exists():
        raise SystemExit(f"ERROR: tasks-cache not found: {args.tasks_cache}")

    # Step 1: discover
    all_tasks = _discover_all_tasks(args.tasks_cache)
    print(f"Discovered {len(all_tasks)} tasks in {args.tasks_cache}")

    # Step 2: filter by timeout
    eligible: list[tuple[str, float | None]] = []
    excluded: list[tuple[str, float]] = []
    for task in all_tasks:
        timeout = _get_agent_timeout(task, args.tasks_cache)
        if timeout is not None and timeout > args.max_agent_timeout:
            excluded.append((task, timeout))
        else:
            eligible.append((task, timeout))

    print(f"Eligible (timeout <= {args.max_agent_timeout:.0f}s): {len(eligible)}  "
          f"Excluded: {len(excluded)}")

    if args.n > len(eligible):
        raise SystemExit(
            f"ERROR: --n {args.n} exceeds eligible task count ({len(eligible)})"
        )

    # Step 3: sample
    rng = random.Random(args.seed)
    sample: list[tuple[str, float | None]] = sorted(rng.sample(eligible, args.n))

    # Step 4: write
    timeout_min = int(args.max_agent_timeout // 60)
    output = args.output or (
        _DEFAULT_OUTPUT_DIR
        / f"tasks_sample{args.n}_seed{args.seed}_lt{timeout_min}m.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([t for t, _ in sample], indent=2) + "\n")

    print(f"\nSampled {args.n} tasks (seed={args.seed}) → {output}")
    for task, timeout in sample:
        t_str = f"{timeout:.0f}s" if timeout is not None else "unknown"
        print(f"  {task}  (agent_timeout={t_str})")

    if excluded:
        print(f"\nExcluded {len(excluded)} task(s) with timeout > {args.max_agent_timeout:.0f}s:")
        for task, timeout in excluded:
            print(f"  {task}  ({timeout:.0f}s)")


if __name__ == "__main__":
    main()
