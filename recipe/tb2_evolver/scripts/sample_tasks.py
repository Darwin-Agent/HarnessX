#!/usr/bin/env python3
"""Randomly sample N tasks from a task list JSON with a fixed seed.

Also looks up each task's agent timeout from the harbor task cache and
filters out tasks whose agent timeout exceeds --max-agent-timeout (default: 1800s).

Usage:
    python sample_tasks.py --seed 42
    python sample_tasks.py --seed 42 --n 10
    python sample_tasks.py --seed 42 --input tasks_all_tb2.json --output tasks_sample10_seed42.json
    python sample_tasks.py --seed 42 --n 20 --max-agent-timeout 900
"""

from __future__ import annotations

import argparse
import json
import random
import tomllib
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.parent  # recipe/tb2_evolver/
_HARBOR_TASKS_CACHE = Path("/root/.cache/harbor/tasks")
_DEFAULT_MAX_AGENT_TIMEOUT = 1800.0


def _find_task_toml(task_name: str, cache_dir: Path) -> Path | None:
    """Search harbor task cache for task.toml by task directory name."""
    for hash_dir in cache_dir.iterdir():
        if not hash_dir.is_dir():
            continue
        task_dir = hash_dir / task_name
        toml = task_dir / "task.toml"
        if toml.exists():
            return toml
    return None


def get_agent_timeout(task_name: str, cache_dir: Path = _HARBOR_TASKS_CACHE) -> float | None:
    """Return agent timeout_sec from task.toml, or None if not found."""
    toml_path = _find_task_toml(task_name, cache_dir)
    if toml_path is None:
        return None
    data = tomllib.loads(toml_path.read_text())
    return data.get("agent", {}).get("timeout_sec")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample tasks with a fixed seed.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("--n", type=int, default=10, help="Number of tasks to sample (default: 10).")
    parser.add_argument(
        "--input",
        type=Path,
        default=_SCRIPT_DIR / "tasks_all_tb2.json",
        help="Input task list JSON (default: tasks_all_tb2.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: tasks_sample{n}_seed{seed}.json next to input).",
    )
    parser.add_argument(
        "--max-agent-timeout",
        type=float,
        default=_DEFAULT_MAX_AGENT_TIMEOUT,
        help=f"Drop tasks whose agent timeout_sec exceeds this value (default: {_DEFAULT_MAX_AGENT_TIMEOUT:.0f}).",
    )
    parser.add_argument(
        "--tasks-cache",
        type=Path,
        default=_HARBOR_TASKS_CACHE,
        help=f"Harbor task cache directory (default: {_HARBOR_TASKS_CACHE}).",
    )
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    tasks: list[str] = json.loads(input_path.read_text())

    if args.n > len(tasks):
        raise ValueError(f"--n {args.n} exceeds available tasks ({len(tasks)})")

    rng = random.Random(args.seed)
    sample = sorted(rng.sample(tasks, args.n))

    # Look up agent timeouts and filter.
    filtered: list[str] = []
    skipped: list[tuple[str, float | None]] = []
    for task in sample:
        timeout = get_agent_timeout(task, args.tasks_cache)
        if timeout is not None and timeout > args.max_agent_timeout:
            skipped.append((task, timeout))
        else:
            filtered.append(task)

    output_path: Path = (
        args.output or input_path.parent / f"tasks_sample{args.n}_seed{args.seed}_act{len(filtered)}.json"
    )
    output_path.write_text(json.dumps(filtered, indent=2) + "\n")

    print(f"Sampled {args.n} tasks (seed={args.seed}), kept {len(filtered)} after timeout filter → {output_path}")
    for task in filtered:
        timeout = get_agent_timeout(task, args.tasks_cache)
        t_str = f"{timeout:.0f}s" if timeout is not None else "unknown"
        print(f"  {task}  (agent_timeout={t_str})")
    if skipped:
        print(f"\nSkipped {len(skipped)} task(s) with agent_timeout > {args.max_agent_timeout:.0f}s:")
        for task, timeout in skipped:
            t_str = f"{timeout:.0f}s" if timeout is not None else "unknown"
            print(f"  {task}  (agent_timeout={t_str})")


if __name__ == "__main__":
    # python recipe/tb2_evolver/scripts/sample_tasks.py --seed 42 --n 16 --max-agent-timeout 2000
    main()
