#!/usr/bin/env python3
"""Copy task trajectory directories from a source eval dir to a destination.

Reads a tasks JSON file (list of task names), then copies matching
subdirectories from --src to --dst.  Source directories are named
``<task-name>__<hash>`` (the standard terminal-bench layout); the script
matches on the task-name prefix so the hash suffix is ignored.

Usage:
    python copy_task_trajs.py \\
        --tasks tasks_sample16_seed42_timeout900s.json \\
        --src .benchmarks/tb2-baseline-results/tb2-0429 \\
        --dst /tmp/selected_trajs

Options:
    --tasks   Path to task list JSON (list of task-name strings).
    --src     Source eval output directory containing <task>__<hash> dirs.
    --dst     Destination directory (created if it does not exist).
    --dry-run Print what would be copied without copying anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy task trajectory dirs from a source eval dir to a destination.")
    parser.add_argument("--tasks", type=Path, required=True, help="Task list JSON file.")
    parser.add_argument("--src", type=Path, required=True, help="Source eval directory.")
    parser.add_argument("--dst", type=Path, required=True, help="Destination directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying.")
    args = parser.parse_args()

    tasks: list[str] = json.loads(args.tasks.read_text())
    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()

    if not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")

    # Build a lookup: task_name -> list of matching source dirs
    task_set = set(tasks)
    matches: dict[str, list[Path]] = {t: [] for t in task_set}
    for entry in src.iterdir():
        if not entry.is_dir():
            continue
        task_name = entry.name.split("__")[0]
        if task_name in task_set:
            matches[task_name].append(entry)

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    found = 0
    missing = []
    for task in sorted(tasks):
        dirs = matches.get(task, [])
        if not dirs:
            missing.append(task)
            print(f"  MISSING  {task}")
            continue
        for src_dir in sorted(dirs):
            dest_dir = dst / src_dir.name
            if args.dry_run:
                print(f"  DRY-RUN  {src_dir} -> {dest_dir}")
            else:
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(src_dir, dest_dir)
                print(f"  COPIED   {src_dir.name} -> {dst}")
            found += 1

    print(
        f"\n{'[dry-run] ' if args.dry_run else ''}Done: {found} dir(s) copied"
        f", {len(missing)} task(s) not found in source."
    )
    if missing:
        print("Missing tasks:")
        for t in missing:
            print(f"  {t}")


if __name__ == "__main__":
    main()
