#!/usr/bin/env python3
"""Monitor a running SWE-bench benchmark and show live stats."""

import json
import sys
from pathlib import Path


def analyze_results(path: str) -> None:
    data = json.load(open(path))
    total = len(data)
    if total == 0:
        print("No results yet.")
        return

    with_patch = sum(1 for r in data if r.get("has_patch"))
    errors = sum(1 for r in data if r.get("error") or r.get("exit_reason") == "error")
    total_cost = sum(r.get("cost_usd", 0) for r in data)
    avg_cost = total_cost / total
    avg_steps = sum(r.get("steps", 0) for r in data) / total
    avg_time = sum(r.get("elapsed_s", 0) for r in data) / total
    total_tokens = sum(r.get("tokens", 0) for r in data)

    print(f"{'=' * 60}")
    print(f"Progress: {total}/300")
    print(f"With patch: {with_patch} ({100 * with_patch / total:.1f}%)")
    print(f"Errors: {errors} ({100 * errors / total:.1f}%)")
    print(f"Avg steps: {avg_steps:.1f} | Avg time: {avg_time:.0f}s")
    print(f"Total cost: ${total_cost:.2f} | Avg cost: ${avg_cost:.4f}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Est. total cost (300): ${avg_cost * 300:.2f}")
    print(f"{'=' * 60}")

    # Per-repo breakdown
    repos = {}
    for r in data:
        repo = r["instance_id"].rsplit("-", 1)[0].replace("__", "/")
        # Simplify to repo name
        repo = repo.split("/")[0] + "/" + repo.split("/")[1] if "/" in repo else repo
        if repo not in repos:
            repos[repo] = {"total": 0, "patch": 0, "errors": 0}
        repos[repo]["total"] += 1
        if r.get("has_patch"):
            repos[repo]["patch"] += 1
        if r.get("error") or r.get("exit_reason") == "error":
            repos[repo]["errors"] += 1

    print("\nPer-repo breakdown:")
    for repo in sorted(repos, key=lambda r: repos[r]["total"], reverse=True):
        s = repos[repo]
        pct = 100 * s["patch"] / s["total"] if s["total"] else 0
        print(f"  {repo:40s} {s['patch']:3d}/{s['total']:3d} ({pct:5.1f}%) err={s['errors']}")

    # Show errors
    error_records = [r for r in data if r.get("error") or r.get("exit_reason") == "error"]
    if error_records:
        print(f"\nErrors ({len(error_records)}):")
        for r in error_records[:10]:
            err = r.get("error", r.get("exit_reason", "unknown"))
            print(f"  {r['instance_id']}: {str(err)[:100]}")

    # Show no-patch instances
    no_patch = [r for r in data if not r.get("has_patch") and r.get("exit_reason") != "error"]
    if no_patch:
        print(f"\nNo patch ({len(no_patch)}):")
        for r in no_patch[:10]:
            print(f"  {r['instance_id']}: exit={r.get('exit_reason')} steps={r.get('steps')}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/swebench/results/improved_v2/results.json"
    if not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)
    analyze_results(path)
