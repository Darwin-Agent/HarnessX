#!/usr/bin/env python3
"""Evaluate HarnessX SWE-bench results using the official SWE-bench harness.

Usage:
    python -m benchmarks.swebench.evaluate benchmarks/swebench/results/improved_v2/results.json
    python -m benchmarks.swebench.evaluate results.json --max-workers 4
"""

import argparse
import json
import os
from pathlib import Path


def convert_to_predictions(results_path: str, output_path: str) -> list[str]:
    """Convert HarnessX results JSON to SWE-bench predictions format.

    SWE-bench expects a JSONL file with: instance_id, model_name_or_path, model_patch
    Returns list of instance_ids that have patches.
    """
    results = json.load(open(results_path))
    instance_ids = []

    with open(output_path, "w") as f:
        for r in results:
            if not r.get("patch"):
                continue
            pred = {
                "instance_id": r["instance_id"],
                "model_name_or_path": "harnessx",
                "model_patch": r["patch"],
            }
            f.write(json.dumps(pred) + "\n")
            instance_ids.append(r["instance_id"])

    print(f"Wrote {len(instance_ids)} predictions to {output_path}")
    return instance_ids


def run_evaluation(
    predictions_path: str,
    instance_ids: list[str],
    max_workers: int = 2,
    timeout: int = 300,
    run_id: str = "harnessx",
    report_dir: str = ".",
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
) -> dict:
    """Run the SWE-bench evaluation harness."""
    from swebench.harness.run_evaluation import main as swebench_eval

    swebench_eval(
        dataset_name=dataset_name,
        split="test",
        instance_ids=instance_ids,
        predictions_path=predictions_path,
        max_workers=max_workers,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id=run_id,
        timeout=timeout,
        namespace=None,
        rewrite_reports=True,
        modal=False,
        report_dir=report_dir,
    )


def summarize_reports(report_dir: str, run_id: str) -> dict:
    """Read SWE-bench evaluation reports and summarize results."""
    report_path = Path(report_dir) / run_id
    if not report_path.exists():
        print(f"No report directory found at {report_path}")
        return {}

    # Look for the results JSON
    results = {}
    for f in report_path.glob("*.json"):
        data = json.load(open(f))
        if isinstance(data, dict):
            results.update(data)

    resolved = sum(1 for v in results.values() if isinstance(v, dict) and v.get("resolved", False))
    total = len(results)
    print(f"\n{'=' * 60}")
    print("SWE-bench Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Total evaluated: {total}")
    print(f"Resolved: {resolved} ({100 * resolved / total:.1f}%)" if total else "No results")
    print(f"{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SWE-bench results")
    parser.add_argument("results", help="Path to HarnessX results JSON")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--run-id", default="harnessx")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    args = parser.parse_args()

    if args.report_dir is None:
        args.report_dir = str(Path(args.results).parent / "eval_reports")
    os.makedirs(args.report_dir, exist_ok=True)

    # Convert to predictions format
    pred_path = str(Path(args.results).parent / "predictions.jsonl")
    instance_ids = convert_to_predictions(args.results, pred_path)

    if not instance_ids:
        print("No patches to evaluate!")
        return

    print(f"Evaluating {len(instance_ids)} predictions...")
    print(f"Report dir: {args.report_dir}")

    # Run evaluation
    run_evaluation(
        predictions_path=pred_path,
        instance_ids=instance_ids,
        max_workers=args.max_workers,
        timeout=args.timeout,
        run_id=args.run_id,
        report_dir=args.report_dir,
        dataset_name=args.dataset,
    )

    # Summarize
    summarize_reports(args.report_dir, args.run_id)


if __name__ == "__main__":
    main()
