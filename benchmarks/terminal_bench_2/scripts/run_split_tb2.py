#!/usr/bin/env python3
# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.terminal_bench_2.scripts.tb2_eval import (
    build_command,
    _completed_task_names,
)
from benchmarks.terminal_bench_2.defaults import REQUEST_TIMEOUT_SEC, JOBS_DIR

# ── Task split ────────────────────────────────────────────────────────────────
# Criteria:
#   MODAL   — needs live internet (API calls, web scraping, downloads) OR
#             large storage >5 GB / memory >4 GB (QEMU images, ML datasets,
#             FastText training, C4 reshard)
#   DAYTONA — everything else (standard coding / system / file tasks)

MODAL_TASKS: list[str] = [
    # ML / internet / large-resource tasks — empirically validated against TB2.0 dataset.
    # Source: haiku-eval-modal + full-eval-v4-modal (14 tasks that ran in Modal historically)
    # plus 4 resource-heavy tasks confirmed to exist in TB2.0.
    #
    # NOTE: task names below were verified against actual result.json files from past runs.
    # Do NOT add names that haven't been confirmed to exist in the TB2.0 dataset.
    "build-cython-ext",
    "build-pmars",
    "caffe-cifar-10",
    "count-dataset-tokens",
    "extract-moves-from-video",
    "gpt2-codegolf",
    "hf-model-inference",
    "mcmc-sampling-stan",
    "mteb-leaderboard",
    "mteb-retrieve",
    "openssl-selfsigned-cert",
    "pypi-server",
    "sam-cell-seg",
    "sanitize-git-repo",
    # Resource-heavy: large storage / QEMU / ML training
    "qemu-alpine-ssh",
    "qemu-startup",
    "reshard-c4-data",
    "train-fasttext",
]


def _batch_cmd(
    batch: str,
    model: str,
    api_base: str,
    api_key: str,
    extra_headers: str | None,
    jobs_dir: str,
    max_steps: int | None,
    request_timeout_sec: int,
    job_name: str | None,
    n_tasks: int | None = None,
    completed_tasks: list[str] | None = None,
) -> list[str]:
    completed = set(completed_tasks or [])
    if batch == "modal":
        environment, n_concurrent = "modal", 1
        # Explicit include list — filter out already-completed tasks
        task_names = [t for t in MODAL_TASKS if t not in completed]
        exclude_names: list[str] = []
    else:
        environment, n_concurrent = "daytona", 2
        task_names = []
        # Exclude modal tasks + already-completed daytona tasks
        exclude_names = MODAL_TASKS + [t for t in completed if t not in set(MODAL_TASKS)]
    return build_command(
        model=model,
        api_base=api_base,
        api_key=api_key,
        extra_headers=extra_headers,
        jobs_dir=jobs_dir,
        task_names=task_names,
        exclude_task_names=exclude_names,
        n_concurrent=n_concurrent,
        n_tasks=n_tasks,
        max_steps=max_steps,
        request_timeout_sec=request_timeout_sec,
        job_name=f"{job_name}-{batch}" if job_name else None,
        environment=environment,
    )


def _build_batch_cmd(
    batch: str,
    dry_run: bool,
    completed_tasks: list[str] | None = None,
    **kwargs,
) -> list[str] | None:
    cmd = _batch_cmd(batch=batch, completed_tasks=completed_tasks, **kwargs)
    env = "modal" if batch == "modal" else "daytona"
    n = 1 if batch == "modal" else 2
    n_completed = len([t for t in (completed_tasks or []) if (t in MODAL_TASKS) == (batch == "modal")])
    total = len(MODAL_TASKS) if batch == "modal" else "~63"
    resume_note = f"  ({n_completed} already done, skipping)" if n_completed else ""
    print(f"\n{'=' * 60}")
    print(f"Batch: {batch.upper()}  env={env}  n={n}  tasks={total}{resume_note}")
    print(f"{'=' * 60}")
    print("  " + " \\\n    ".join(cmd))
    print()
    if dry_run:
        print(f"[dry-run] Skipping {batch} batch.")
        return None
    return cmd


def run_batch(batch: str, dry_run: bool, completed_tasks: list[str] | None = None, **kwargs) -> int:
    cmd = _build_batch_cmd(batch=batch, dry_run=dry_run, completed_tasks=completed_tasks, **kwargs)
    if cmd is None:
        return 0
    os.chdir(PROJECT_ROOT)
    return subprocess.run(cmd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TB2 eval across Modal (heavy) + Daytona (standard) backends in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        metavar="MODEL",
        help="Model name (e.g. claude-haiku-4-5-20251001, openai/gpt-4o)",
    )
    parser.add_argument(
        "-k",
        "--api-key",
        required=True,
        metavar="KEY",
        help="API key for the model provider",
    )
    parser.add_argument(
        "-b",
        "--api-base",
        required=True,
        metavar="URL",
        help="API base URL (e.g. https://api.anthropic.com, http://127.0.0.1:8061/v1)",
    )
    parser.add_argument(
        "--extra-headers",
        default=None,
        metavar="HEADERS",
        help="Extra HTTP headers as 'Name: Value, Name2: Value2'",
    )
    parser.add_argument(
        "--batch",
        choices=["modal", "daytona", "both"],
        default="both",
        help="Which batch to run (default: both, parallel)",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Stable job name prefix; '-modal' / '-daytona' suffix is appended. Required when using --resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks that already have a result.json. Requires --job-name.",
    )
    parser.add_argument(
        "-l",
        "--n-tasks",
        type=int,
        default=None,
        help="Max tasks per batch (useful for smoke tests)",
    )
    parser.add_argument(
        "-o",
        "--jobs-dir",
        default=JOBS_DIR,
        help=f"Output directory for job results (default: {JOBS_DIR})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max RunLoop steps per task (default: no limit, rely on Harbor timeout)",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=int,
        default=REQUEST_TIMEOUT_SEC,
        help=f"Per-request timeout in seconds (default: {REQUEST_TIMEOUT_SEC})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    if args.resume and not args.job_name:
        parser.error("--resume requires --job-name")

    # Configure Modal credentials before spawning subprocesses, so the Modal
    # SDK can authenticate without browser OAuth.
    if args.batch in ("modal", "both"):
        token_id = os.environ.get("MODAL_TOKEN_ID")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET")
        if not token_id or not token_secret:
            parser.error("Modal batch requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to be set")
        subprocess.run(
            [
                "modal",
                "token",
                "set",
                "--token-id",
                token_id,
                "--token-secret",
                token_secret,
            ],
            check=True,
        )

    jobs_dir_path = PROJECT_ROOT / args.jobs_dir
    jobs_dir = str(jobs_dir_path)

    print(f"Model : {args.model}")
    print(f"Base  : {args.api_base}")
    print(f"Jobs  : {jobs_dir}")

    # ── Resume: compute per-batch completed tasks ─────────────────────────────
    def _batch_completed(batch: str) -> list[str]:
        if not args.resume:
            return []
        job_dir = jobs_dir_path / f"{args.job_name}-{batch}"
        done = _completed_task_names(job_dir)
        if done:
            print(f"[resume] {batch}: {len(done)} completed task(s) in {job_dir} — skipping.")
        return done

    batches = ["modal", "daytona"] if args.batch == "both" else [args.batch]

    common = dict(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        extra_headers=args.extra_headers,
        jobs_dir=jobs_dir,
        n_tasks=args.n_tasks,
        max_steps=args.max_steps,
        request_timeout_sec=args.request_timeout_sec,
        job_name=args.job_name,
        dry_run=args.dry_run,
    )

    if len(batches) == 1:
        rc = run_batch(batch=batches[0], completed_tasks=_batch_completed(batches[0]), **common)
        if rc != 0:
            sys.exit(rc)
    else:
        # Launch both batches in parallel — independent environments (Modal vs Daytona),
        # disjoint task sets, no contention.
        procs: list[tuple[str, subprocess.Popen]] = []
        for batch in batches:
            cmd = _build_batch_cmd(batch=batch, completed_tasks=_batch_completed(batch), **common)
            if cmd is None:
                continue  # dry_run
            print(f"[parallel] Launching {batch} batch...")
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
            print(f"[parallel] {batch} PID={proc.pid}")
            procs.append((batch, proc))

        failed = []
        for batch, proc in procs:
            rc = proc.wait()
            print(f"[parallel] {batch} finished with exit code {rc}")
            if rc != 0:
                failed.append(batch)

        if failed:
            print(f"\n[ERROR] Batches failed: {failed}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
