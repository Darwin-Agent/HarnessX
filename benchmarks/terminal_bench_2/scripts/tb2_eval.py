#!/usr/bin/env python3
# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.terminal_bench_2.defaults import (
    DATASET,
    JOBS_DIR,
    MAX_STEPS,
    N_CONCURRENT,
    REQUEST_TIMEOUT_SEC,
)


def _had_rate_limit_error(trial_dir: Path) -> bool:
    """Return True if the last trace.jsonl in *trial_dir* ended with a 429 error."""
    oh_runs = trial_dir / "agent" / "oh_runs"
    if not oh_runs.is_dir():
        return False
    for trace_file in sorted(oh_runs.glob("*/trace.jsonl")):
        try:
            lines = trace_file.read_text().splitlines()
            for line in reversed(lines):
                event = json.loads(line)
                if event.get("event_type") == "task_end":
                    error = event.get("error") or ""
                    return "429" in error or "RateLimitError" in error or "rate limit" in error.lower()
        except (OSError, json.JSONDecodeError):
            pass
    return False


def _completed_task_names(job_dir: Path) -> list[str]:
    """Return task names that already have a result.json in *job_dir*.

    Tasks whose agent run ended with a 429 rate-limit error are excluded so
    that ``--resume`` re-runs them automatically — unless the task already
    received reward=1.0 (it passed despite the error, no re-run needed).
    """
    completed = []
    if not job_dir.is_dir():
        return completed
    for trial_dir in job_dir.iterdir():
        result_path = trial_dir / "result.json"
        if result_path.is_file():
            try:
                data = json.loads(result_path.read_text())
                task_name = data.get("task_name")
                if not task_name:
                    continue
                reward = (data.get("verifier_result") or {}).get("rewards", {}).get("reward")
                # Re-run tasks that were killed by a rate-limit error and did
                # not already earn a passing reward.
                if reward != 1.0 and _had_rate_limit_error(trial_dir):
                    continue
                completed.append(task_name)
            except (json.JSONDecodeError, OSError):
                pass
    return completed


def build_command(
    model: str,
    api_base: str,
    api_key: str,
    jobs_dir: str,
    task_names: list[str],
    exclude_task_names: list[str],
    n_concurrent: int,
    n_tasks: int | None,
    max_steps: int | None = None,
    extra_headers: str | None = None,
    request_timeout_sec: int | None = None,
    job_name: str | None = None,
    delete_images: bool = False,
    environment: str = "docker",
    sandbox_url: str | None = None,
    proxy_url: str | None = None,
    no_proxy: str | None = None,
    harness_config_yaml: str | None = None,
    verifier_timeout_multiplier: float | None = None,
) -> list[str]:
    cmd = [
        "harbor",
        "run",
        "--dataset",
        DATASET,
        "--agent-import-path",
        "benchmarks.terminal_bench_2:HarnessXAgent",
        "--model",
        model,
        "--jobs-dir",
        jobs_dir,
        "--n-concurrent",
        str(n_concurrent),
        "--ak",
        f"api_key={api_key}",
        "--ak",
        f"api_base={api_base}",
        "--no-force-build",
        "--delete" if delete_images else "--no-delete",
        "--yes",
    ]
    if environment == "opensandbox":
        cmd += [
            "--environment-import-path",
            "benchmarks.terminal_bench_2.opensandbox:OpenSandboxEnvironment",
        ]
        if sandbox_url:
            cmd += ["--ek", f"server_url={sandbox_url}"]
        if proxy_url:
            cmd += ["--ek", f"proxy_url={proxy_url}"]
        if no_proxy:
            cmd += ["--ek", f"no_proxy={no_proxy}"]
    elif environment == "docker":
        cmd += [
            "--environment-import-path",
            "benchmarks.terminal_bench_2.dind_environment:DinDDockerEnvironment",
        ]
        cmd += ["--ek", "network_mode=host"]
    else:
        cmd += ["--env", environment]
    if max_steps is not None:
        cmd += ["--ak", f"max_steps={max_steps}"]

    if job_name:
        cmd += ["--job-name", job_name]
    if extra_headers:
        cmd += ["--ak", f"extra_headers={extra_headers}"]
    if request_timeout_sec is not None:
        cmd += ["--ak", f"request_timeout_sec={request_timeout_sec}"]
    if harness_config_yaml:
        cmd += ["--ak", f"harness_config_yaml={harness_config_yaml}"]
    if verifier_timeout_multiplier is not None:
        cmd += ["--verifier-timeout-multiplier", str(verifier_timeout_multiplier)]

    for name in task_names:
        cmd += ["--task-name", name]
    for name in exclude_task_names:
        cmd += ["--exclude-task-name", name]

    if n_tasks is not None:
        cmd += ["--n-tasks", str(n_tasks)]

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Terminal Bench 2.0 via Harbor + HarnessX",
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
        "--job-name",
        default=None,
        help="Stable job name. Use with --resume to continue the same job.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="JOB_NAME",
        help="Skip tasks already completed in a different job dir "
        "(useful when switching environments). "
        "E.g. --resume-from haiku-full-eval-0406-modal-all",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks that already have a result.json in the job directory.",
    )
    parser.add_argument(
        "-t",
        "--task-name",
        action="append",
        dest="task_names",
        metavar="NAME",
        default=[],
        help="Task name to include (repeatable)",
    )
    parser.add_argument(
        "-n",
        "--n-concurrent",
        type=int,
        default=N_CONCURRENT,
        help=f"Number of concurrent tasks (default: {N_CONCURRENT})",
    )
    parser.add_argument("-l", "--n-tasks", type=int, default=None, help="Maximum number of tasks to run")
    parser.add_argument(
        "-o",
        "--jobs-dir",
        default=JOBS_DIR,
        help=f"Output directory for job results (default: {JOBS_DIR})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Max RunLoop steps per task (default: {MAX_STEPS})",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=int,
        default=REQUEST_TIMEOUT_SEC,
        help=f"Per-request timeout in seconds (default: {REQUEST_TIMEOUT_SEC})",
    )
    parser.add_argument(
        "--delete-images",
        action="store_true",
        default=False,
        help="Delete Docker images after each task.",
    )
    parser.add_argument(
        "--env",
        default="docker",
        choices=["docker", "daytona", "modal", "e2b", "runloop", "gke", "opensandbox"],
        help="Harbor environment backend (default: docker).",
    )
    parser.add_argument(
        "--sandbox-url",
        default=None,
        help="OpenSandbox server URL (only used with --env opensandbox, e.g. http://127.0.0.1:13081).",
    )
    parser.add_argument(
        "--proxy-url",
        default=None,
        help="HTTP proxy injected into every sandbox exec() call "
        "(only used with --env opensandbox, "
        "e.g. http://10.88.0.1:7898).",
    )
    parser.add_argument(
        "--no-proxy",
        default=None,
        help="Comma-separated list of hosts/IPs that bypass the proxy "
        "(injected as no_proxy/NO_PROXY, only used with --env opensandbox, "
        "e.g. '127.0.0.1,10.53.91.141').",
    )
    parser.add_argument(
        "--verifier-timeout-multiplier",
        type=float,
        default=None,
        metavar="MULT",
        help="Multiplier for per-task verifier timeouts (e.g. 2.0 doubles them). "
        "Useful for slow tasks like pytorch-model-recovery that exceed the default 900s.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the harbor command without running it",
    )
    args = parser.parse_args()

    jobs_dir_path = PROJECT_ROOT / args.jobs_dir
    jobs_dir = str(jobs_dir_path)

    # ── Resume: detect already-completed tasks ────────────────────────────────
    exclude_task_names: list[str] = []
    resume_job_name = args.resume_from or (args.job_name if args.resume else None)
    if resume_job_name:
        if args.resume_from:
            job_dir = jobs_dir_path / args.resume_from
        elif args.job_name:
            job_dir = jobs_dir_path / args.job_name
        else:
            subdirs = [d for d in jobs_dir_path.iterdir() if d.is_dir()] if jobs_dir_path.is_dir() else []
            job_dir = max(subdirs, key=lambda d: d.stat().st_mtime) if subdirs else jobs_dir_path / "__none__"
        exclude_task_names = _completed_task_names(job_dir)
        if exclude_task_names:
            print(f"[resume] Found {len(exclude_task_names)} completed task(s) in {job_dir} — skipping them.")
        else:
            print(f"[resume] No completed tasks found in {job_dir} — running all tasks.")

    harness_config_yaml = (os.environ.get("TB2_HARNESS_CONFIG") or "").strip() or None

    cmd = build_command(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        extra_headers=args.extra_headers,
        jobs_dir=jobs_dir,
        task_names=args.task_names,
        exclude_task_names=exclude_task_names,
        n_concurrent=args.n_concurrent,
        n_tasks=args.n_tasks,
        max_steps=args.max_steps,
        request_timeout_sec=args.request_timeout_sec,
        job_name=args.job_name,
        delete_images=args.delete_images,
        environment=args.env,
        sandbox_url=args.sandbox_url,
        proxy_url=args.proxy_url,
        no_proxy=args.no_proxy,
        harness_config_yaml=harness_config_yaml,
        verifier_timeout_multiplier=args.verifier_timeout_multiplier,
    )

    print("Harbor command:")
    print("  " + " \\\n    ".join(cmd))
    print()

    if args.dry_run:
        print("[dry-run] Not executing.")
        return

    print(f"Results will be saved to: {jobs_dir}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.n_concurrent}")
    if args.task_names:
        print(f"Tasks: {', '.join(args.task_names)}")
    else:
        n_desc = f"first {args.n_tasks}" if args.n_tasks else "all"
        print(f"Tasks: {n_desc}")
    print()

    os.chdir(PROJECT_ROOT)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
