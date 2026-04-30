#!/usr/bin/env python3
"""
SWE-bench Lite runner for HarnessX.

Usage:
    python3 -m benchmarks.swebench.run_swebench --model claude-sonnet-4-6 --n 5
    python3 -m benchmarks.swebench.run_swebench --instance-id psf__requests-2317
    python3 -m benchmarks.swebench.run_swebench --repos django sympy --concurrency 5
    python3 -m benchmarks.swebench.run_swebench --retry-no-patch results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Use cached HF datasets
os.environ.pop("HF_DATASETS_OFFLINE", None)
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "sk-placeholder"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from harnessx.core.harness import BaseTask
from harnessx.core.model_config import ModelConfig
from harnessx.providers.openai_provider import OpenAIProvider

from .defaults import (
    CONCURRENCY,
    DATASET_NAME,
    DATASET_SPLIT,
    DEFAULT_WORK_DIR,
    LOCAL_REPOS_DIR,
    MAX_STEPS,
)
from .harness import make_swebench_harness

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("swebench_runner")


# ---------------------------------------------------------------------------
# Instance loader
# ---------------------------------------------------------------------------


@dataclass
class SWEInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    test_patch: str
    patch: str
    version: str
    FAIL_TO_PASS: str
    PASS_TO_PASS: str


def load_instances(n=None, instance_id=None, offset=0, repos=None, instance_ids=None, dataset_name=None):
    name = dataset_name or DATASET_NAME
    try:
        from swebench.harness.utils import load_swebench_dataset

        dataset = load_swebench_dataset(name, split=DATASET_SPLIT)
    except (ImportError, TypeError):
        from datasets import load_dataset

        logger.info(f"Loading dataset {name} via HuggingFace datasets library...")
        ds = load_dataset(name, split=DATASET_SPLIT)
        dataset = [dict(row) for row in ds]
    if instance_ids:
        id_set = set(instance_ids)
        dataset = [d for d in dataset if d["instance_id"] in id_set]
    elif instance_id:
        dataset = [d for d in dataset if d["instance_id"] == instance_id]
    else:
        if repos:
            repo_set = set()
            for r in repos:
                # Support short names like "django" -> "django/django"
                repo_set.update(d["repo"] for d in dataset if r in d["repo"])
            dataset = [d for d in dataset if d["repo"] in repo_set]
        dataset = dataset[offset:]
        if n:
            dataset = dataset[:n]
    return [
        SWEInstance(
            instance_id=d["instance_id"],
            repo=d["repo"],
            base_commit=d["base_commit"],
            problem_statement=d["problem_statement"],
            hints_text=d.get("hints_text", ""),
            test_patch=d.get("test_patch", ""),
            patch=d.get("patch", ""),
            version=d.get("version", ""),
            FAIL_TO_PASS=d.get("FAIL_TO_PASS", ""),
            PASS_TO_PASS=d.get("PASS_TO_PASS", ""),
        )
        for d in dataset
    ]


# ---------------------------------------------------------------------------
# Repo setup — use GitHub archive tarball (faster than git clone)
# ---------------------------------------------------------------------------

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "swe",
    "GIT_AUTHOR_EMAIL": "s@s",
    "GIT_COMMITTER_NAME": "swe",
    "GIT_COMMITTER_EMAIL": "s@s",
}


def _git_init_repo(repo_dir: str) -> None:
    """Turn a directory into a git repo with a single 'initial' commit."""
    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, timeout=10)
    subprocess.run(["git", "add", "-A"], cwd=repo_dir, capture_output=True, timeout=60)
    subprocess.run(
        ["git", "commit", "-m", "initial", "--allow-empty"],
        cwd=repo_dir,
        capture_output=True,
        timeout=30,
        env=_GIT_ENV,
    )


def setup_repo(instance: SWEInstance, work_dir: str) -> str:
    repo_dir = os.path.join(work_dir, instance.instance_id.replace("/", "_"))
    os.makedirs(work_dir, exist_ok=True)

    # Try reusing existing dir
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        try:
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=repo_dir,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(["git", "clean", "-fdx"], cwd=repo_dir, capture_output=True, timeout=30)
            # Try checking out base_commit (works for full clones)
            result = subprocess.run(
                ["git", "checkout", instance.base_commit],
                cwd=repo_dir,
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info(f"Reused {repo_dir} -> {instance.base_commit[:8]}")
                return repo_dir
            # For tarball-based repos (no full history), reset to initial commit
            result2 = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=repo_dir,
                capture_output=True,
                timeout=10,
            )
            if result2.returncode == 0:
                logger.info(f"Reused tarball repo {repo_dir} (reset to clean state)")
                return repo_dir
        except Exception:
            pass
        shutil.rmtree(repo_dir, ignore_errors=True)

    # Fast path: copy from pre-downloaded local repo cache
    local_src = os.path.join(LOCAL_REPOS_DIR, instance.instance_id)
    if os.path.isdir(os.path.join(local_src, ".git")):
        r = subprocess.run(
            ["cp", "-a", local_src + "/.", repo_dir + "/"],
            capture_output=True,
            timeout=120,
        )
        if r.returncode == 0:
            subprocess.run(["git", "checkout", "--", "."], cwd=repo_dir, capture_output=True, timeout=30)
            subprocess.run(["git", "clean", "-fdx"], cwd=repo_dir, capture_output=True, timeout=30)
            logger.info(f"Copied from local cache: {instance.instance_id}")
            return repo_dir

    # If dir exists without .git (e.g., previous failed setup), re-init git
    if os.path.isdir(repo_dir) and os.listdir(repo_dir):
        logger.info(f"Re-initializing git in existing {repo_dir}")
        _git_init_repo(repo_dir)
        return repo_dir

    # Download source via GitHub tarball API and set up as git repo
    tarball_url = f"https://github.com/{instance.repo}/archive/{instance.base_commit}.tar.gz"
    logger.info(f"Downloading {tarball_url}...")

    os.makedirs(repo_dir, exist_ok=True)
    dl_result = None
    for dl_attempt in range(5):
        dl_result = subprocess.run(
            ["curl", "-sL", "--max-time", "120", tarball_url],
            capture_output=True,
            timeout=180,
        )
        if dl_result.returncode == 0 and len(dl_result.stdout) >= 100:
            break
        delay = 5 * (2**dl_attempt)
        logger.warning(
            f"Download attempt {dl_attempt + 1}/5 failed for {instance.instance_id} "
            f"(got {len(dl_result.stdout)} bytes), retrying in {delay}s..."
        )
        time.sleep(delay)
    if dl_result.returncode != 0 or len(dl_result.stdout) < 100:
        raise RuntimeError(f"Failed to download tarball after 5 attempts: {dl_result.stderr.decode()[:200]}")

    # Extract tarball
    extract_result = subprocess.run(
        ["tar", "xzf", "-", "--strip-components=1", "-C", repo_dir],
        input=dl_result.stdout,
        capture_output=True,
        timeout=120,
    )
    if extract_result.returncode != 0:
        raise RuntimeError(f"Failed to extract: {extract_result.stderr.decode()[:200]}")

    _git_init_repo(repo_dir)
    logger.info(f"Set up {repo_dir} from tarball ({instance.base_commit[:8]})")
    return repo_dir


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def extract_patch(state_messages: list, repo_dir: str = "") -> str:
    # First try: find git diff in tool output
    for msg in reversed(state_messages):
        role = getattr(msg, "role", "")
        content = getattr(msg, "content", "") or ""
        name = getattr(msg, "name", "") or ""
        if role == "tool" and name == "Bash" and content:
            if "diff --git" in content or content.strip().startswith("diff --git"):
                return content.strip()
    # Second try: find git diff in assistant message
    for msg in reversed(state_messages):
        role = getattr(msg, "role", "")
        content = getattr(msg, "content", "") or ""
        if role == "assistant" and content and "diff --git" in content:
            start = content.index("diff --git")
            return content[start:].strip()
    # Fallback: run git diff directly on the repo
    if repo_dir and os.path.isdir(os.path.join(repo_dir, ".git")):
        try:
            r = subprocess.run(
                ["git", "diff"],
                cwd=repo_dir,
                capture_output=True,
                timeout=30,
                text=True,
            )
            if r.returncode == 0 and r.stdout.strip():
                logger.info(f"Extracted patch via fallback git diff ({len(r.stdout.splitlines())} lines)")
                return r.stdout.strip()
        except Exception:
            pass
    return ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_single_instance(
    instance,
    model,
    provider_kwargs,
    work_dir,
    max_steps=MAX_STEPS,
    results_dir="swebench_results",
    api_base=None,
    api_key=None,
    max_tokens=16384,
):
    t0 = time.time()
    result = {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "model": model,
        "status": "error",
        "patch": "",
        "exit_reason": "",
        "steps": 0,
        "tokens": 0,
        "cost_usd": 0.0,
        "time_sec": 0.0,
        "error": "",
    }
    try:
        repo_dir = setup_repo(instance, work_dir)

        fail_to_pass = instance.FAIL_TO_PASS or ""
        test_section = ""
        if fail_to_pass:
            test_section = f"""
## Failing Tests

The following tests should PASS after your fix:
{fail_to_pass}
"""

        task_desc = f"""Fix the following GitHub issue in the repository at {repo_dir}:

## Issue: {instance.instance_id}

{instance.problem_statement}
{test_section}
## Instructions

1. Explore the repository structure to understand the codebase
2. Find the relevant source files related to this issue
3. Understand the root cause of the bug
4. Make the minimal fix needed
5. Verify the fix by running relevant tests if possible
6. IMPORTANT: As your final action, run `cd {repo_dir} && git diff` to output the patch
"""
        task = BaseTask(
            description=task_desc,
            max_steps=max_steps,
            metadata={"instance_id": instance.instance_id},
        )
        extra_headers = provider_kwargs.get("extra_headers", {})
        provider = OpenAIProvider(
            model=model,
            base_url=api_base,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            extra_headers=extra_headers if extra_headers else None,
            max_tokens=max_tokens,
        )
        harness_config = make_swebench_harness(
            repo_dir,
            logs_dir=os.path.join(results_dir, "runs"),
            max_steps=max_steps,
        )
        harness = ModelConfig(main=provider).agentic(harness_config)
        harness_result = await harness.run(task)
        patch = extract_patch(harness_result.resume_state.messages, repo_dir)
        result.update(
            {
                "status": "completed",
                "patch": patch,
                "exit_reason": harness_result.exit_reason,
                "steps": harness_result.total_steps,
                "tokens": harness_result.total_tokens,
                "cost_usd": harness_result.total_cost_usd,
                "has_patch": bool(patch),
                "patch_lines": len(patch.splitlines()) if patch else 0,
                "final_output": (harness_result.final_output or "")[:500],
            }
        )
    except Exception as e:
        logger.error(f"Error running {instance.instance_id}: {e}", exc_info=True)
        result["error"] = str(e)
    finally:
        result["time_sec"] = time.time() - t0
        # Clean up work dir after run to free disk space (important when work_dir is /tmp)
        if repo_dir and os.path.isdir(repo_dir):
            try:
                shutil.rmtree(repo_dir, ignore_errors=True)
            except Exception:
                pass
    return result


async def run_batch(
    instances,
    model,
    provider_kwargs,
    work_dir,
    max_steps=MAX_STEPS,
    concurrency=CONCURRENCY,
    results_dir="swebench_results",
    api_base=None,
    api_key=None,
    max_tokens=16384,
):
    os.makedirs(results_dir, exist_ok=True)

    # Skip pre-download when local repos are available — copy on demand is faster
    # (only concurrency repos exist simultaneously, saving disk space)
    if os.path.isdir(LOCAL_REPOS_DIR):
        logger.info(f"Local repos available at {LOCAL_REPOS_DIR} — skipping pre-download, will copy on demand")
    else:
        # Pre-download repos with controlled concurrency to avoid GitHub rate limits
        to_download = []
        for inst in instances:
            rd = os.path.join(work_dir, inst.instance_id.replace("/", "_"))
            if os.path.isdir(os.path.join(rd, ".git")):
                continue
            if os.path.isdir(rd) and os.listdir(rd):
                continue
            to_download.append(inst)
        logger.info(f"Pre-downloading {len(to_download)} repos ({len(instances) - len(to_download)} already cached)...")
        if to_download:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            done_count = 0
            failed_count = 0
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(setup_repo, inst, work_dir): inst for inst in to_download}
                for fut in as_completed(futures):
                    inst = futures[fut]
                    try:
                        fut.result()
                        done_count += 1
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"  Pre-download failed for {inst.instance_id}: {e}")
                    if (done_count + failed_count) % 20 == 0:
                        logger.info(f"  Pre-downloaded {done_count}/{len(to_download)} repos ({failed_count} failed)")
            logger.info(f"Pre-download complete: {done_count} OK, {failed_count} failed.")

    sem = asyncio.Semaphore(concurrency)

    async def run_with_sem(inst):
        async with sem:
            logger.info(f"▶ Running {inst.instance_id}...")
            r = await run_single_instance(
                inst,
                model,
                provider_kwargs,
                work_dir,
                max_steps,
                results_dir,
                api_base=api_base,
                api_key=api_key,
                max_tokens=max_tokens,
            )
            logger.info(
                f"{'✓' if r.get('has_patch') else '✗'} {inst.instance_id} "
                f"[{r['status']}] {r.get('steps', 0)} steps, "
                f"${r.get('cost_usd', 0):.3f}, {r['time_sec']:.0f}s"
            )
            return r

    results = await asyncio.gather(*[run_with_sem(i) for i in instances], return_exceptions=True)
    final = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            final.append(
                {
                    "instance_id": instances[i].instance_id,
                    "status": "error",
                    "error": str(r),
                }
            )
        else:
            final.append(r)

    ts = int(time.time())
    results_file = os.path.join(results_dir, f"results_{ts}.json")
    with open(results_file, "w") as f:
        json.dump(final, f, indent=2)
    logger.info(f"Results written to {results_file}")

    completed = [r for r in final if r["status"] == "completed"]
    with_patch = [r for r in completed if r.get("has_patch")]
    errors = [r for r in final if r["status"] == "error"]
    print(f"\n{'=' * 60}")
    print("SWE-bench Lite Results Summary")
    print(f"{'=' * 60}")
    print(f"Total: {len(final)} | Completed: {len(completed)} | With patch: {len(with_patch)} | Errors: {len(errors)}")
    if completed:
        print(
            f"Avg steps: {sum(r['steps'] for r in completed) / len(completed):.1f} | "
            f"Avg cost: ${sum(r['cost_usd'] for r in completed) / len(completed):.3f} | "
            f"Avg time: {sum(r['time_sec'] for r in completed) / len(completed):.0f}s | "
            f"Total tokens: {sum(r['tokens'] for r in completed):,}"
        )
    print(f"{'=' * 60}")
    for r in final:
        icon = "✓" if r.get("has_patch") else "✗"
        print(f"  {icon} {r['instance_id']}: {r['status']} ({r.get('steps', 0)} steps, ${r.get('cost_usd', 0):.3f})")
    return final


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench Verified with HarnessX")
    parser.add_argument("--model", default="pa/claude-sonnet-4-6")
    parser.add_argument("--provider-id", default="")
    parser.add_argument("--api-base", default=None, help="Custom OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key (fallback: OPENAI_API_KEY env)")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max output tokens per completion")
    parser.add_argument("--dataset", default=DATASET_NAME, help="Dataset name (default: SWE-bench_Verified)")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--trials", type=int, default=1, help="Number of independent trials per instance")
    parser.add_argument(
        "--repos",
        nargs="*",
        default=None,
        help="Filter by repo names (e.g., django sympy)",
    )
    parser.add_argument(
        "--retry-no-patch",
        default="",
        help="Path to results JSON — re-run instances that had no patch",
    )
    parser.add_argument(
        "--instance-ids-file",
        default="",
        help="Path to JSON file with list of instance IDs to run",
    )
    parser.add_argument(
        "--exclude-ids-file",
        default="",
        help="Path to JSON file with list of instance IDs to SKIP",
    )
    args = parser.parse_args()

    provider_kwargs = {}
    if args.provider_id:
        provider_kwargs["extra_headers"] = {"X-Model-Provider-Id": args.provider_id}
    results_dir = args.results_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    retry_ids = None
    if args.instance_ids_file:
        retry_ids = json.load(open(args.instance_ids_file))
        logger.info(f"Running {len(retry_ids)} instances from {args.instance_ids_file}")
    elif args.retry_no_patch:
        retry_data = json.load(open(args.retry_no_patch))
        retry_ids = [r["instance_id"] for r in retry_data if r.get("status") == "completed" and not r.get("has_patch")]
        logger.info(f"Retrying {len(retry_ids)} no-patch instances from {args.retry_no_patch}")
    instances = load_instances(
        n=args.n if not args.instance_id else None,
        instance_id=args.instance_id or None,
        offset=args.offset,
        repos=args.repos,
        instance_ids=retry_ids,
        dataset_name=args.dataset,
    )
    if args.exclude_ids_file:
        exclude_ids = set(json.load(open(args.exclude_ids_file)))
        instances = [inst for inst in instances if inst.instance_id not in exclude_ids]
        logger.info(f"After exclusion: {len(instances)} instances")
    if not instances:
        logger.error("No instances found!")
        sys.exit(1)

    for trial in range(1, args.trials + 1):
        trial_dir = os.path.join(results_dir, f"trial_{trial}") if args.trials > 1 else results_dir
        logger.info(f"=== Trial {trial}/{args.trials}: {len(instances)} instances with model={args.model} ===")
        asyncio.run(
            run_batch(
                instances,
                args.model,
                provider_kwargs,
                args.work_dir,
                args.max_steps,
                args.concurrency,
                trial_dir,
                api_base=args.api_base,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
            )
        )

    if args.trials > 1:
        print(f"\nCompleted {args.trials} trials. Results in {results_dir}/trial_*/")
        print("Run evaluation on each trial directory for per-trial scores.")


if __name__ == "__main__":
    main()
