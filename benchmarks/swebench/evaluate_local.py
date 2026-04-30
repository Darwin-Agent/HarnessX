#!/usr/bin/env python3
"""Local SWE-bench evaluation without Docker.

Applies patches to repos, runs tests, and grades results.
Works with tarball repos at the path set by SWEBENCH_REPOS_DIR (default: /tmp/swebench-repos).

Usage:
    python -m benchmarks.swebench.evaluate_local benchmarks/swebench/results/merged_v3v4/results.json
    python -m benchmarks.swebench.evaluate_local results.json --max-workers 4 --instance-ids django__django-11039
    python -m benchmarks.swebench.evaluate_local results.json --repos django  # only django instances
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

# Load dataset from arrow cache (avoids network dependency)
_DATASET_CACHE = None


def _load_dataset() -> dict:
    global _DATASET_CACHE
    if _DATASET_CACHE is not None:
        return _DATASET_CACHE

    import pyarrow as pa

    arrow_path = (
        "/root/.cache/huggingface/datasets/princeton-nlp___swe-bench_lite"
        "/default/0.0.0/6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"
        "/swe-bench_lite-test.arrow"
    )
    reader = pa.ipc.open_stream(arrow_path)
    table = reader.read_all()
    instances = {}
    for i in range(len(table)):
        row = {col: table[col][i].as_py() for col in table.column_names}
        instances[row["instance_id"]] = row
    _DATASET_CACHE = instances
    return instances


def _recalculate_hunk_headers(patch: str) -> str:
    """Recalculate @@ hunk headers to match actual line counts in the patch body.

    Fixes patches where the @@ header claims N old/new lines but the body has
    a different count (caused by truncation, _clean_patch corruption, or model errors).
    """
    lines = patch.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)", line)
        if m:
            old_start = int(m.group(1))
            new_start = int(m.group(3))
            suffix = m.group(5)
            # Count actual lines in the hunk body
            old_count = 0
            new_count = 0
            j = i + 1
            while j < len(lines):
                hl = lines[j]
                if hl.startswith(" "):
                    old_count += 1
                    new_count += 1
                elif hl.startswith("-"):
                    old_count += 1
                elif hl.startswith("+"):
                    new_count += 1
                elif hl.startswith("\\"):
                    pass  # "\ No newline at end of file"
                elif hl == "":
                    # Empty line inside hunk = context line
                    # But check if next line is still a hunk line
                    if j + 1 < len(lines) and lines[j + 1].startswith(("+", "-", " ", "\\")):
                        old_count += 1
                        new_count += 1
                    else:
                        break
                else:
                    break
                j += 1
            result.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}")
        else:
            result.append(line)
        i += 1
    return "\n".join(result)


def _fix_malformed_patch(patch: str, repo_dir: str) -> str:
    """Fix patches with bare @@ hunk headers by computing line numbers."""
    lines = patch.split("\n")
    fixed = []
    current_file = None

    for i, line in enumerate(lines):
        if line.startswith("--- a/"):
            current_file = line[6:]
            fixed.append(line)
            continue
        if line.startswith("+++ b/"):
            fixed.append(line)
            continue

        # Fix bare @@ header or symbolic @@ (e.g., "@@ class Foo:")
        is_bare = line.strip() == "@@"
        is_symbolic = line.startswith("@@") and not re.match(r"^@@ -\d+", line)
        if (is_bare or is_symbolic) and current_file:
            # Look at the next lines to find old content
            remaining = lines[i + 1 :]
            old_lines = []
            new_lines = []
            for rl in remaining:
                if rl.startswith("-"):
                    old_lines.append(rl[1:])
                elif rl.startswith("+"):
                    new_lines.append(rl[1:])
                elif rl.startswith(" "):
                    old_lines.append(rl[1:])
                    new_lines.append(rl[1:])
                elif rl.startswith("diff --git") or rl.startswith("@@"):
                    break

            # Find the old content in the file
            file_path = os.path.join(repo_dir, current_file)
            start_line = 1
            if old_lines and os.path.isfile(file_path):
                with open(file_path) as f:
                    file_content = f.readlines()
                # Search for the first old line (whitespace-tolerant)
                target = old_lines[0].rstrip()
                for fi, fc in enumerate(file_content):
                    if fc.rstrip() == target:
                        # Verify more lines match
                        match = True
                        for k, ol in enumerate(old_lines[1:], 1):
                            if fi + k >= len(file_content):
                                match = False
                                break
                            if file_content[fi + k].rstrip() != ol.rstrip():
                                match = False
                                break
                        if match:
                            start_line = fi + 1
                            break
                        # Still use first line match as fallback
                        if start_line == 1:
                            start_line = fi + 1

            old_count = len(old_lines)
            new_count = len(new_lines)
            fixed.append(f"@@ -{start_line},{old_count} +{start_line},{new_count} @@")
            continue

        fixed.append(line)

    return "\n".join(fixed)


def _relocate_hunks(patch: str, repo_dir: str) -> str:
    """Try to relocate hunks by searching for context/removed lines in the actual file.

    When a hunk fails because the line numbers are wrong (code moved), this
    searches the file for the actual location of the old lines and rewrites
    the @@ header with the correct start line.
    """
    lines = patch.split("\n")
    result = []
    current_file = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- a/"):
            current_file = line[6:]
            result.append(line)
            i += 1
            continue
        if line.startswith("+++ b/"):
            result.append(line)
            i += 1
            continue

        m = re.match(r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)", line)
        if m and current_file:
            old_start = int(m.group(1))
            _old_count_str = m.group(2)
            suffix = m.group(5)

            # Collect old lines (context + removed) from the hunk
            old_lines = []
            j = i + 1
            while j < len(lines):
                hl = lines[j]
                if hl.startswith(" "):
                    old_lines.append(hl[1:])
                elif hl.startswith("-"):
                    old_lines.append(hl[1:])
                elif hl.startswith("+") or hl.startswith("\\"):
                    pass
                elif hl == "":
                    old_lines.append("")
                else:
                    break
                j += 1

            # Try to find the old lines in the actual file
            file_path = os.path.join(repo_dir, current_file)
            if old_lines and os.path.isfile(file_path):
                try:
                    with open(file_path) as f:
                        file_lines = [line.rstrip("\n") for line in f.readlines()]
                except Exception:
                    file_lines = []

                if file_lines:
                    best_offset = None
                    best_score = 0
                    # Search within a reasonable range
                    for offset in range(len(file_lines)):
                        if offset + len(old_lines) > len(file_lines):
                            break
                        score = sum(
                            1 for k, ol in enumerate(old_lines) if file_lines[offset + k].rstrip() == ol.rstrip()
                        )
                        if score > best_score:
                            best_score = score
                            best_offset = offset
                        if score == len(old_lines):
                            break  # Perfect match

                    # Accept if >70% of lines match
                    if best_offset is not None and best_score >= max(1, len(old_lines) * 0.7):
                        new_start_line = best_offset + 1
                        if new_start_line != old_start:
                            # Recalculate new_start with same delta
                            new_start = int(m.group(3))
                            delta = new_start_line - old_start
                            new_new_start = new_start + delta
                            old_count = m.group(2) or str(len(old_lines))
                            new_count = m.group(4) or old_count
                            result.append(f"@@ -{new_start_line},{old_count} +{new_new_start},{new_count} @@{suffix}")
                            i += 1
                            continue

            result.append(line)
        else:
            result.append(line)
        i += 1
    return "\n".join(result)


def _write_patch_file(patch: str) -> str:
    """Write patch to a temp file and return path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch)
        return f.name


def _try_apply(patch_file: str, repo_dir: str, method: str) -> tuple[bool, str]:
    """Try a single patch application method. Returns (success, error)."""
    try:
        if method == "git_strict":
            result = subprocess.run(
                ["git", "apply", "--allow-empty", patch_file],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
        elif method == "git_lenient":
            result = subprocess.run(
                ["git", "apply", "--allow-empty", "-C0", "--no-check", patch_file],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
        elif method == "git_3way":
            result = subprocess.run(
                ["git", "apply", "--allow-empty", "--3way", patch_file],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
        elif method == "patch_fuzz":
            result = subprocess.run(
                [
                    "patch",
                    "-p1",
                    "-i",
                    patch_file,
                    "--forward",
                    "--no-backup-if-mismatch",
                    "-l",
                    "--fuzz=3",
                ],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
        elif method == "patch_default":
            result = subprocess.run(
                [
                    "patch",
                    "-p1",
                    "-i",
                    patch_file,
                    "--forward",
                    "--no-backup-if-mismatch",
                    "-l",
                ],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
        else:
            return False, f"Unknown method: {method}"

        if result.returncode == 0:
            return True, ""
        return False, f"{result.stderr[:200]} | {result.stdout[:200]}"
    except Exception as e:
        return False, str(e)


def _apply_patch(patch: str, repo_dir: str) -> tuple[bool, str]:
    """Apply patch to repo with progressive repair strategies.

    Order:
    1. Fix known malformed patterns (bare @@, symbolic @@)
    2. Recalculate hunk headers (fix line count mismatches)
    3. Try git apply (strict)
    4. Try git apply (lenient, -C0)
    5. Try git apply (3-way merge)
    6. Try relocated hunks (search file for correct line numbers)
    7. Try patch command with fuzz=3
    """
    # Step 1: Fix bare/symbolic @@ headers
    if re.search(r"^@@(?:\s.*[^@])?$", patch, re.MULTILINE):
        if re.search(r"^@@\s*$", patch, re.MULTILINE) or re.search(r"^@@\s+(?!-)(?!.*@@\s*$)", patch, re.MULTILINE):
            patch = _fix_malformed_patch(patch, repo_dir)

    # Step 2: Recalculate hunk headers to fix count mismatches
    patch = _recalculate_hunk_headers(patch)

    patch_file = _write_patch_file(patch)
    try:
        # Step 3: git apply (strict)
        ok, err = _try_apply(patch_file, repo_dir, "git_strict")
        if ok:
            return True, ""

        # Step 4: git apply (lenient)
        ok, err = _try_apply(patch_file, repo_dir, "git_lenient")
        if ok:
            return True, ""

        # Step 5: git apply --3way
        ok, err = _try_apply(patch_file, repo_dir, "git_3way")
        if ok:
            return True, ""
    finally:
        os.unlink(patch_file)

    # Step 6: Try relocating hunks (search for correct line numbers)
    relocated = _relocate_hunks(patch, repo_dir)
    if relocated != patch:
        relocated = _recalculate_hunk_headers(relocated)
        patch_file = _write_patch_file(relocated)
        try:
            ok, err = _try_apply(patch_file, repo_dir, "git_strict")
            if ok:
                return True, ""
            ok, err = _try_apply(patch_file, repo_dir, "git_lenient")
            if ok:
                return True, ""
        finally:
            os.unlink(patch_file)

    # Step 7: patch command with fuzz (last resort)
    patch_file = _write_patch_file(patch)
    try:
        ok, err = _try_apply(patch_file, repo_dir, "patch_fuzz")
        if ok:
            return True, ""
        return False, f"Apply failed: {err}"
    finally:
        os.unlink(patch_file)


def _has_pytest_timeout(repo_dir: str) -> bool:
    """Check if pytest-timeout plugin is available in the repo's pytest."""
    _result = subprocess.run(
        ["python3", "-m", "pytest", "--co", "-q", "--override-ini=addopts="],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=10,
    )
    # If pytest-timeout is installed, --timeout will be recognized
    check = subprocess.run(
        ["python3", "-m", "pytest", "--timeout=1", "--co", "-q"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return check.returncode != 4  # rc=4 means usage error


def _get_test_command(repo: str, tests: list[str], repo_dir: str, instance_data: dict | None = None) -> list[str]:
    """Get the test command for a repo."""
    if "django" in repo:
        # Django uses its own test runner
        # Convert FAIL_TO_PASS format: "test_method (module.TestClass)" → "module.TestClass.test_method"
        django_tests = []
        for t in tests:
            m = re.match(r"^(\S+)\s+\(([^)]+)\)$", t)
            if m:
                method, dotpath = m.groups()
                django_tests.append(f"{dotpath}.{method}")
            else:
                django_tests.append(t)
        return [
            "python3",
            "-W",
            "ignore",
            "tests/runtests.py",
            "--verbosity",
            "2",
            "--noinput",
            "--parallel",
            "1",
        ] + django_tests
    elif "sympy" in repo:
        # SymPy FAIL_TO_PASS often gives bare function names like 'test_Derivative'.
        # We use the test_patch to find which files contain the test functions.
        test_files_from_patch = set()
        if instance_data:
            tp = instance_data.get("test_patch", "")
            for line in tp.split("\n"):
                if line.startswith("diff --git"):
                    parts = line.split(" b/")
                    if len(parts) > 1:
                        fpath = parts[1].strip()
                        if "/tests/" in fpath and fpath.endswith(".py"):
                            test_files_from_patch.add(fpath)

        test_args = []
        for t in tests:
            if "/" in t or "::" in t:
                test_args.append(t)
            else:
                # First try to match against test_patch files (most reliable)
                matched = False
                for tf in test_files_from_patch:
                    full_path = os.path.join(repo_dir, tf)
                    if os.path.isfile(full_path):
                        try:
                            with open(full_path) as fobj:
                                content = fobj.read()
                            if f"def {t}(" in content:
                                test_args.append(f"{tf}::{t}")
                                matched = True
                                break
                        except Exception:
                            pass

                if not matched:
                    # Fallback: grep for the test function in test dirs
                    result = subprocess.run(
                        ["grep", "-rl", f"def {t}(", "--include=*.py", "."],
                        cwd=repo_dir,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.stdout.strip():
                        # Prefer files with '/tests/' in the path
                        candidates = result.stdout.strip().split("\n")
                        test_file = next((c for c in candidates if "/tests/" in c), candidates[0])
                        test_args.append(f"{test_file}::{t}")
                    else:
                        test_args.append(t)
        return ["python3", "-m", "pytest", "-xvs"] + test_args
    elif "scikit-learn" in repo:
        return ["python3", "-m", "pytest", "-xvs"] + tests
    elif "pytest" in repo:
        return ["python3", "-m", "pytest", "-xvs"] + tests
    elif "sphinx" in repo:
        return ["python3", "-m", "pytest", "-xvs"] + tests
    elif "matplotlib" in repo:
        return ["python3", "-m", "pytest", "-xvs"] + tests
    else:
        return ["python3", "-m", "pytest", "-xvs"] + tests


def _reset_repo(repo_dir: str) -> bool:
    """Reset repo to initial state. Returns True if successful."""
    try:
        has_commits = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                capture_output=True,
                timeout=5,
            ).returncode
            == 0
        )

        if has_commits:
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=repo_dir,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True, timeout=10)
        return True
    except Exception as e:
        logger.warning("Failed to reset %s: %s", repo_dir, e)
        return False


# Python 3.10+ removed several aliases from `collections` (Mapping, MutableMapping,
# Callable, etc.) that older library code still imports.  We restore them at the
# subprocess level via a sitecustomize.py injected into PYTHONPATH.
_PY3_COMPAT_CODE = """\
import collections, collections.abc, sys
_aliases = [
    "Awaitable","Coroutine","AsyncIterable","AsyncIterator","AsyncGenerator",
    "Hashable","Iterable","Iterator","Generator","Reversible","Container",
    "Collection","Callable","Set","MutableSet","Mapping","MutableMapping",
    "MappingView","KeysView","ItemsView","ValuesView","Sequence",
    "MutableSequence","ByteString",
]
for _a in _aliases:
    if not hasattr(collections, _a) and hasattr(collections.abc, _a):
        setattr(collections, _a, getattr(collections.abc, _a))
"""


def _inject_py3_compat(repo_dir: str) -> str | None:
    """Write a sitecustomize.py into a temp dir for Python 3.10+ compat.

    Returns the temp dir path (to prepend to PYTHONPATH), or None on failure.
    """
    try:
        compat_dir = os.path.join(repo_dir, ".swebench_compat")
        os.makedirs(compat_dir, exist_ok=True)
        with open(os.path.join(compat_dir, "sitecustomize.py"), "w") as f:
            f.write(_PY3_COMPAT_CODE)
        return compat_dir
    except Exception:
        return None


def _remove_py3_compat(repo_dir: str) -> None:
    """Clean up the injected compat dir."""
    compat_dir = os.path.join(repo_dir, ".swebench_compat")
    shutil.rmtree(compat_dir, ignore_errors=True)


def evaluate_instance(
    instance_id: str,
    patch: str,
    repo_dir: str,
    instance_data: dict,
    timeout: int = 120,
) -> dict:
    """Evaluate a single instance locally.

    SWE-bench evaluation order:
    1. Reset repo to clean state
    2. Apply test_patch (updates tests to expect the fix)
    3. Apply model's patch (the code fix)
    4. Run FAIL_TO_PASS tests — they should now pass
    """
    result = {"instance_id": instance_id, "resolved": False}

    # Parse FAIL_TO_PASS tests
    fail_to_pass_raw = instance_data.get("FAIL_TO_PASS", "")
    if isinstance(fail_to_pass_raw, str):
        try:
            tests = json.loads(fail_to_pass_raw)
        except json.JSONDecodeError:
            tests = [t.strip() for t in fail_to_pass_raw.split(",") if t.strip()]
    else:
        tests = list(fail_to_pass_raw or [])

    if not tests:
        result["error"] = "No FAIL_TO_PASS tests"
        return result

    # Reset repo
    if not _reset_repo(repo_dir):
        result["error"] = "Reset failed"
        return result

    # Step 1: Apply test_patch first (this updates tests to expect the fix)
    test_patch = instance_data.get("test_patch", "")
    if test_patch:
        success, error = _apply_patch(test_patch, repo_dir)
        if not success:
            result["error"] = f"Test patch: {error}"
            return result

    # Step 2: Apply model's code patch
    success, error = _apply_patch(patch, repo_dir)
    if not success:
        result["error"] = f"Patch: {error}"
        return result

    # Verify patches applied (git diff should show changes)
    diff_result = subprocess.run(
        ["git", "diff", "--stat"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if not diff_result.stdout.strip():
        result["error"] = "Patches applied but no changes detected"
        return result

    result["files_changed"] = diff_result.stdout.strip()

    # Run tests
    repo = instance_data.get("repo", "")
    test_cmd = _get_test_command(repo, tests, repo_dir, instance_data)

    # Inject Python 3.10+ compat (collections.Mapping etc.)
    compat_dir = _inject_py3_compat(repo_dir)

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{compat_dir}:{repo_dir}" if compat_dir else repo_dir,
    }
    # Django: don't set DJANGO_SETTINGS_MODULE — runtests.py configures its own settings

    try:
        test_result = subprocess.run(
            test_cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        stdout = (test_result.stdout or "")[-3000:]
        stderr = (test_result.stderr or "")[-1000:]
        result["test_output"] = (stdout + "\n" + stderr)[-2000:]
        result["test_returncode"] = test_result.returncode
        result["resolved"] = test_result.returncode == 0
    except subprocess.TimeoutExpired:
        result["error"] = f"Tests timed out ({timeout}s)"
        result["test_output"] = "TIMEOUT"
    except Exception as e:
        result["error"] = f"Test error: {e}"

    # Clean up compat dir and reset repo after evaluation
    _remove_py3_compat(repo_dir)
    _reset_repo(repo_dir)

    return result


def _eval_worker(args: tuple) -> dict:
    """Worker function for parallel evaluation."""
    instance_id, patch, repo_dir, instance_data, timeout = args
    try:
        return evaluate_instance(instance_id, patch, repo_dir, instance_data, timeout)
    except Exception as e:
        return {
            "instance_id": instance_id,
            "resolved": False,
            "error": f"Worker error: {e}",
        }


def main():
    parser = argparse.ArgumentParser(description="Local SWE-bench evaluation")
    parser.add_argument("results", help="Path to results JSON")
    parser.add_argument("--instance-ids", nargs="+", default=None)
    parser.add_argument(
        "--repos",
        nargs="+",
        default=None,
        help="Only evaluate instances from these repos (substring match)",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel workers (use 1 for shared repos)",
    )
    parser.add_argument("--base-dir", default=os.environ.get("SWEBENCH_REPOS_DIR", "/tmp/swebench-repos"))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of instances to evaluate")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Load results and dataset
    results = json.load(open(args.results))
    dataset = _load_dataset()

    # Filter
    eval_items = []
    for r in results:
        iid = r["instance_id"]
        if args.instance_ids and iid not in args.instance_ids:
            continue
        if args.repos:
            if not any(repo in iid for repo in args.repos):
                continue
        if not r.get("patch", "").strip():
            continue

        inst = dataset.get(iid)
        if not inst:
            logger.warning("Instance %s not in dataset", iid)
            continue

        safe_id = iid.replace("/", "__")
        repo_dir = os.path.join(args.base_dir, safe_id)
        if not os.path.isdir(repo_dir) or len(os.listdir(repo_dir)) < 3:
            logger.warning("Repo dir missing/empty: %s", repo_dir)
            continue

        eval_items.append((iid, r["patch"], repo_dir, inst, args.timeout))

    if args.limit:
        eval_items = eval_items[: args.limit]

    logger.info("Evaluating %d instances...", len(eval_items))

    # Run evaluations
    eval_results = []
    if args.max_workers <= 1:
        for i, item in enumerate(eval_items):
            iid = item[0]
            logger.info("[%d/%d] Evaluating %s...", i + 1, len(eval_items), iid)
            er = _eval_worker(item)
            eval_results.append(er)
            status = "RESOLVED" if er.get("resolved") else "FAILED"
            detail = er.get("error", "")
            logger.info("  %s: %s %s", iid, status, detail)
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_iid = {}
            for item in eval_items:
                fut = executor.submit(_eval_worker, item)
                future_to_iid[fut] = item[0]

            for i, fut in enumerate(as_completed(future_to_iid)):
                iid = future_to_iid[fut]
                er = fut.result()
                eval_results.append(er)
                status = "RESOLVED" if er.get("resolved") else "FAILED"
                logger.info("[%d/%d] %s: %s", i + 1, len(eval_items), iid, status)

    # Summary
    total = len(eval_results)
    resolved = sum(1 for r in eval_results if r.get("resolved"))
    errors = sum(1 for r in eval_results if r.get("error"))
    patch_failed = sum(1 for r in eval_results if "Patch" in str(r.get("error", "")))

    print(f"\n{'=' * 60}")
    print(f"Total evaluated: {total}")
    print(f"Resolved (FAIL_TO_PASS tests pass): {resolved} ({100 * resolved / max(total, 1):.1f}%)")
    print(f"Patch apply failures: {patch_failed}")
    print(f"Other errors: {errors - patch_failed}")
    print(f"{'=' * 60}")

    # Per-repo breakdown
    repo_stats = {}
    for er in eval_results:
        repo = er["instance_id"].split("__")[0].replace("__", "/")
        if repo not in repo_stats:
            repo_stats[repo] = {"total": 0, "resolved": 0}
        repo_stats[repo]["total"] += 1
        if er.get("resolved"):
            repo_stats[repo]["resolved"] += 1

    print("\nPer-repo breakdown:")
    for repo, stats in sorted(repo_stats.items()):
        pct = 100 * stats["resolved"] / max(stats["total"], 1)
        print(f"  {repo}: {stats['resolved']}/{stats['total']} ({pct:.1f}%)")

    # Save
    output_path = str(Path(args.results).parent / "eval_local.json")
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
