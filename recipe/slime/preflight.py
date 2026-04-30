# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def check_slime() -> bool:
    try:
        import slime  # noqa: F401

        path = slime.__file__
        logger.info("  slime OK — %s", path)
        return True
    except ImportError as exc:
        logger.error("  slime NOT found: %s", exc)
        return False


def check_harnessx() -> bool:
    try:
        import harnessx  # noqa: F401

        logger.info("  harnessx OK")
        return True
    except ImportError as exc:
        logger.error("  harnessx NOT found: %s", exc)
        return False


def check_tool_registry() -> bool:
    try:
        from recipe.slime.math.tools import get_registry

        registry = get_registry()
        names = registry.list_names()
        assert "code_interpreter" in names, f"code_interpreter not in {names}"
        logger.info("  tool_registry OK — tools: %s", names)
        return True
    except Exception as exc:
        logger.error("  tool_registry FAILED: %s", exc)
        return False


def check_rollout_imports() -> bool:
    try:
        from recipe.slime.harness_rollout import generate, reward_func  # noqa: F401

        logger.info("  harness_rollout imports OK")
        return True
    except Exception as exc:
        logger.error("  harness_rollout import FAILED: %s", exc)
        return False


def check_math_dapo() -> bool:
    try:
        from slime.rollout.rm_hub.math_dapo_utils import compute_score

        result = compute_score("Answer: \\boxed{42}", "42", strict_box_verify=True)
        assert result["score"] > 0, f"Expected positive score, got {result}"
        logger.info("  math_dapo_utils OK — smoke test passed")
        return True
    except Exception as exc:
        logger.error("  math_dapo_utils FAILED: %s", exc)
        return False


def check_data_files(data_dir: str | None = None) -> bool:
    base = data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data", "slime", "retool")
    files = {
        "dapo-math-17k.jsonl": "RL training data",
        "ReTool-SFT.parquet": "SFT training data",
        "aime-2024.jsonl": "RL eval data",
    }
    ok = True
    for fname, desc in files.items():
        path = os.path.join(base, fname)
        if os.path.exists(path):
            size = os.path.getsize(path)
            logger.info("  %s OK — %s (%.1f MB)", desc, fname, size / 1e6)
        else:
            logger.warning("  %s MISSING — %s (run data_prep.sh)", desc, path)
            ok = False
    return ok


def check_sglang(url: str | None = None) -> bool:
    if url is None:
        logger.info("  SGLang check skipped (no --sglang-url provided)")
        return True
    import urllib.request

    health_url = url.rstrip("/").replace("/generate", "") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            logger.info("  SGLang OK — %s → HTTP %d", health_url, resp.status)
        return True
    except Exception as exc:
        logger.error("  SGLang NOT reachable at %s: %s", health_url, exc)
        return False


def check_megatron() -> bool:
    megatron = os.environ.get("MEGATRON_ROOT", "/root/Megatron-LM")
    if os.path.isdir(megatron):
        logger.info("  Megatron-LM OK — %s", megatron)
        return True
    logger.error("  Megatron-LM NOT found at %s", megatron)
    return False


def run_all(sglang_url: str | None = None) -> bool:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("=== HarnessX + Slime preflight checks ===")

    checks = [
        ("slime", check_slime),
        ("harnessx", check_harnessx),
        ("tool_registry", check_tool_registry),
        ("rollout imports", check_rollout_imports),
        ("math_dapo_utils", check_math_dapo),
        ("data files", check_data_files),
        ("Megatron-LM", check_megatron),
        ("SGLang", lambda: check_sglang(sglang_url)),
    ]

    results = {}
    for name, fn in checks:
        logger.info("[%s]", name)
        try:
            results[name] = fn()
        except Exception as exc:
            logger.error("  Unexpected error: %s", exc)
            results[name] = False

    logger.info("")
    logger.info("=== Summary ===")
    all_ok = True
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        logger.info("  %-25s %s", name, status)
        if not ok:
            all_ok = False

    if all_ok:
        logger.info("\nAll checks passed. Ready to train.")
    else:
        logger.warning("\nSome checks failed. Fix the issues above before training.")

    return all_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preflight checks for HarnessX+Slime")
    parser.add_argument("--sglang-url", default=None, help="SGLang server URL to ping")
    args = parser.parse_args()

    ok = run_all(sglang_url=args.sglang_url)
    sys.exit(0 if ok else 1)
