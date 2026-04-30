# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# Default output directory — data files live in data/slime/retool/
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "slime", "retool")


def prepare_rl_data(
    output_path: str | None = None,
) -> str:
    """Download zhuzilin/dapo-math-17k and preprocess for RL training.

    Mirrors retool rl_data_preprocess.py exactly:
        prompt = example["prompt"][0]["content"]
        label  = example["reward_model"]["ground_truth"]

    Args:
        output_path: Where to write the JSONL file.
                     Defaults to ./data/retool/dapo-math-17k.jsonl

    Returns:
        Absolute path to the written file.
    """
    from datasets import load_dataset

    if output_path is None:
        output_path = os.path.join(_DEFAULT_DATA_DIR, "dapo-math-17k.jsonl")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    logger.info("Downloading zhuzilin/dapo-math-17k …")
    ds = load_dataset("zhuzilin/dapo-math-17k", split="train")

    def transform(example: dict) -> dict:
        prompt = example.get("prompt", "")
        if isinstance(prompt, list) and prompt:
            content = prompt[0].get("content", "")
        elif isinstance(prompt, str):
            content = prompt
        else:
            content = ""

        # Support both field names seen across dataset versions
        rm = example.get("reward_model", {})
        label = example.get("label") or rm.get("ground_truth") or rm.get("answer") or ""
        return {"prompt": content, "label": label}

    ds2 = ds.map(transform, remove_columns=ds.column_names)
    ds2.to_json(output_path, orient="records", lines=True)
    logger.info("RL data: %d samples → %s", len(ds2), output_path)
    return os.path.abspath(output_path)


def prepare_sft_data(
    output_path: str | None = None,
) -> str:
    """Download JoeYing/ReTool-SFT and preprocess for SFT training.

    Mirrors retool sft_data_processing.py exactly:
        messages = [{role, content}, ...]

    Args:
        output_path: Where to write the parquet file.
                     Defaults to ./data/retool/ReTool-SFT.parquet

    Returns:
        Absolute path to the written file.
    """
    from datasets import load_dataset

    if output_path is None:
        output_path = os.path.join(_DEFAULT_DATA_DIR, "ReTool-SFT.parquet")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    logger.info("Downloading JoeYing/ReTool-SFT …")
    ds = load_dataset("JoeYing/ReTool-SFT")["train"]

    def convert(sample: dict) -> dict:
        return {"messages": [{"role": t["role"], "content": t["content"]} for t in sample["messages"]]}

    ds = ds.map(convert)
    ds.to_parquet(output_path)
    logger.info("SFT data: %d samples → %s", len(ds), output_path)
    return os.path.abspath(output_path)


def prepare_eval_data(
    output_path: str | None = None,
) -> str:
    """Download zhuzilin/aime-2024 for RL evaluation.

    Saves as JSONL with each line: {"prompt": [...], "label": "..."}.
    The prompt field preserves the original messages list format so that
    the Slime eval pipeline's --apply-chat-template works correctly.

    Args:
        output_path: Where to write the JSONL file.
                     Defaults to ./data/retool/aime-2024.jsonl

    Returns:
        Absolute path to the written file.
    """
    from datasets import load_dataset

    if output_path is None:
        output_path = os.path.join(_DEFAULT_DATA_DIR, "aime-2024.jsonl")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    logger.info("Downloading zhuzilin/aime-2024 …")
    ds = load_dataset("zhuzilin/aime-2024", split="train")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Eval data: %d samples → %s", len(ds), output_path)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Prepare Slime math datasets")
    parser.add_argument("--all", action="store_true", help="Prepare all datasets")
    parser.add_argument("--rl", action="store_true", help="Prepare RL training data (dapo-math-17k)")
    parser.add_argument("--sft", action="store_true", help="Prepare SFT training data (ReTool-SFT)")
    parser.add_argument("--eval", action="store_true", help="Prepare eval data (aime-2024)")
    parser.add_argument("--output-dir", default=None, help="Output directory override")
    args = parser.parse_args()

    if not any([args.all, args.rl, args.sft, args.eval]):
        parser.print_help()
        raise SystemExit(1)

    def _path(name: str) -> str | None:
        if args.output_dir:
            return os.path.join(args.output_dir, name)
        return None

    if args.all or args.sft:
        prepare_sft_data(_path("ReTool-SFT.parquet"))
    if args.all or args.rl:
        prepare_rl_data(_path("dapo-math-17k.jsonl"))
    if args.all or args.eval:
        prepare_eval_data(_path("aime-2024.jsonl"))
