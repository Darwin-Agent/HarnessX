"""One-shot: dump a pristine GAIA base config to YAML.

Writes the *current* GAIA harness as the evolver's starting point —
using ``PlainMarkdownSystemPromptBuilder(gaia_agent.md)`` (the live
`.md` prompt, not some v5-vintage `.j2` artefact), ``build_gaia_tools()``
(the standard tool set), and the standard processor stack. Deliberately
excludes any shipped artefacts from prior runs so an evolver started
from this file has no hidden inheritance.

Run::

    python3 recipe/gaia_evolver/tools/dump_pristine_base.py \
        --out recipe/gaia_evolver/data/gaia_base_pristine.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_pristine_config():
    from harnessx.core.builder import HarnessBuilder
    from harnessx.processors.context.system_prompt import SystemPromptProcessor
    from harnessx.processors.context.user_wrapper import UserWrapperProcessor
    from harnessx.processors.control.cost_guard import CostGuardProcessor
    from harnessx.processors.control.loop_detection import LoopDetectionProcessor
    from harnessx.processors.control.token_budget import TokenBudgetProcessor
    from harnessx.processors.observability.checkpoint import CheckpointProcessor
    from harnessx.processors.observability.otel_proc import OTelProcessor
    from harnessx.processors.context.strategies.system_prompt.plain_markdown import (
        PlainMarkdownSystemPromptBuilder,
    )
    from harnessx.tools.builtin import build_gaia_tools
    from benchmarks.gaia.defaults import (
        CHECKPOINT_EVERY_N,
        COST_GUARD_MAX_USD,
        LOOP_THRESHOLD,
        LOOP_WINDOW_SIZE,
        TOKEN_BUDGET_RATIO,
    )

    gaia_prompt_path = _REPO_ROOT / "benchmarks" / "gaia" / "prompts" / "gaia_agent.md"
    assert gaia_prompt_path.is_file(), f"missing GAIA prompt: {gaia_prompt_path}"

    return (
        HarnessBuilder()
        .slot(tool_registry=build_gaia_tools())
        .add(SystemPromptProcessor(
            PlainMarkdownSystemPromptBuilder(str(gaia_prompt_path)),
        ))
        .add(UserWrapperProcessor())
        .add(TokenBudgetProcessor(ratio=TOKEN_BUDGET_RATIO))
        .add(CostGuardProcessor(max_usd=COST_GUARD_MAX_USD))
        .add(LoopDetectionProcessor(
            window_size=LOOP_WINDOW_SIZE, threshold=LOOP_THRESHOLD,
        ))
        .add(CheckpointProcessor(every_n=CHECKPOINT_EVERY_N))
        .add(OTelProcessor())
        .build()
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hc = build_pristine_config()
    out.write_text(hc.to_yaml(), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
