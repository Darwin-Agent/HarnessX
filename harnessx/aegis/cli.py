# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""AEGIS pilot CLI.

Usage:
    python -m harnessx.aegis.cli pilot \\
        --tag aegis_pilot_v1 \\
        --rounds 3 \\
        --num_evolvers 4 \\
        --benchmark gaia_small \\
        --budget_per_round_usd 20
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pilot", help="Run a pilot experiment")
    p.add_argument("--tag", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--num_evolvers", type=int, default=4)
    p.add_argument("--k_rollouts", type=int, default=2)
    p.add_argument("--budget_per_round_usd", type=float, default=20.0)
    p.add_argument("--run_dir", type=Path, default=Path("runs"))

    args = parser.parse_args()
    if args.command == "pilot":
        asyncio.run(_run_pilot(args))


async def _run_pilot(args):
    from harnessx.aegis import AegisAgent  # noqa: F401 — reserved for integration layer
    print(f"[aegis] pilot tag={args.tag} rounds={args.rounds} evolvers={args.num_evolvers}")
    print("See recipe/gaia_evolver/run.py for integration pattern.")


if __name__ == "__main__":
    main()
