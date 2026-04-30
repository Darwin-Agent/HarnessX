#!/usr/bin/env bash
# Must run from HarnessX repo root so `python -m recipe.tau2_evolver.run`
# can find the `recipe` package. `uv run --project` only changes the venv,
# not cwd. Meta model left at defaults.py (anthropic/.../claude-opus-4-6
# with extended thinking — see _make_provider in run.py).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

# Internal .srv endpoints must NOT go through the WSL2→Windows proxy;
# otherwise OpenAI/LiteLLM SDKs throw sporadic "Connection error." under load.
# Unset is more reliable than NO_PROXY across uv-run / httpx / litellm edges.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$HOME/tau2-bench/data}"

# Uncomment to bootstrap from an existing report and skip R0 simulation:
# FROM_REPORT=(--from-report recipe/tau2_evolver/runs/retail_evolve_subset_0428/R4/report.json)
FROM_REPORT=()

uv run --project ~/tau2-bench \
    python -m recipe.tau2_evolver.run \
    --domain retail \
    --base-config recipe/tau2_evolver/runs/retail_evolve_subset_0428/R4/config.yaml \
    "${FROM_REPORT[@]}" \
    --num-rounds 5 \
    --run-tag retail_evolve_subset \
    --clean \
    --evolve-cost 200 \
    --max-concurrency 30 \
    --regression-tolerance 0.05 \
    --agent-temperature 0.3 \
    --user-temperature 0 \
    --task-ids 30,32,37,41,49,60,79,100,102,109,4,11,12,31,38,59,64,66,71,105,112
