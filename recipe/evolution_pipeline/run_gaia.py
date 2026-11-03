from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

for module_name in tuple(sys.modules):
    if module_name == "recipe" or module_name.startswith("recipe."):
        del sys.modules[module_name]

runpy.run_module("recipe.gaia_evolver.run", run_name="__main__")
