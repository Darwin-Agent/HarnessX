"""Verify committed co-evolution checkpoints and both directions of artifact handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recipe.coevolution.behavior import resolve_behavior
from recipe.coevolution.run import digest, model_digest


def verify(output: Path) -> dict:
    state = json.loads((Path(output) / "state.json").read_text())
    previous = state["initial"]
    for index, record in enumerate([previous, *state["rounds"]]):
        resolve_behavior(explicit=record["harness"])
        if digest(Path(record["harness"])) != record["harness_hash"]:
            raise ValueError("Harness artifact checksum mismatch")
        if model_digest(Path(record["model"])) != record["model_hash"]:
            raise ValueError("Model weight checksum mismatch")
        if index:
            buffer_path = Path(record["buffer"])
            proposal_path = Path(record["harness_proposal"])
            model_updated = record.get("model_updated")
            if (
                record["round"] != index
                or record["input_model"] != previous["model"]
                or record["input_model_hash"] != previous["model_hash"]
                or record["input_harness"] != previous["harness"]
                or not isinstance(model_updated, bool)
                or (model_updated and record["model_hash"] == previous["model_hash"])
                or (not model_updated and record["model_hash"] != previous["model_hash"])
                or record.get("harness_selection") != "latest_proposal"
                or not proposal_path.is_file()
                or digest(proposal_path) != record.get("harness_proposal_hash")
                or not buffer_path.is_file()
                or digest(buffer_path) != record.get("buffer_hash")
            ):
                raise ValueError("Broken Harness/model/buffer alternating handoff")
        previous = record
    return {
        "backend": state["settings"]["backend"],
        "completed_rounds": len(state["rounds"]),
        "final_model": previous["model"],
        "final_harness": previous["harness"],
        "model_update_rounds": sum(bool(record.get("model_updated")) for record in state["rounds"]),
        "handoff_verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.output), indent=2))


if __name__ == "__main__":
    main()
