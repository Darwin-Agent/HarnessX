"""Run agent exploration, AEGIS Harness evolution, then offline model evolution."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from harnessx.core.harness import HarnessConfig
from recipe.coevolution.behavior import baseline_behavior, validate_behavior


@dataclass(frozen=True)
class Settings:
    backend: str
    output: str
    rounds: int = 2
    initial_model: str | None = None
    initial_harness: str | None = None
    search_tasks: str | None = None
    meta_model: str | None = None
    train_file: str | None = None
    steps: int = 10
    train_batch_size: int = 1
    buffer_capacity: int = 2048
    learning_rate: float = 1e-6
    gpus: int = 8
    port: int = 31000
    server_args: tuple[str, ...] = ()
    seed: int = 42

    def validate(self):
        if self.backend not in {"cpu", "verl"}:
            raise ValueError("backend must be cpu or verl")
        if self.rounds < 1 or self.steps < 1 or self.gpus < 1 or self.train_batch_size < 1:
            raise ValueError("Positive rounds, training steps, batch size, and GPU count are required")
        if self.buffer_capacity < 2:
            raise ValueError("buffer_capacity must be at least 2")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 1 <= self.port <= 65535:
            raise ValueError("Invalid server port")
        if self.backend == "verl":
            for name in ("initial_model", "search_tasks", "meta_model", "train_file"):
                value = getattr(self, name)
                if not value or not Path(value).exists():
                    raise ValueError(f"{name} must name an existing local artifact")
            if not Path(self.initial_model).is_dir():
                raise ValueError("initial_model must be a local Hugging Face model directory")
        if self.initial_harness and not Path(self.initial_harness).is_file():
            raise ValueError(f"Harness configuration does not exist: {self.initial_harness}")


def digest(path: Path) -> str:
    """Hash file contents, including actual model weights rather than just a path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    hasher = hashlib.sha256()
    count = 0
    for artifact in files:
        if not artifact.is_file():
            continue
        count += 1
        if path.is_dir():
            hasher.update(str(artifact.relative_to(path)).encode())
            hasher.update(b"\0")
        with artifact.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                hasher.update(chunk)
    if not count:
        raise ValueError(f"Empty artifact: {path}")
    return hasher.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def model_digest(path: Path) -> str:
    weights = sorted(
        artifact
        for artifact in Path(path).rglob("*")
        if artifact.is_file()
        and (
            artifact.suffix == ".safetensors"
            or artifact.name == "policy.pt"
            or artifact.name.startswith("pytorch_model")
        )
        and artifact.suffix != ".json"
    )
    if not weights:
        raise ValueError(f"No exported model weights found in {path}")
    return hashlib.sha256("".join(digest(artifact) for artifact in weights).encode()).hexdigest()


async def alternate(settings: Settings, backend, *, resume=False) -> dict:
    """Hold an exclusive run lock across exploration and both evolution stages."""
    root = Path(settings.output).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.coevolution.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is already using this co-evolution output") from error
        return await _alternate(settings, backend, resume=resume)


async def _alternate(settings: Settings, backend, *, resume=False) -> dict:
    """Commit a round only after exploration, Harness evolution, and model update succeed."""
    settings.validate()
    root = Path(settings.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    identity = asdict(settings)
    identity.pop("rounds")
    identity["output"] = str(root)
    identity = json.loads(json.dumps(identity))
    inputs = {}
    for name in ("initial_model", "initial_harness", "search_tasks", "meta_model", "train_file"):
        value = getattr(settings, name)
        if value:
            inputs[name] = digest(Path(value))
    if state_path.exists():
        if not resume:
            raise FileExistsError("Output already initialized; use --resume or a new output directory")
        state = json.loads(state_path.read_text())
        if state.get("schema") != 3:
            raise ValueError("This run uses an incompatible state schema; start a new output directory")
        if state["settings"] != identity or state["inputs"] != inputs:
            raise ValueError("Resume settings or input contents differ from the original run")
    else:
        if resume:
            raise FileNotFoundError("Cannot resume without state.json")
        if any(root.iterdir()):
            raise FileExistsError("New run requires an empty output directory")
        initial_model = await backend.initialize(root)
        initial_harness = root / "initial_harness.yaml"
        if settings.initial_harness:
            config = HarnessConfig.from_yaml_file(Path(settings.initial_harness))
        else:
            config = baseline_behavior()
        validate_behavior(config)
        config.to_yaml_file(initial_harness)
        state = {
            "schema": 3,
            "settings": identity,
            "inputs": inputs,
            "initial": {
                "model": str(initial_model),
                "model_hash": model_digest(initial_model),
                "harness": str(initial_harness),
                "harness_hash": digest(initial_harness),
            },
            "rounds": [],
        }
        write_json(state_path, state)
    current = state["initial"]
    for completed in [state["initial"], *state["rounds"]]:
        for kind in ("model", "harness"):
            checksum = model_digest if kind == "model" else digest
            if checksum(Path(completed[kind])) != completed[f"{kind}_hash"]:
                raise ValueError(f"Completed {kind} artifact changed; refusing unsafe resume")
        if completed is not state["initial"]:
            proposal_path = Path(completed["harness_proposal"])
            buffer_path = Path(completed["buffer"])
            if (
                not proposal_path.is_file()
                or digest(proposal_path) != completed.get("harness_proposal_hash")
                or completed.get("harness_selection") != "latest_proposal"
            ):
                raise ValueError("A committed latest Harness proposal changed; refusing unsafe resume")
            if not buffer_path.is_file() or digest(buffer_path) != completed.get("buffer_hash"):
                raise ValueError("A committed trajectory buffer changed or is missing; refusing unsafe resume")
        current = completed
    if settings.rounds < len(state["rounds"]):
        raise ValueError("Requested fewer rounds than already committed")
    while len(state["rounds"]) < settings.rounds:
        index = len(state["rounds"]) + 1
        attempt = 1
        while (root / f"round_{index:03d}_attempt_{attempt:02d}").exists():
            attempt += 1
        work = root / f"round_{index:03d}_attempt_{attempt:02d}"
        work.mkdir()
        model = Path(current["model"])
        before_hash = model_digest(model)
        if before_hash != current["model_hash"]:
            raise ValueError("Input checkpoint changed between rounds")
        exploration_harness = Path(current["harness"])
        buffer_path = Path(await backend.collect(model, [exploration_harness], work, index)).resolve()
        if not buffer_path.is_file():
            raise FileNotFoundError(f"Trajectory collection did not create its buffer: {buffer_path}")
        exploration_dir = work / "exploration"
        async with backend.search(model, work / "aegis"):
            proposal = await backend.propose(model, exploration_harness, exploration_dir, work / "proposal", index)
            try:
                candidate = HarnessConfig.from_yaml_file(proposal)
                validate_behavior(candidate)
            except (ValueError, TypeError, FileNotFoundError) as error:
                raise ValueError(f"AEGIS produced an invalid Harness: {error}") from error
        if model_digest(model) != before_hash:
            raise ValueError("Harness search must not mutate the frozen model")
        harness = work / "latest_harness.yaml"
        candidate.to_yaml_file(harness)
        selected_hash = digest(harness)
        write_json(
            work / "search.json",
            {
                "model": str(model),
                "model_hash": before_hash,
                "input_harness": str(exploration_harness),
                "proposal": str(proposal),
                "selected_harness": str(harness),
                "selection": "latest_proposal",
            },
        )
        trained = await backend.train(model, harness, work)
        if trained is None:
            trained = model
            model_updated = False
        else:
            trained = Path(trained).resolve()
            if not trained.is_relative_to(work.resolve()) or not trained.is_dir():
                raise ValueError("Trainer must export a new model directory inside this round")
            if model_digest(trained) == before_hash:
                raise ValueError("Model weights did not change")
            model_updated = True
        if digest(harness) != selected_hash:
            raise ValueError("Training mutated the latest Harness")
        after_hash = model_digest(trained)
        if model_digest(model) != before_hash:
            raise ValueError("Training mutated its input checkpoint")
        record = {
            "round": index,
            "input_model": str(model),
            "input_model_hash": before_hash,
            "input_harness": str(exploration_harness),
            "model": str(trained),
            "model_hash": after_hash,
            "harness": str(harness),
            "harness_hash": digest(harness),
            "harness_proposal": str(proposal),
            "harness_proposal_hash": digest(Path(proposal)),
            "harness_selection": "latest_proposal",
            "model_updated": model_updated,
            "buffer": str(buffer_path),
            "buffer_hash": digest(buffer_path),
        }
        write_json(work / "completed.json", record)
        state["rounds"].append(record)
        write_json(state_path, state)
        current = record
        print(f"round={index} latest_harness={harness} model_updated={model_updated}", flush=True)
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "verl"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    for name in ("initial-model", "initial-harness", "search-tasks", "meta-model", "train-file"):
        parser.add_argument(f"--{name}")
    for name, default in (
        ("steps", 10),
        ("train-batch-size", 1),
        ("buffer-capacity", 2048),
        ("gpus", 8),
        ("port", 31000),
        ("seed", 42),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--server-args", type=json.loads, default=[])
    parser.add_argument("--resume", action="store_true")
    values = vars(parser.parse_args())
    resume = values.pop("resume")
    if not isinstance(values["server_args"], list) or any(not isinstance(item, str) for item in values["server_args"]):
        parser.error("server_args must be a JSON array of strings")
    values["server_args"] = tuple(values["server_args"])
    for field in ("initial_model", "initial_harness", "search_tasks", "meta_model", "train_file"):
        if values[field]:
            values[field] = str(Path(values[field]).resolve())
    values["output"] = str(Path(values["output"]).resolve())
    return Settings(**values), resume


def main():
    settings, resume = parse_args()
    settings.validate()
    if settings.backend == "cpu":
        from recipe.coevolution.smoke import CPUBackend

        backend = CPUBackend(settings)
    else:
        from recipe.coevolution.verl_backend import VerlBackend

        backend = VerlBackend(settings)
    asyncio.run(alternate(settings, backend, resume=resume))


if __name__ == "__main__":
    main()
