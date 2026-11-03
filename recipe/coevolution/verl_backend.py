"""Production stages: owned SGLang server, existing MetaAgent, pinned veRL/FSDP."""

from __future__ import annotations

import asyncio
import json
import hashlib
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from harnessx.core.model_config import ModelConfig
from recipe.coevolution.run import model_digest, write_json
from recipe.coevolution.buffer import append_records, load_records

ROOT = Path(__file__).resolve().parents[2]
ROLLOUTS_PER_TASK_HARNESS = 1


def offline_training_command(settings, model, buffer_paths, output):
    """Build a model-update command that consumes saved trajectories only."""
    environment = dict(os.environ)
    environment.update(
        HX_ROOT=str(ROOT),
        PYTHONPATH=f"{ROOT}:{ROOT / '.deps/verl'}"
        + (f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""),
    )
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={settings.gpus}",
        "-m",
        "recipe.coevolution.offline_train",
        "--model",
        str(model),
        "--output",
        str(output),
        "--steps",
        str(settings.steps),
        "--batch-size",
        str(settings.train_batch_size),
        "--buffer-capacity",
        str(settings.buffer_capacity),
        "--learning-rate",
        str(settings.learning_rate),
    ]
    for buffer_path in buffer_paths:
        argv.extend(("--buffer", str(buffer_path)))
    return argv, environment


class VerlBackend:
    def __init__(self, settings):
        self.settings = settings
        from recipe.verl.launch.environment import apply_defaults
        from recipe.verl.research.config import cfg

        apply_defaults(cfg.to_env_dict())
        self.url = f"http://127.0.0.1:{settings.port}"
        self.tokenizer = None
        self.tasks = []
        for variable in ("NO_PROXY", "no_proxy"):
            existing = os.environ.get(variable, "")
            os.environ[variable] = ",".join(filter(None, [existing, "127.0.0.1", "localhost"]))

    async def initialize(self, root):
        self.preflight()
        return Path(self.settings.initial_model).resolve()

    def preflight(self):
        from recipe.verl.bootstrap import check_installation
        from recipe.verl.launch.preflight import check_versions

        check_installation()
        check_versions()
        import verl

        if not Path(verl.__file__).resolve().is_relative_to(ROOT / ".deps/verl"):
            raise RuntimeError("PYTHONPATH must select the generated pinned .deps/verl checkout")
        if not self.tasks:
            tasks = [
                json.loads(line) for line in Path(self.settings.search_tasks).read_text().splitlines() if line.strip()
            ]
            if not tasks or any(
                not isinstance(task, dict)
                or not isinstance(task.get("question"), str)
                or not task["question"].strip()
                or not isinstance(task.get("answer"), str)
                or not task["answer"].strip()
                for task in tasks
            ):
                raise ValueError("search-tasks must be nonempty JSONL with string question and answer fields")
            self.tasks = tasks

    def _training_rows(self):
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(Path(self.settings.train_file).resolve()), split="train")
        rows = []
        for row in dataset:
            prompt = row.get("prompt")
            reward_model = row.get("reward_model") or {}
            label = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            if prompt is None or not isinstance(label, str) or not label:
                raise ValueError("Training Parquet rows require prompt and reward_model.ground_truth")
            extra = row.get("extra_info") or {}
            task_uid = extra.get("id") if isinstance(extra, dict) else None
            if not isinstance(task_uid, str) or not task_uid:
                canonical = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
                task_uid = hashlib.sha256(canonical.encode()).hexdigest()
            rows.append((task_uid, prompt, label, extra if isinstance(extra, dict) else {}))
        if not rows:
            raise ValueError("Training Parquet contains no records")
        return rows

    async def collect(self, model, pool, work, round_index):
        """Collect one trajectory per task and Harness before offline training."""
        from recipe.verl.harness_rollout import VerlSample, generate, reward_func

        buffer_path = work / "buffer" / "trajectories.jsonl"
        source_model = model_digest(model)
        rows = self._training_rows()
        async with self.search(model, work / "collection_service"):
            records = []
            exploration_dir = work / "exploration"
            exploration_dir.mkdir(parents=True, exist_ok=True)
            for task_uid, prompt, label, extra in rows:
                for harness_path in pool:
                    for _ in range(ROLLOUTS_PER_TASK_HARNESS):
                        sample = VerlSample(
                            prompt=prompt,
                            label=label,
                            metadata={**extra, "id": task_uid, "task_type": extra.get("task_type", "research")},
                        )
                        episode = await generate(
                            sample,
                            {"temperature": 1.0, "max_new_tokens": 2048},
                            self.url + "/generate",
                            self.tokenizer,
                            harness_config=harness_path,
                        )
                        if not episode or episode.get("status") == "aborted" or not episode.get("eval_result"):
                            continue
                        response_length = episode["response_length"]
                        tokens = episode["tokens"]
                        response_ids = tokens[-response_length:]
                        prompt_ids = tokens[:-response_length]
                        response_mask = episode["loss_mask"]
                        old_logprobs = episode["rollout_log_probs"]
                        if (
                            not prompt_ids
                            or not response_ids
                            or len(response_ids) != len(response_mask)
                            or len(response_ids) != len(old_logprobs)
                            or not any(response_mask)
                        ):
                            continue
                        reward = await reward_func(sample, episode)
                        harness_id = hashlib.sha256(Path(harness_path).read_bytes()).hexdigest()
                        trajectory_path = (
                            exploration_dir / f"{hashlib.sha256(task_uid.encode()).hexdigest()}_{harness_id}.md"
                        )
                        trajectory_path.write_text(episode["trajectory"].to_markdown(level="full"), encoding="utf-8")
                        records.append(
                            {
                                "task_uid": task_uid,
                                "harness_id": harness_id,
                                "harness": str(harness_path),
                                "round": round_index,
                                "source_model": source_model,
                                "prompt_ids": prompt_ids,
                                "response_ids": response_ids,
                                "response_mask": response_mask,
                                "old_logprobs": old_logprobs,
                                "reward": float(reward["score"]),
                            }
                        )
            append_records(buffer_path, records)
        if not records:
            raise ValueError("No valid annotated trajectories were collected for the offline buffer")
        return buffer_path

    @asynccontextmanager
    async def search(self, model, work):
        from transformers import AutoTokenizer

        self.preflight()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model))
        reserved = {"--model-path", "--model", "--host", "--port", "--api-key"}
        if any(argument.split("=", 1)[0] in reserved for argument in self.settings.server_args):
            raise ValueError("Do not override owned SGLang model/host/port/auth arguments")
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.settings.port),
            *self.settings.server_args,
        ]
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            try:
                await client.get(self.url + "/health")
            except httpx.RequestError:
                pass
            else:
                raise RuntimeError("SGLang port is already occupied; refusing to use an unrelated model server")
        write_json(work / "loaded_model.json", {"path": str(model), "weights": model_digest(model), "argv": command})
        with (work / "sglang.log").open("w") as log:
            process = await asyncio.create_subprocess_exec(
                *command, cwd=ROOT, stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True
            )
            try:
                deadline = time.monotonic() + 600
                async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                    while True:
                        if process.returncode is not None:
                            raise RuntimeError(f"SGLang exited ({process.returncode}); inspect {work / 'sglang.log'}")
                        try:
                            response = await client.get(self.url + "/health")
                            if response.status_code == 200:
                                break
                        except httpx.RequestError:
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Owned SGLang server did not become healthy within 600 seconds")
                        await asyncio.sleep(1)
                yield
            finally:
                if process.returncode is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=30)
                    except asyncio.TimeoutError:
                        os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()

    async def evaluate(self, model, harness, output):
        from recipe.verl.harness_rollout import VerlSample, generate

        output.mkdir(parents=True)
        scores = []
        for index, task in enumerate(self.tasks):
            episode = await generate(
                VerlSample(prompt=task["question"], label=task["answer"]),
                {"temperature": 0, "max_new_tokens": 2048, "seed": self.settings.seed},
                self.url + "/generate",
                self.tokenizer,
                harness_config=harness,
            )
            if episode is None or episode["status"] == "aborted" or episode["eval_result"] is None:
                raise RuntimeError("Search rollout failed; refusing to score a failure as a successful candidate")
            scores.append(float(episode["eval_result"].score))
            (output / f"task_{index:05d}.md").write_text(episode["trajectory"].to_markdown(level="full"))
        score = sum(scores) / len(scores)
        write_json(output / "metrics.json", {"accuracy": score, "task_scores": scores, "model": str(model)})
        return score

    async def propose(self, model, incumbent, trajectories, output, index):
        from harnessx.meta_harness import MetaAgent
        from harnessx.providers.sglang import SGLangProvider
        from recipe.verl.research.config import cfg

        contract = (
            "This is a behavior-only alternating research recipe. Preserve all top-level config slots. "
            "Keep exactly one SystemPromptProcessor with _FixedSystemPromptBuilder(text=...), and one "
            "recipe.verl.processor.early_commit.EarlyCommitProcessor. "
            "Only evolve the prompt text and early-commit arguments; do not add Python modules, tools, "
            "evaluators, plugins, reward code, workspace or sandbox configuration. "
            "The task evaluator, tool runtime, labels, model and training algorithm are immutable. "
            "Use only training/search-dev trajectories; these are not final held-out test results."
        )
        (trajectories / "search_contract.md").write_text(contract)
        replay_provider = SGLangProvider(
            url=self.url + "/generate",
            tokenizer=self.tokenizer,
            rollout_max_context_len=32768,
            sampling_params={"temperature": 0, "max_new_tokens": 2048},
            tool_call_format=cfg.tool_call_format,
            tool_result_role=cfg.tool_result_role,
            strip_think_prefix=True,
            stop_strings=["</tool_call>"],
        )
        agent = MetaAgent(
            inner_model=ModelConfig.from_yaml_file(Path(self.settings.meta_model)),
            memo_path=Path(self.settings.output).resolve() / "meta_memo.md",
            allowed_write_roots=(output.resolve(),),
            wall_clock_s=600,
            max_steps=80,
        )
        try:
            return await agent.evolve(
                current_config=incumbent,
                trajectories_dir=trajectories,
                output_dir=output,
                replay_model=ModelConfig(main=replay_provider),
            )
        finally:
            await replay_provider.aclose()

    async def train(self, model, harness, work):
        round_index = int(work.name.split("_")[1])
        if round_index <= self.settings.model_warmup_rounds:
            write_json(
                work / "training.json",
                {
                    "mode": "offline_buffer_grpo",
                    "model_updated": False,
                    "reason": "warmup",
                    "warmup_rounds": self.settings.model_warmup_rounds,
                },
            )
            return None
        buffer_paths = []
        state_path = Path(self.settings.output) / "state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            buffer_paths.extend(
                str(record["buffer"])
                for record in state.get("rounds", [])
                if record.get("buffer") and Path(record["buffer"]).is_file()
            )
        current_buffer = work / "buffer" / "trajectories.jsonl"
        buffer_paths.append(str(current_buffer))
        capacity = self.settings.buffer_capacity
        train_records = load_records(buffer_paths, capacity=capacity, allow_empty=True)
        if not train_records:
            write_json(
                work / "training.json",
                {"mode": "offline_buffer_grpo", "model_updated": False, "reason": "no_cross_harness_groups"},
            )
            return None
        write_json(
            work / "buffer" / "manifest.json",
            {
                "capacity": capacity,
                "rollouts_per_task_harness": ROLLOUTS_PER_TASK_HARNESS,
                "trajectory_count": len(train_records),
                "task_groups": len({row["task_uid"] for row in train_records}),
                "exploration_harnesses": sorted({row["harness"] for row in train_records if row.get("harness")}),
                "sources": buffer_paths,
            },
        )
        output = work / "model"
        command, environment = offline_training_command(self.settings, model, buffer_paths, output)
        write_json(work / "training_command.json", {"argv": command, "mode": "offline_buffer_grpo"})
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        if not (output / "config.json").is_file():
            raise FileNotFoundError("Offline trainer did not export a Hugging Face model config.json")
        model_digest(output)
        return output
