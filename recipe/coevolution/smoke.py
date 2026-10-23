"""CPU closed-loop verification: a tiny stochastic policy, real Harness and GRPO updates.

This backend tests handoff and optimization without claiming SGLang/CUDA or live
MetaAgent coverage. The proposal is deterministic; model updates are not mocked.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import torch

from harnessx.core.harness import HarnessConfig
from recipe.coevolution.behavior import SYSTEM
from recipe.coevolution.run import model_digest, write_json
from recipe.coevolution.buffer import append_records
from recipe.verl.harness_rollout import VerlSample, generate


class ByteTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [byte + 3 for byte in text.encode()]

    def decode(self, tokens, **kwargs):
        return bytes(token - 3 for token in tokens if 3 <= token < 259).decode(errors="replace")

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        text = "".join(f"<{message['role']}>{message.get('content') or ''}</{message['role']}>" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.6]]))

    def features(self, prompt):
        bit = int("bit=1" in prompt)
        parity = float("Use parity" in prompt) * (2 * bit - 1)
        return torch.tensor([1.0, float(bit), parity])

    def forward(self, prompt):
        return self.features(prompt) @ self.weights

    @classmethod
    def load(cls, directory):
        policy = cls()
        policy.load_state_dict(torch.load(Path(directory) / "policy.pt", map_location="cpu", weights_only=True))
        return policy


class PolicyServer:
    def __init__(self, policy, tokenizer, seed):
        self.policy = policy
        self.tokenizer = tokenizer
        self.random = torch.Generator().manual_seed(seed)
        self.records = []

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None):
        prompt = self.tokenizer.decode(prompt_ids)
        with torch.no_grad():
            logprobs = self.policy(prompt).log_softmax(-1)
            action = (
                int(logprobs.argmax())
                if sampling_params.get("temperature") == 0
                else int(torch.multinomial(logprobs.exp(), 1, generator=self.random))
            )
        prefix = "<think>Compute the bit.</think>\n\n<answer>"
        text = f"{prefix}{action}</answer>"
        tokens = self.tokenizer.encode(text)
        probabilities = [0.0] * len(tokens)
        probabilities[len(self.tokenizer.encode(prefix))] = float(logprobs[action])
        self.records.append({"prompt": prompt, "action": action, "old_logprob": float(logprobs[action])})
        return SimpleNamespace(token_ids=tokens, log_probs=probabilities, stop_reason="stop")


class CPUBackend:
    def __init__(self, settings):
        self.settings = settings
        self.tokenizer = ByteTokenizer()
        torch.set_num_threads(1)

    async def initialize(self, root):
        if self.settings.initial_model:
            TinyPolicy.load(self.settings.initial_model)
            return Path(self.settings.initial_model).resolve()
        model = root / "initial_model"
        model.mkdir()
        torch.save(TinyPolicy().state_dict(), model / "policy.pt")
        return model

    @asynccontextmanager
    async def search(self, model, work):
        write_json(work / "loaded_model.json", {"path": str(model), "weights": model_digest(model)})
        yield

    async def episode(self, policy, harness, bit, server):
        sample = VerlSample(prompt=f"Question: bit={bit}. Return that bit.", label=str(bit))
        episode = await generate(
            sample,
            {"temperature": server.temperature, "max_new_tokens": 128},
            "http://cpu-policy/generate",
            self.tokenizer,
            server_manager=server,
            harness_config=harness,
        )
        if not episode or episode["status"] != "completed" or episode["eval_result"] is None:
            raise RuntimeError("CPU smoke rollout failed")
        return episode

    async def evaluate(self, model, harness, output):
        output.mkdir(parents=True)
        policy = TinyPolicy.load(model)
        server = PolicyServer(policy, self.tokenizer, self.settings.seed)
        server.temperature = 0
        scores = []
        for bit in (0, 1):
            episode = await self.episode(policy, harness, bit, server)
            scores.append(episode["eval_result"].score)
            (output / f"task_{bit}.md").write_text(episode["trajectory"].to_markdown(level="full"))
        write_json(output / "metrics.json", {"accuracy": sum(scores) / len(scores), "model": str(model)})
        return sum(scores) / len(scores)

    async def propose(self, model, incumbent, trajectories, output, index):
        output.mkdir(parents=True)
        config = HarnessConfig.from_yaml_file(incumbent)
        system = next(processor for processor in config.processors if processor["_target_"] == SYSTEM)
        system["system_builder"]["text"] += "\nUse parity to recover the requested bit."
        system["system_builder"]["text"] += f"\nCheck the answer (round {index})."
        path = output / "config.yaml"
        config.to_yaml_file(path)
        return path

    async def collect(self, model, pool, work, round_index):
        policy = TinyPolicy.load(model)
        server = PolicyServer(policy, self.tokenizer, self.settings.seed + int(work.name.split("_")[1]))
        server.temperature = 1
        source_model = model_digest(model)
        rewards_by_uid = {}
        records = []
        exploration_dir = work / "exploration"
        exploration_dir.mkdir(parents=True, exist_ok=True)
        for bit in (0, 1):
            task_uid = f"tiny-bit-{bit}"
            rewards_by_uid[task_uid] = []
            for harness_path in pool:
                episode = await self.episode(policy, harness_path, bit, server)
                record = server.records[-1]
                harness_id = hashlib.sha256(Path(harness_path).read_bytes()).hexdigest()
                trajectory_path = exploration_dir / f"task_{bit:04d}_{harness_id}.md"
                trajectory_path.write_text(episode["trajectory"].to_markdown(level="full"), encoding="utf-8")
                rewards_by_uid[task_uid].append(
                    {
                        "task_uid": task_uid,
                        "round": round_index,
                        "source_model": source_model,
                        "harness_id": harness_id,
                        "prompt": record["prompt"],
                        "action": int(record["action"]),
                        "old_logprob": float(record["old_logprob"]),
                        "reward": float(episode["eval_result"].reward),
                    }
                )
        for group in rewards_by_uid.values():
            records.extend(group)
        buffer_path = work / "buffer" / "trajectories.jsonl"
        append_records(buffer_path, records)
        return buffer_path

    async def train(self, model, harness, work):
        round_index = int(work.name.split("_")[1])
        if round_index <= self.settings.model_warmup_rounds:
            write_json(
                work / "training.json",
                {
                    "backend": "offline-buffer-grpo",
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
                Path(record["buffer"])
                for record in state.get("rounds", [])
                if record.get("buffer") and Path(record["buffer"]).is_file()
            )
        buffer_paths.append(work / "buffer" / "trajectories.jsonl")
        records = [
            json.loads(line)
            for buffer_path in buffer_paths
            for line in buffer_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        grouped = defaultdict(list)
        for record in records:
            grouped[record["task_uid"]].append(record)
        for group in grouped.values():
            group.sort(key=lambda record: (record["round"], record["harness_id"]))
        selected = []
        count = 0
        for group in reversed(list(grouped.values())):
            if count + len(group) > self.settings.buffer_capacity:
                if not selected:
                    raise ValueError("Buffer capacity is smaller than the newest complete cross-Harness task group")
                break
            selected.append(group)
            count += len(group)
        records = []
        for group in reversed(selected):
            if len({record["harness_id"] for record in group}) < 2:
                continue
            rewards = torch.tensor([record["reward"] for record in group])
            mean = rewards.mean()
            std = rewards.std(unbiased=False)
            for record in group:
                record["advantage"] = float((record["reward"] - mean) / (std + 1e-8)) if std > 0 else 0.0
                records.append(record)
        if not records or not any(abs(record["advantage"]) > 1e-12 for record in records):
            write_json(
                work / "training.json",
                {"backend": "offline-buffer-grpo", "model_updated": False, "reason": "no_cross_harness_groups"},
            )
            return None
        policy = TinyPolicy.load(model)
        reference = TinyPolicy.load(model)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
        updates = []
        for step in range(self.settings.steps):
            losses = []
            for record in records:
                logits = policy(record["prompt"])
                logprobs = logits.log_softmax(-1)
                ratio = (logprobs[record["action"]] - record["old_logprob"]).exp()
                advantage = torch.tensor(record["advantage"])
                surrogate = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
                with torch.no_grad():
                    ref_logprobs = reference(record["prompt"]).log_softmax(-1)
                kl = (logprobs.exp() * (logprobs - ref_logprobs)).sum()
                losses.append(surrogate + 0.001 * kl)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            if not all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in policy.parameters()
            ):
                raise RuntimeError("Nonfinite CPU policy gradient")
            optimizer.step()
            updates.append({"step": step + 1, "loss": float(loss.detach())})
        output = work / "model"
        output.mkdir()
        torch.save(policy.state_dict(), output / "policy.pt")
        write_json(
            work / "training.json",
            {
                "backend": "offline-buffer-grpo",
                "model_updated": True,
                "input_model": str(model),
                "accepted_harness": str(harness),
                "updates": updates,
                "buffer_records": len(records),
            },
        )
        return output
