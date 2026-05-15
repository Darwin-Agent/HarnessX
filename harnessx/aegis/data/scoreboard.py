# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Scoreboard — persistent structured state of the evolution run.

Per-bucket hit-rate rollup + raw `ships` ledger. The Planner / Critic read
this + ship_outcomes.json and reason about portfolio drift or under-shipped
buckets in natural language — the scoreboard itself does NOT compute a
coverage grid or under-exploration flags (those were shown to be too rigid
in the v0.9 pilot; agent judgment handles them better).

Persistent state; atomic write on save. Every stage's input contract may
reference this file — it is NOT a free-text artefact.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShipRecord:
    cid: str
    round: int
    bucket: str
    predicted_tasks: tuple[str, ...]
    flipped_in_ship_round: tuple[str, ...]

    def hit_rate(self) -> float:
        if not self.predicted_tasks:
            return 0.0
        return len(self.flipped_in_ship_round) / len(self.predicted_tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "round": self.round,
            "bucket": self.bucket,
            "predicted_tasks": list(self.predicted_tasks),
            "flipped_in_ship_round": list(self.flipped_in_ship_round),
            "hit_rate": round(self.hit_rate(), 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShipRecord":
        return cls(
            cid=str(d.get("cid", "")),
            round=int(d.get("round", 0)),
            bucket=str(d.get("bucket", "")),
            predicted_tasks=tuple(d.get("predicted_tasks") or ()),
            flipped_in_ship_round=tuple(d.get("flipped_in_ship_round") or ()),
        )


@dataclass
class Scoreboard:
    version: int = 1
    last_updated_round: int = 0
    ships: list[ShipRecord] = field(default_factory=list)

    def add_ship(self, rec: ShipRecord) -> None:
        # Idempotent on re-run: drop prior entry with same cid before append.
        self.ships = [s for s in self.ships if s.cid != rec.cid]
        self.ships.append(rec)
        self.ships.sort(key=lambda s: (s.round, s.cid))

    def _rollup_bucket(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for s in self.ships:
            b = s.bucket or "unknown"
            entry = out.setdefault(b, {"ships": 0, "predicted": 0, "flipped": 0})
            entry["ships"] += 1
            entry["predicted"] += len(s.predicted_tasks)
            entry["flipped"] += len(s.flipped_in_ship_round)
        for e in out.values():
            e["hit_rate"] = (
                round(e["flipped"] / e["predicted"], 4) if e["predicted"] else 0.0
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "last_updated_round": self.last_updated_round,
            "ships": [s.to_dict() for s in self.ships],
            "by_bucket": self._rollup_bucket(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "Scoreboard":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()
        sb = cls(
            version=int(data.get("version", 1)),
            last_updated_round=int(data.get("last_updated_round", 0)),
        )
        for entry in data.get("ships") or []:
            try:
                sb.ships.append(ShipRecord.from_dict(entry))
            except Exception:
                continue
        sb.ships.sort(key=lambda s: (s.round, s.cid))
        return sb
