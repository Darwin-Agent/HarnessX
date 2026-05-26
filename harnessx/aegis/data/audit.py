# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Append-only audit log (JSONL). Single source of truth per round."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator

_ALLOWED_KINDS = frozenset({
    "preprocess", "plan", "propose", "propose_fail",
    "critic_turn", "decision", "gate", "commit", "revert",
    "adjudicate", "journal",
    "ship_followup",  # C2: cross-round "ship didn't take" diagnostic
})


@dataclass
class AuditEvent:
    round: int
    stage: str
    kind: str
    payload: dict
    evidence_refs: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError(f"unknown kind: {self.kind}")


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: AuditEvent) -> None:
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(event), ensure_ascii=False))
            fp.write("\n")
            fp.flush()

    def read_all(self) -> Iterator[AuditEvent]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield AuditEvent(**obj)

    def query(self, *, round: int | None = None, kind: str | None = None) -> Iterable[AuditEvent]:
        for e in self.read_all():
            if round is not None and e.round != round:
                continue
            if kind is not None and e.kind != kind:
                continue
            yield e
