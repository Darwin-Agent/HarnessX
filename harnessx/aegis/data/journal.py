# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Journal — cross-round first-person memo + refuted_signatures ledger."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

_log = logging.getLogger(__name__)


@dataclass
class RoundEntry:
    round: int
    action: str  # "ship" | "rollback" | "no_op"
    shipped_cid: str | None   # back-compat: first shipped (or None)
    hypothesis_signatures: list[str]
    refuted_signatures: list[str]
    hit_rate: float | None
    narrative: str
    predicted_tasks_pass: list[str] = field(default_factory=list)
    # Multi-ship: the full list of shipped candidates this round (may be
    # empty for no_op, single-element for 1-ship, multiple when Critic
    # shipped orthogonal-bucket candidates together). ``shipped_cid`` is
    # retained as shipped_cids[0] for back-compat with older consumers.
    shipped_cids: list[str] = field(default_factory=list)
    # Per-cid predicted-tasks map so C2 follow-up can attribute predictions
    # to the ship that made them. Empty for no_op rounds.
    predicted_tasks_by_cid: dict[str, list[str]] = field(default_factory=dict)


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: RoundEntry) -> None:
        frontmatter = json.dumps(asdict(entry), ensure_ascii=False, indent=2)
        block = (
            f"\n## Round {entry.round}\n"
            f"```json\n{frontmatter}\n```\n\n"
            f"{entry.narrative}\n"
        )
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(block)

    def read_all(self) -> list[RoundEntry]:
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        out = []
        for block in text.split("\n## Round "):
            if "```json" not in block:
                continue
            try:
                json_start = block.index("```json\n") + len("```json\n")
                json_end = block.index("\n```", json_start)
                data = json.loads(block[json_start:json_end])
                out.append(RoundEntry(**data))
            except (ValueError, json.JSONDecodeError) as exc:
                _log.warning(
                    "Skipping malformed journal block: %s", exc,
                )
                continue
        return out

    def recent(self, window: int) -> list[RoundEntry]:
        return self.read_all()[-window:]

    def all_refuted_signatures(self) -> set[str]:
        out: set[str] = set()
        for e in self.read_all():
            out.update(e.refuted_signatures)
        return out
