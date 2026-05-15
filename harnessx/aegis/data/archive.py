# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Archive — stores rejected / non-shipped candidates for cross-round memory."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ArchivedCandidate:
    round: int
    cid: str
    manifest_md: str
    failure_context: dict | None


class Archive:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _round_dir(self, round_n: int) -> Path:
        d = self.root / f"R{round_n}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def store(self, *, round_n: int, cid: str, manifest_md: str,
              failure_context: dict | None) -> Path:
        d = self._round_dir(round_n)
        md_path = d / f"{cid}.md"
        md_path.write_text(manifest_md, encoding="utf-8")
        if failure_context is not None:
            (d / f"{cid}.context.json").write_text(
                json.dumps(failure_context, ensure_ascii=False, indent=2)
            )
        return md_path

    def list_round(self, round_n: int) -> list[ArchivedCandidate]:
        d = self.root / f"R{round_n}"
        if not d.exists():
            return []
        out = []
        for md in sorted(d.glob("*.md")):
            cid = md.stem
            ctx_path = d / f"{cid}.context.json"
            ctx = json.loads(ctx_path.read_text()) if ctx_path.exists() else None
            out.append(ArchivedCandidate(
                round=round_n, cid=cid,
                manifest_md=md.read_text(encoding="utf-8"),
                failure_context=ctx,
            ))
        return out

    def list_all(self) -> list[ArchivedCandidate]:
        out = []
        for rd in sorted(self.root.glob("R*")):
            try:
                round_n = int(rd.name[1:])
            except ValueError:
                continue
            out.extend(self.list_round(round_n))
        return out

    def recent(self, n: int) -> list[ArchivedCandidate]:
        all_items = self.list_all()
        all_items.sort(key=lambda c: c.round)
        return all_items[-n:]
