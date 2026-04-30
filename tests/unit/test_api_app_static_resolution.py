# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from harnessx.api.app import _resolve_dist_file


class TestApiAppStaticResolution:
    def test_resolve_dist_file_returns_existing_file(self, tmp_path: Path) -> None:
        dist = tmp_path / "dist"
        dist.mkdir()
        target = dist / "favicon.svg"
        target.write_text("<svg/>", encoding="utf-8")

        resolved = _resolve_dist_file(dist, "favicon.svg")
        assert resolved == target.resolve()

    def test_resolve_dist_file_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        dist = tmp_path / "dist"
        dist.mkdir()

        assert _resolve_dist_file(dist, "missing.svg") is None

    def test_resolve_dist_file_blocks_path_traversal(self, tmp_path: Path) -> None:
        dist = tmp_path / "dist"
        dist.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("secret", encoding="utf-8")

        assert _resolve_dist_file(dist, "../secret.txt") is None
