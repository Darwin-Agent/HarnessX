# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

from harnessx.api.routes import run as run_route


class TestApiRunSlotConfig:
    def test_persist_slot_config_writes_agent_home_file(self, tmp_path, monkeypatch) -> None:
        home = tmp_path / "hx-home"
        monkeypatch.setenv("HARNESSX_HOME", str(home))

        run_route._persist_slot_config(
            {
                "enabled_tools": ["Read", "Write"],
                "enabled_skills": ["pptx"],
                "sandbox_type": "local",
                "sandbox_url": None,
            }
        )

        path = home / "slot_config.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["enabled_tools"] == ["Read", "Write"]
        assert data["enabled_skills"] == ["pptx"]
