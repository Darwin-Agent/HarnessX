# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from types import SimpleNamespace

import pytest

from harnessx.core.events import TaskStartEvent
from harnessx.core.processor import pipe
from harnessx.processors.context.env_context_injector import EnvironmentContextInjector


class TestEnvContextInjector:
    @pytest.mark.asyncio
    async def test_task_start_prefers_workspace_root_and_exposes_project_path(self, tmp_path, monkeypatch):
        project_root = (tmp_path / "project").resolve()
        project_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(project_root)

        workspace_root = (tmp_path / "agent_ws").resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)

        proc = EnvironmentContextInjector(
            working_dir="/tmp/placeholder",
            inject_workspace_tree=False,
            inject_integrity_rules=False,
            non_interactive=False,
        )

        event = TaskStartEvent(
            run_id="r1",
            step_id=0,
            task_description="demo",
            system_prompt="",
            workspace=SimpleNamespace(root=workspace_root),
            tools=(),
        )

        out = await pipe(event, [proc])
        assert out is not None

        prompt = out.system_prompt
        ws_line = f"- Agent workspace path: `{workspace_root}`"
        prj_line = f"- Project path: `{project_root}`"
        assert ws_line in prompt
        assert prj_line in prompt
        assert prompt.index(ws_line) < prompt.index(prj_line)
