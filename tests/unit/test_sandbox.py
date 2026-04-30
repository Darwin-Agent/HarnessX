# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import pytest
from pathlib import Path

from harnessx.sandbox.base import (
    Mount,
    _sandbox_ctx,
    get_current_sandbox,
)
from harnessx.sandbox.local import LocalSandbox, LocalSandboxProvider
from harnessx.workspace.workspace import (
    Workspace,
    WorkspaceEscapeError,
    WorkspaceWriteError,
)


# ---------------------------------------------------------------------------
# Mount
# ---------------------------------------------------------------------------


class _FakeExecResult:
    """Mimics the tuple returned by container.exec_run(demux=True)."""

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", exit_code: int = 0):
        self.exit_code = exit_code
        self._stdout = stdout
        self._stderr = stderr

    def __iter__(self):
        yield self.exit_code
        yield (self._stdout, self._stderr)


def _make_fake_container(stdout: bytes = b"hello\n", stderr: bytes = b""):
    from unittest.mock import MagicMock

    container = MagicMock()
    container.status = "running"
    container.short_id = "abc123"
    container.exec_run.return_value = _FakeExecResult(stdout=stdout, stderr=stderr)
    return container


def _make_fake_e2b_sbx(stdout: str = "ok", stderr: str = ""):
    from unittest.mock import AsyncMock, MagicMock

    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr

    commands = MagicMock()
    commands.run = AsyncMock(return_value=result)

    files = MagicMock()
    files.read = AsyncMock(return_value="file content")
    files.write = AsyncMock(return_value=None)
    files.list = AsyncMock(return_value=[])

    sbx = MagicMock()
    sbx.commands = commands
    sbx.files = files
    sbx.sandbox_id = "sbx-test-123"
    sbx.kill = AsyncMock()
    sbx.pause = AsyncMock(return_value="sbx-paused-123")
    return sbx


class TestSandbox:
    def test_mount_resolves_host_path(self, tmp_path):
        m = Mount(host_path=tmp_path, container_path="/workspace")
        assert m.host_path == tmp_path
        assert m.container_path == "/workspace"
        assert not m.read_only

    # ---------------------------------------------------------------------------
    # LocalSandbox — resolve / jail
    # ---------------------------------------------------------------------------

    def test_local_sandbox_resolve_relative(self, tmp_path):
        sb = LocalSandbox(root=tmp_path, mode="isolated")
        assert sb.resolve("foo.txt") == str(tmp_path / "foo.txt")

    def test_local_sandbox_resolve_absolute_inside(self, tmp_path):
        sb = LocalSandbox(root=tmp_path, mode="isolated")
        assert sb.resolve(str(tmp_path / "a" / "b.txt")) == str(tmp_path / "a" / "b.txt")

    def test_local_sandbox_jail_escape_raises(self, tmp_path):
        sb = LocalSandbox(root=tmp_path, mode="isolated")
        with pytest.raises(WorkspaceEscapeError):
            sb.resolve("../../etc/passwd")

    def test_local_sandbox_shared_allows_sibling(self, tmp_path):
        child = tmp_path / "agents" / "a1"
        child.mkdir(parents=True)
        sb = LocalSandbox(root=child, mode="shared")
        # Shared mode allows access within the parent directory (agents/)
        resolved = sb.resolve("../a2/file.txt")
        assert "a2" in resolved  # Does not raise, resolves within agents/

    def test_local_sandbox_readonly_check_write(self, tmp_path):
        sb = LocalSandbox(root=tmp_path, mode="readonly")
        with pytest.raises(WorkspaceWriteError):
            sb.check_write()

    # ---------------------------------------------------------------------------
    # LocalSandbox — exec / read_file / write_file / list_dir
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_local_sandbox_exec(self, tmp_path):
        sb = LocalSandbox(root=tmp_path)
        result = await sb.exec("echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_local_sandbox_exec_stderr(self, tmp_path):
        sb = LocalSandbox(root=tmp_path)
        result = await sb.exec("echo err >&2")
        assert "err" in result

    @pytest.mark.asyncio
    async def test_local_sandbox_exec_timeout(self, tmp_path):
        sb = LocalSandbox(root=tmp_path)
        result = await sb.exec("sleep 10", timeout=0.05)
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_local_sandbox_read_write(self, tmp_path):
        sb = LocalSandbox(root=tmp_path)
        path = str(tmp_path / "hello.txt")
        await sb.write_file(path, "world")
        content = await sb.read_file(path)
        assert content == "world"

    @pytest.mark.asyncio
    async def test_local_sandbox_write_creates_parents(self, tmp_path):
        sb = LocalSandbox(root=tmp_path)
        nested = str(tmp_path / "a" / "b" / "c.txt")
        await sb.write_file(nested, "deep")
        assert Path(nested).read_text() == "deep"

    @pytest.mark.asyncio
    async def test_local_sandbox_list_dir(self, tmp_path):
        sb = LocalSandbox(root=tmp_path)
        (tmp_path / "file.txt").write_text("x")
        (tmp_path / "subdir").mkdir()
        entries = await sb.list_dir(str(tmp_path))
        assert "subdir/" in entries
        assert "file.txt" in entries

    @pytest.mark.asyncio
    async def test_local_sandbox_glob(self, tmp_path):
        sb = LocalSandbox(root=tmp_path)
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        matches = await sb.glob_files("*.py", base=str(tmp_path))
        assert "a.py" in matches
        assert "b.py" in matches
        assert "c.txt" not in matches

    @pytest.mark.asyncio
    async def test_local_sandbox_grep(self, tmp_path):
        sb = LocalSandbox(root=tmp_path)
        (tmp_path / "code.py").write_text("def hello():\n    pass\n")
        result = await sb.grep_files("def hello", path=str(tmp_path))
        assert "hello" in result

    # ---------------------------------------------------------------------------
    # LocalSandboxProvider
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_local_provider_acquire_with_workspace(self, tmp_path):
        ws = Workspace(root=tmp_path, agent_id="test")
        provider = LocalSandboxProvider()
        sb = await provider.acquire(workspace=ws)
        assert isinstance(sb, LocalSandbox)
        assert sb.workspace_path == str(tmp_path)

    @pytest.mark.asyncio
    async def test_local_provider_acquire_without_workspace(self):
        provider = LocalSandboxProvider()
        sb = await provider.acquire()
        assert isinstance(sb, LocalSandbox)

    @pytest.mark.asyncio
    async def test_local_provider_release_is_noop(self, tmp_path):
        ws = Workspace(root=tmp_path, agent_id="test")
        provider = LocalSandboxProvider()
        sb = await provider.acquire(workspace=ws)
        await provider.release(sb)  # Should not raise

    # ---------------------------------------------------------------------------
    # ContextVar injection
    # ---------------------------------------------------------------------------

    def test_get_current_sandbox_default_none(self):
        assert get_current_sandbox() is None

    @pytest.mark.asyncio
    async def test_sandbox_ctx_set_and_get(self, tmp_path):
        sb = LocalSandbox(root=tmp_path)
        token = _sandbox_ctx.set(sb)
        try:
            assert get_current_sandbox() is sb
        finally:
            _sandbox_ctx.reset(token)
        assert get_current_sandbox() is None

    # ---------------------------------------------------------------------------
    # Builtin tools — sandbox-aware via get_current_sandbox() ContextVar
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_bash_tool_uses_sandbox_cwd(self, tmp_path):
        """bash_tool uses sandbox.workspace_path as cwd when ContextVar is set."""
        from harnessx.tools.builtin.bash import bash_tool

        sandbox = LocalSandbox(root=tmp_path)
        token = _sandbox_ctx.set(sandbox)
        try:
            result = await bash_tool.fn(command="pwd")
            assert str(tmp_path) in result
        finally:
            _sandbox_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_read_write_tools_route_through_sandbox(self, tmp_path):
        """read_tool and write_tool route through sandbox when ContextVar is set."""
        from harnessx.tools.builtin.read import read_tool
        from harnessx.tools.builtin.write import write_tool

        sandbox = LocalSandbox(root=tmp_path, mode="isolated")
        token = _sandbox_ctx.set(sandbox)
        try:
            # write relative path — resolves to tmp_path/hello.txt via sandbox
            await write_tool.fn(file_path="hello.txt", content="sandbox test")
            assert (tmp_path / "hello.txt").read_text() == "sandbox test"

            result = await read_tool.fn(file_path="hello.txt")
            assert "sandbox test" in result
        finally:
            _sandbox_ctx.reset(token)

    @pytest.mark.asyncio
    async def test_tools_fallback_without_sandbox(self, tmp_path):
        """Without a sandbox in ContextVar, tools fall back to direct os calls."""
        from harnessx.tools.builtin.read import read_tool
        from harnessx.tools.builtin.write import write_tool

        target = tmp_path / "fallback.txt"
        # No ContextVar set — write_tool uses os.path.abspath (absolute path required)
        await write_tool.fn(file_path=str(target), content="fallback")
        result = await read_tool.fn(file_path=str(target))
        assert "fallback" in result

    @pytest.mark.asyncio
    async def test_sandbox_ctx_controls_tool_cwd(self, tmp_path):
        """Swapping the ContextVar sandbox immediately changes where bash runs."""
        from harnessx.tools.builtin.bash import bash_tool

        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()

        token = _sandbox_ctx.set(LocalSandbox(root=root_a))
        try:
            result_a = await bash_tool.fn(command="pwd")
            assert str(root_a) in result_a
        finally:
            _sandbox_ctx.reset(token)

        token = _sandbox_ctx.set(LocalSandbox(root=root_b))
        try:
            result_b = await bash_tool.fn(command="pwd")
            assert str(root_b) in result_b
        finally:
            _sandbox_ctx.reset(token)

    # ---------------------------------------------------------------------------
    # HarnessConfig defaults to LocalSandboxProvider
    # ---------------------------------------------------------------------------

    def test_harness_config_default_sandbox_provider(self):
        from harnessx.core.harness import HarnessConfig, _instantiate_runtime

        config = HarnessConfig()
        rt = _instantiate_runtime(config)
        assert isinstance(rt.sandbox_provider, LocalSandboxProvider)

    # ---------------------------------------------------------------------------
    # Workspace.extra_mounts
    # ---------------------------------------------------------------------------

    def test_workspace_extra_mounts(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        ws = Workspace(
            root=tmp_path / "agent",
            agent_id="pa",
            extra_mounts=[Mount(host_path=skills, container_path="/skills", read_only=True)],
        )
        assert len(ws.extra_mounts) == 1
        assert ws.extra_mounts[0].container_path == "/skills"
        assert ws.extra_mounts[0].read_only

    def test_workspace_child_inherits_extra_mounts(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        parent = Workspace(
            root=tmp_path / "agent",
            agent_id="pa",
            extra_mounts=[Mount(host_path=skills, container_path="/skills", read_only=True)],
        )
        child = parent.child("worker-1")
        assert len(child.extra_mounts) == 1
        assert child.extra_mounts[0].container_path == "/skills"

    # ---------------------------------------------------------------------------
    # DockerSandbox (mocked docker-py — no real daemon needed)
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_docker_sandbox_exec_returns_stdout(self):
        from harnessx.sandbox.docker import DockerSandbox

        container = _make_fake_container(stdout=b"hello\n")
        sb = DockerSandbox(container)
        result = await sb.exec("echo hello")
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_docker_sandbox_exec_includes_stderr(self):
        from harnessx.sandbox.docker import DockerSandbox

        container = _make_fake_container(stdout=b"", stderr=b"err msg\n")
        sb = DockerSandbox(container)
        result = await sb.exec("bad_cmd")
        assert "err msg" in result

    @pytest.mark.asyncio
    async def test_docker_sandbox_workspace_path(self):
        from harnessx.sandbox.docker import DockerSandbox

        container = _make_fake_container()
        sb = DockerSandbox(container)
        assert sb.workspace_path == "/workspace"

    @pytest.mark.asyncio
    async def test_docker_sandbox_write_then_read(self, tmp_path):
        """write_file uses base64 + exec; read_file uses cat via exec."""
        from harnessx.sandbox.docker import DockerSandbox
        from unittest.mock import MagicMock

        calls: list[str] = []

        container = MagicMock()
        container.status = "running"
        container.short_id = "abc123"

        def fake_exec_run(cmd, **kwargs):
            calls.append(cmd[-1] if isinstance(cmd, list) else cmd)
            return _FakeExecResult(stdout=b"file content\n")

        container.exec_run.side_effect = fake_exec_run
        sb = DockerSandbox(container)

        await sb.write_file("/workspace/test.txt", "file content")
        assert any("base64" in c for c in calls), "write_file should use base64 encoding"

        result = await sb.read_file("/workspace/test.txt")
        assert "file content" in result

    @pytest.mark.asyncio
    async def test_docker_provider_import_error_without_package(self, monkeypatch):
        """DockerSandboxProvider raises ImportError when docker-py is not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "docker":
                raise ImportError("No module named 'docker'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pip install harnessx"):
            from harnessx.sandbox.docker import DockerSandboxProvider

            DockerSandboxProvider()

    @pytest.mark.asyncio
    async def test_docker_provider_release_stops_ephemeral_container(self):
        """release() stops containers not in the warm pool."""
        from harnessx.sandbox.docker import DockerSandbox, DockerSandboxProvider
        from unittest.mock import MagicMock

        provider = DockerSandboxProvider.__new__(DockerSandboxProvider)
        provider._pool = {}
        provider._lock = asyncio.Lock()

        container = _make_fake_container()
        container.stop = MagicMock()
        sb = DockerSandbox(container)

        await provider.release(sb)
        container.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_docker_provider_release_keeps_warm_pool_container(self):
        """release() does NOT stop containers in the warm pool."""
        from harnessx.sandbox.docker import DockerSandbox, DockerSandboxProvider
        from unittest.mock import MagicMock

        provider = DockerSandboxProvider.__new__(DockerSandboxProvider)
        provider._lock = asyncio.Lock()

        container = _make_fake_container()
        container.stop = MagicMock()
        provider._pool = {"user-alice": container}

        sb = DockerSandbox(container)
        await provider.release(sb)
        container.stop.assert_not_called()

    def test_docker_provider_build_volumes_with_workspace(self, tmp_path):
        """_build_volumes maps workspace.root → /workspace."""
        from harnessx.sandbox.docker import DockerSandboxProvider, _CONTAINER_WORKSPACE
        from unittest.mock import patch

        with patch("harnessx.sandbox.docker.DockerSandboxProvider.__init__", return_value=None):
            provider = DockerSandboxProvider.__new__(DockerSandboxProvider)

        agent_dir = tmp_path / "agent"
        agent_dir.mkdir(exist_ok=True)
        ws = Workspace(root=agent_dir, agent_id="test")
        vols = provider._build_volumes(ws)

        assert str(ws.root) in vols
        assert vols[str(ws.root)]["bind"] == _CONTAINER_WORKSPACE
        assert vols[str(ws.root)]["mode"] == "rw"

    def test_docker_provider_build_volumes_with_extra_mounts(self, tmp_path):
        """extra_mounts are added as read-only volumes."""
        from harnessx.sandbox.docker import DockerSandboxProvider
        from unittest.mock import patch

        skills = tmp_path / "skills"
        skills.mkdir()
        (tmp_path / "agent").mkdir(exist_ok=True)

        with patch("harnessx.sandbox.docker.DockerSandboxProvider.__init__", return_value=None):
            provider = DockerSandboxProvider.__new__(DockerSandboxProvider)

        ws = Workspace(
            root=tmp_path / "agent",
            agent_id="test",
            extra_mounts=[Mount(host_path=skills, container_path="/skills", read_only=True)],
        )
        vols = provider._build_volumes(ws)
        assert str(skills) in vols
        assert vols[str(skills)] == {"bind": "/skills", "mode": "ro"}

    # ---------------------------------------------------------------------------
    # E2BSandbox (mocked e2b SDK — no real API key needed)
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_e2b_sandbox_exec_returns_stdout(self):
        from harnessx.sandbox.e2b import E2BSandbox

        sbx = _make_fake_e2b_sbx(stdout="hello world")
        sb = E2BSandbox(sbx)
        result = await sb.exec("echo hello world")
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_e2b_sandbox_exec_includes_stderr(self):
        from harnessx.sandbox.e2b import E2BSandbox

        sbx = _make_fake_e2b_sbx(stdout="", stderr="error!")
        sb = E2BSandbox(sbx)
        result = await sb.exec("bad_cmd")
        assert "error!" in result

    @pytest.mark.asyncio
    async def test_e2b_sandbox_workspace_path(self):
        from harnessx.sandbox.e2b import E2BSandbox

        sbx = _make_fake_e2b_sbx()
        sb = E2BSandbox(sbx)
        assert sb.workspace_path == "/workspace"

    @pytest.mark.asyncio
    async def test_e2b_sandbox_read_file(self):
        from harnessx.sandbox.e2b import E2BSandbox

        sbx = _make_fake_e2b_sbx()
        sbx.files.read.return_value = "hello from e2b"
        sb = E2BSandbox(sbx)
        result = await sb.read_file("/workspace/hello.txt")
        assert result == "hello from e2b"
        sbx.files.read.assert_called_once_with("/workspace/hello.txt")

    @pytest.mark.asyncio
    async def test_e2b_sandbox_write_file(self):
        from harnessx.sandbox.e2b import E2BSandbox

        sbx = _make_fake_e2b_sbx()
        sb = E2BSandbox(sbx)
        await sb.write_file("/workspace/out.txt", "content")
        sbx.files.write.assert_called_once_with("/workspace/out.txt", "content")

    @pytest.mark.asyncio
    async def test_e2b_provider_import_error_without_package(self, monkeypatch):
        """E2BSandboxProvider raises ImportError when e2b is not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "e2b":
                raise ImportError("No module named 'e2b'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pip install harnessx"):
            from harnessx.sandbox.e2b import E2BSandboxProvider

            E2BSandboxProvider()

    @pytest.mark.asyncio
    async def test_e2b_provider_release_kills_sandbox(self):
        """release() calls sbx.kill() when warm_pool=False."""
        from harnessx.sandbox.e2b import E2BSandbox, E2BSandboxProvider

        provider = E2BSandboxProvider.__new__(E2BSandboxProvider)
        provider.warm_pool = False
        provider._paused = {}

        sbx = _make_fake_e2b_sbx()
        sb = E2BSandbox(sbx)
        await provider.release(sb)
        sbx.kill.assert_called_once()

    # ---------------------------------------------------------------------------
    # Lazy import via __getattr__
    # ---------------------------------------------------------------------------

    def test_sandbox_module_lazy_docker_import(self):
        """harnessx.sandbox.DockerSandboxProvider is importable via __getattr__."""
        from harnessx import sandbox

        cls = sandbox.DockerSandboxProvider
        from harnessx.sandbox.docker import DockerSandboxProvider

        assert cls is DockerSandboxProvider

    def test_sandbox_module_lazy_e2b_import(self):
        """harnessx.sandbox.E2BSandboxProvider is importable via __getattr__."""
        from harnessx import sandbox

        cls = sandbox.E2BSandboxProvider
        from harnessx.sandbox.e2b import E2BSandboxProvider

        assert cls is E2BSandboxProvider

    def test_sandbox_module_unknown_attr_raises(self):
        from harnessx import sandbox

        with pytest.raises(AttributeError):
            _ = sandbox.NonExistentProvider
