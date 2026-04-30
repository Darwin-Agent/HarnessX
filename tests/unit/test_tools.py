# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import pytest
from harnessx.tools.inmemory import InMemoryToolRegistry
from harnessx.tools.base import ToolConflictError

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fixtures.mock_tools import add_tool, echo_tool, fail_tool, make_registry


class TestTools:
    def test_registry_register_and_list(self):
        registry = make_registry(add_tool, echo_tool)
        names = registry.list_names()
        assert "add" in names
        assert "echo" in names

    def test_registry_get_schemas(self):
        registry = make_registry(add_tool)
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "add"

    @pytest.mark.asyncio
    async def test_registry_execute_sync_tool(self):
        registry = make_registry(add_tool)
        result = await registry.execute("add", {"a": 3, "b": 4})
        assert result.error is None
        assert result.output == "7"

    @pytest.mark.asyncio
    async def test_registry_execute_async_tool(self):
        registry = make_registry(echo_tool)
        result = await registry.execute("echo", {"message": "hello"})
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_registry_execute_missing_tool(self):
        registry = InMemoryToolRegistry()
        result = await registry.execute("nonexistent", {})
        assert result.error is not None
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_registry_execute_failing_tool(self):
        registry = make_registry(fail_tool)
        result = await registry.execute("fail_tool", {"message": "oops"})
        assert result.error is not None
        assert "always fails" in result.error

    # ---------------------------------------------------------------------------
    # ToolConflictError
    # ---------------------------------------------------------------------------

    def test_register_duplicate_raises_conflict_error(self):
        """Registering a tool with an already-taken name raises ToolConflictError."""
        registry = InMemoryToolRegistry()
        registry.register(add_tool)
        with pytest.raises(ToolConflictError, match="add"):
            registry.register(add_tool)

    def test_register_duplicate_error_message_contains_qualnames(self):
        """Error message includes module.qualname of both existing and new tool."""
        registry = InMemoryToolRegistry()
        registry.register(add_tool)
        with pytest.raises(ToolConflictError) as exc_info:
            registry.register(add_tool)
        msg = str(exc_info.value)
        assert "add" in msg
        assert "already registered" in msg

    def test_register_replace_true_overwrites_silently(self):
        """register(tool, replace=True) overwrites without raising."""
        registry = InMemoryToolRegistry()
        registry.register(add_tool)
        registry.register(add_tool, replace=True)  # must not raise
        assert "add" in registry.list_names()

    def test_register_different_names_no_conflict(self):
        """Two tools with distinct names register without error."""
        registry = InMemoryToolRegistry()
        registry.register(add_tool)
        registry.register(echo_tool)  # different name — must not raise
        assert "add" in registry.list_names()
        assert "echo" in registry.list_names()

    def test_tool_conflict_error_importable_from_top_level(self):
        """ToolConflictError is exported from the top-level harnessx package."""
        from harnessx import ToolConflictError as TCE

        assert TCE is ToolConflictError
