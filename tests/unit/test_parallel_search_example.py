# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from examples.parallel_search import search as example
from harnessx.tools.base import ToolResult
from harnessx.tools.mcp import MCPClient, MCPToolRegistry

TEST_SEARCH_URL = "https://search-service.invalid/mcp"


@pytest.mark.parametrize("headers", [None, {"User-Agent": "HarnessX/test"}])
async def test_http_headers_reach_transport(monkeypatch, headers):
    from contextlib import AsyncExitStack
    from mcp.client import streamable_http

    observed = []

    @asynccontextmanager
    async def transport(url, **kwargs):
        observed.append((url, kwargs))
        yield ("read", "write", None)

    monkeypatch.setattr(streamable_http, "streamablehttp_client", transport)
    client = MCPClient("http", url=TEST_SEARCH_URL, headers=headers)
    async with AsyncExitStack() as stack:
        assert await client._enter_http(stack) == ("read", "write")
    assert observed == [(TEST_SEARCH_URL, {} if headers is None else {"headers": headers})]


async def test_registry_passes_headers_to_client(monkeypatch):
    observed = []

    class Client:
        def __init__(self, **kwargs):
            observed.append(kwargs)

        async def connect(self):
            pass

        async def list_tools(self):
            return []

        async def disconnect(self):
            pass

    monkeypatch.setattr("harnessx.tools.mcp.MCPClient", Client)
    headers = {"User-Agent": "HarnessX/test"}
    registry = MCPToolRegistry(transport="http", url=TEST_SEARCH_URL, headers=headers)
    headers["User-Agent"] = "mutated"
    await registry.connect()
    assert observed[0]["url"] == TEST_SEARCH_URL
    assert observed[0]["headers"] == {"User-Agent": "HarnessX/test"}
    await registry.disconnect()


@pytest.mark.parametrize("urls", [[], ["https://example.com"]])
@pytest.mark.parametrize("endpoint", [None, TEST_SEARCH_URL])
async def test_example_dispatches_and_closes(monkeypatch, capsys, urls, endpoint):
    calls = []
    if endpoint is None:
        monkeypatch.delenv("MCP_SEARCH_URL", raising=False)
    else:
        monkeypatch.setenv("MCP_SEARCH_URL", endpoint)

    class Registry:
        def __init__(self, **kwargs):
            assert kwargs == {
                "transport": "http",
                "url": endpoint or example.DEFAULT_SEARCH_URL,
                "name": "parallel-search",
                "headers": {"User-Agent": "HarnessX/0.1.0 (Parallel Search example)"},
            }

        async def connect(self):
            calls.append("connect")

        async def execute(self, name, arguments):
            calls.append((name, arguments))
            return ToolResult(output=f"{name} excerpts")

        async def disconnect(self):
            calls.append("disconnect")

    monkeypatch.setattr(example, "MCPToolRegistry", Registry)
    await example.search("Find sources", ["related query"], urls)
    assert calls[0] == "connect"
    assert calls[-1] == "disconnect"
    name, arguments = calls[1]
    assert name == "web_search"
    assert arguments["objective"] == "Find sources"
    assert arguments["search_queries"] == ["related query"]
    assert uuid.UUID(hex=arguments["session_id"]).version == 4
    if urls:
        assert calls[2] == (
            "web_fetch",
            {
                "urls": urls,
                "objective": "Find sources",
                "search_queries": ["related query"],
                "session_id": arguments["session_id"],
            },
        )
    else:
        assert len(calls) == 3
    assert "web_search excerpts" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["connect", "web_search", "web_fetch"])
async def test_example_reports_errors_and_closes(monkeypatch, failure):
    closed = []

    class Registry:
        def __init__(self, **kwargs):
            pass

        async def connect(self):
            if failure == "connect":
                raise RuntimeError("unavailable")

        async def execute(self, name, arguments):
            return SimpleNamespace(error="unavailable" if failure == name else None, output="excerpts")

        async def disconnect(self):
            closed.append(True)

    monkeypatch.setattr(example, "MCPToolRegistry", Registry)
    with pytest.raises(RuntimeError, match="unavailable"):
        await example.search("Find sources", ["related query"], ["https://example.com"])
    assert closed == [True]
