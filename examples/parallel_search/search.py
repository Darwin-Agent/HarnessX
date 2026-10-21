# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Search and fetch with Parallel through HarnessX's MCP tool registry."""

import argparse
import asyncio
import os
import uuid

from harnessx.tools.mcp import MCPToolRegistry


async def search(objective, queries, urls):
    endpoint = os.environ.get("MCP_SEARCH_URL", "").strip()
    if not endpoint:
        raise RuntimeError("Set MCP_SEARCH_URL to the search service endpoint before running this example.")
    registry = MCPToolRegistry(
        transport="http",
        url=endpoint,
        name="parallel-search",
        headers={"User-Agent": "HarnessX/0.1.0 (Parallel Search example)"},
    )
    session_id = uuid.uuid4().hex
    try:
        await registry.connect()
        result = await registry.execute(
            "web_search",
            {"objective": objective, "search_queries": queries, "session_id": session_id},
        )
        if result.error:
            raise RuntimeError(result.error)
        print(result.output)
        if urls:
            result = await registry.execute(
                "web_fetch",
                {"urls": urls, "objective": objective[:200], "search_queries": queries, "session_id": session_id},
            )
            if result.error:
                raise RuntimeError(result.error)
            print(result.output)
    finally:
        await registry.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objective", help="What you want to find")
    parser.add_argument("--query", action="append", required=True, help="Search query; repeat for related queries")
    parser.add_argument(
        "--fetch-url", action="append", default=[], help="Also fetch this URL; repeat for multiple pages"
    )
    args = parser.parse_args()
    if len(args.fetch_url) > 20:
        parser.error("--fetch-url supports at most 20 URLs per request")
    asyncio.run(search(args.objective, args.query, args.fetch_url))


if __name__ == "__main__":
    main()
