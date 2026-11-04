# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Locate code with Sem; optionally preview/apply explicit Weave edits."""

import argparse
import asyncio
import json
from pathlib import Path

from harnessx.tools.mcp import MCPToolRegistry


async def call(registry, name, arguments):
    try:
        await registry.connect()
        result = await registry.execute(name, arguments)
        if result.error:
            raise RuntimeError(result.error)
        print(result.output)
    finally:
        await registry.disconnect()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("query")
    parser.add_argument("--edits", type=Path, help="JSON array of Weave edit specifications")
    parser.add_argument("--apply", action="store_true", help="Write the supplied edits (default: dry run)")
    args = parser.parse_args()
    if args.apply and not args.edits:
        parser.error("--apply requires --edits")
    root = args.repo.resolve(strict=True)
    if not root.is_dir():
        parser.error("repo must be a directory")
    edits = None
    if args.edits:
        edits = json.loads(args.edits.read_text())
        if not isinstance(edits, list) or not edits:
            parser.error("--edits must contain a nonempty JSON array")
    await call(MCPToolRegistry(
        transport="stdio", command="sem mcp", name="sem",
        env={"SEM_REPO": str(root), "SEM_LOCAL": "1", "SEM_MCP_NO_SHARED": "1",
             "SEM_NO_NETWORK": "1", "SEM_NO_TELEMETRY": "1"},
    ), "sem_find", {"query": args.query})
    if edits is not None:
        await call(MCPToolRegistry(
            transport="stdio", command="weave-mcp", name="weave",
            env={"WEAVE_REPO": str(root)},
        ), "weave_edit", {"edits": edits, "dry_run": not args.apply})


if __name__ == "__main__":
    asyncio.run(main())
