# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Publish an HTML page to a public deed.page URL through HarnessX's MCP tool registry."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from harnessx.tools.mcp import MCPToolRegistry

SAMPLE_PAGE = """<!doctype html>
<title>HarnessX report</title>
<h1>Hello from HarnessX</h1>
<p>Published through the deed.page MCP example.</p>
"""


def _registry():
    return MCPToolRegistry(
        transport="http",
        url="https://deed.page/mcp",
        name="deed-page",
        headers={"User-Agent": "HarnessX/0.1.0 (deed.page example)"},
    )


async def publish(html, slug=None, ttl_seconds=3600, token=None):
    """Publish ``html`` as index.html and return the tool output (url, slug, token).

    Pass the owner ``token`` from the first publish to update an existing slug.
    """
    arguments = {"files": {"index.html": html}}
    if slug:
        arguments["slug"] = slug
    if ttl_seconds:
        arguments["ttl_seconds"] = ttl_seconds
    if token:
        arguments["token"] = token
    return await _call("publish_site", arguments)


async def delete(slug, token):
    """Delete a page with the owner token returned by its first publish."""
    return await _call("delete_site", {"slug": slug, "token": token})


async def _call(name, arguments):
    registry = _registry()
    try:
        await registry.connect()
        result = await registry.execute(name, arguments)
        if result.error:
            raise RuntimeError(result.error)
        print(result.output)
        return result.output
    finally:
        await registry.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help="HTML file to publish (default: a small sample page)")
    parser.add_argument("--slug", help="Optional slug for https://{slug}.deed.page/")
    parser.add_argument("--ttl", type=int, default=3600, help="Expire after this many seconds; 0 keeps the page")
    parser.add_argument("--delete", metavar="SLUG", help="Delete this page instead (needs DEEDPAGE_TOKEN)")
    args = parser.parse_args()
    if args.ttl < 0:
        parser.error("--ttl must not be negative")
    if args.delete:
        token = os.environ.get("DEEDPAGE_TOKEN")
        if not token:
            parser.error("set DEEDPAGE_TOKEN to the token from the first publish")
        asyncio.run(delete(args.delete, token))
        return
    html = Path(args.file).read_text(encoding="utf-8") if args.file else SAMPLE_PAGE
    output = asyncio.run(publish(html, args.slug, args.ttl, os.environ.get("DEEDPAGE_TOKEN")))
    try:
        token = json.loads(output).get("token")
    except (TypeError, ValueError, AttributeError):
        token = None
    if token:
        print("Save the token above as DEEDPAGE_TOKEN to update or delete this page.")


if __name__ == "__main__":
    main()
