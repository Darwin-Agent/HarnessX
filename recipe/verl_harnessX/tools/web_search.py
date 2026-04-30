"""
WebSearch tool with multi-provider fallback chain.

Fallback order: MCP pool → SerpAPI (Google) → Tavily → DuckDuckGo HTML → DuckDuckGo Lite.
"""

from __future__ import annotations

import html
import logging
import os
import random
import re
import urllib.parse

import json as _json

import httpx

from .base import tool
from ._web_utils import _USER_AGENT

logger = logging.getLogger(__name__)

_MCP_TIMEOUT = int(os.environ.get("VERL_WEBSEARCH_TIMEOUT", "30"))
_SERPAPI_TIMEOUT = int(os.environ.get("VERL_WEBSEARCH_TIMEOUT", "30"))
_TAVILY_TIMEOUT = int(os.environ.get("VERL_WEBSEARCH_TIMEOUT", "30"))
_DDG_TIMEOUT = int(os.environ.get("VERL_WEBSEARCH_DDG_TIMEOUT", "20"))
_RETRY_COUNT = int(os.environ.get("VERL_WEBSEARCH_RETRIES", "2"))
_RETRY_DELAY = float(os.environ.get("VERL_WEBSEARCH_RETRY_DELAY", "1.0"))

_consecutive_failures: int = 0
_MAX_CONSECUTIVE_FAILURES = 5
_success_count: int = 0
_fail_count: int = 0

_MCP_POOL = [
    {
        "url": os.environ.get("MCP_SEARCH_URL", "https://your-mcp-search-endpoint/mcp"),
        "tool": "search",
        "init": False,
    },
]

_INIT_PAYLOAD = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "verl-agent", "version": "0.1"},
    },
}


def _parse_mcp_results(body: str, max_results: int) -> list[dict]:
    data_line = ""
    for line in body.splitlines():
        if line.startswith("data: "):
            data_line = line[6:]
            break
    if not data_line:
        return []
    rpc = _json.loads(data_line)
    if rpc.get("error"):
        raise RuntimeError(rpc["error"].get("message", "MCP error"))
    content_list = rpc.get("result", {}).get("content", [])
    if not content_list:
        return []
    inner = _json.loads(content_list[0].get("text", "{}"))
    results = []
    for bucket in inner.get("x", []):
        for r in bucket.get("organic_results", [])[:max_results]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "snippet": (r.get("snippet", ""))[:300],
                }
            )
    return results[:max_results]


async def _search_mcp(query: str, max_results: int) -> list[dict]:
    pool = list(_MCP_POOL)
    random.shuffle(pool)
    headers = {"Content-Type": "application/json"}
    mcp_key = os.environ.get("MCP_SEARCH_KEY", "")
    if mcp_key:
        headers["Authorization"] = f"Bearer {mcp_key}"
    last_err = None
    for ep in pool:
        call_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": ep["tool"],
                "arguments": {"query": query, "k": int(max_results)},
            },
        }
        try:
            async with httpx.AsyncClient(timeout=_MCP_TIMEOUT) as client:
                if ep["init"]:
                    await client.post(ep["url"], json=_INIT_PAYLOAD, headers=headers)
                resp = await client.post(ep["url"], json=call_payload, headers=headers)
                resp.raise_for_status()
                results = _parse_mcp_results(resp.text, max_results)
                if results:
                    return results
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return []


async def _search_serpapi(query: str, max_results: int) -> list[dict]:
    api_key = os.environ.get("SERPAPI_API_KEY", "")
    if not api_key:
        return []
    params = {
        "api_key": api_key,
        "q": query,
        "engine": "google",
        "num": max_results,
    }
    async with httpx.AsyncClient(timeout=_SERPAPI_TIMEOUT) as client:
        resp = await client.get("https://serpapi.com/search", params=params)
        resp.raise_for_status()
        data = resp.json()
    results = []
    for r in data.get("organic_results", [])[:max_results]:
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": (r.get("snippet", ""))[:300],
            }
        )
    return results


async def _search_tavily(query: str, max_results: int) -> list[dict]:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
    }
    async with httpx.AsyncClient(timeout=_TAVILY_TIMEOUT) as client:
        resp = await client.post("https://api.tavily.com/search", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or r.get("snippet", ""))[:300],
        }
        for r in data.get("results", [])
    ]


async def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
    async with httpx.AsyncClient(timeout=_DDG_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        body = resp.text

    link_pattern = re.compile(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
    snippet_pattern = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)

    links = link_pattern.findall(body)
    snippets = snippet_pattern.findall(body)

    results = []
    for i, (href, title) in enumerate(links[:max_results]):
        uddg = re.search(r"uddg=([^&]+)", href)
        final_url = urllib.parse.unquote(uddg.group(1)) if uddg else href
        clean_title = html.unescape(re.sub(r"<[^>]+>", "", title).strip())
        snippet = ""
        if i < len(snippets):
            snippet = html.unescape(re.sub(r"<[^>]+>", "", snippets[i]).strip())[:300]
        results.append({"title": clean_title, "url": final_url, "snippet": snippet})
    return results


async def _search_duckduckgo_lite(query: str, max_results: int) -> list[dict]:
    url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote_plus(query)}"
    async with httpx.AsyncClient(timeout=_DDG_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        body = resp.text

    link_pattern = re.compile(r'<a[^>]+href="([^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>', re.DOTALL)
    if not link_pattern.findall(body):
        link_pattern = re.compile(r'<a[^>]+rel="nofollow"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)

    snippet_pattern = re.compile(r'<td\s+class="result-snippet"[^>]*>(.*?)</td>', re.DOTALL)

    links = link_pattern.findall(body)
    snippets = snippet_pattern.findall(body)

    results = []
    for i, (href, title) in enumerate(links[:max_results]):
        clean_title = html.unescape(re.sub(r"<[^>]+>", "", title).strip())
        if not clean_title or clean_title.startswith("http"):
            clean_title = href[:60]
        snippet = ""
        if i < len(snippets):
            snippet = html.unescape(re.sub(r"<[^>]+>", "", snippets[i]).strip())[:300]
        results.append({"title": clean_title, "url": href, "snippet": snippet})
    return results


def _format_results(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r['title']}]({r['url']})")
        if r["snippet"]:
            lines.append(f"   {r['snippet']}")
    return "\n".join(lines)


@tool(
    name="WebSearch",
    description=(
        "Search the web and return a list of relevant results with titles, URLs, and snippets. "
        "Uses MCP/SerpAPI/Tavily/DuckDuckGo with automatic fallback."
    ),
)
async def web_search_tool(query: str, max_results: int = 5) -> str:
    """Search the web. Returns a numbered list of results with title, URL, and snippet."""
    global _consecutive_failures, _success_count, _fail_count
    max_results = int(max_results)

    _unavailable_msg = (
        f"[SEARCH UNAVAILABLE] All search providers failed for query: {query}\n"
        "Web search is currently not accessible. Please answer using your training "
        "knowledge instead of continuing to search. If you are not confident in "
        "your answer, state your uncertainty explicitly."
    )

    if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
        if _consecutive_failures % 3 != 0:
            _consecutive_failures += 1
            return _unavailable_msg

    providers = [
        ("mcp", _search_mcp),
        ("serpapi", _search_serpapi),
        ("tavily", _search_tavily),
        ("duckduckgo", _search_duckduckgo),
        ("duckduckgo_lite", _search_duckduckgo_lite),
    ]

    for name, fn in providers:
        try:
            results = await fn(query, max_results)
            if results:
                _consecutive_failures = 0
                _success_count += 1
                logger.warning(
                    "WebSearch OK via %s (%d results) [total: %d ok, %d fail]",
                    name,
                    len(results),
                    _success_count,
                    _fail_count,
                )
                return _format_results(results)
        except Exception as e:
            logger.warning("%s search failed: %s", name, e)

    _consecutive_failures += 1
    _fail_count += 1
    logger.warning(
        "WebSearch ALL FAILED for query: %s [total: %d ok, %d fail]",
        query[:80],
        _success_count,
        _fail_count,
    )
    return _unavailable_msg
