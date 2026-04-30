# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import html
import logging
import os
import random
import re
import time
import urllib.parse

import base64

import httpx

from ..base import tool
from ._web_utils import _USER_AGENT

logger = logging.getLogger(__name__)

# Timeouts (seconds) — raised for proxy environments where latency is higher
_SERPAPI_TIMEOUT = 20
_TAVILY_TIMEOUT = 25
_DDG_TIMEOUT = 8

# After this many consecutive failures IN A ROW, skip the full fallback chain
# and return immediately to avoid wasting ~30s per search attempt.
# NOTE: this resets on any successful search, so transient failures don't
# permanently disable search.
_consecutive_failures: int = 0
_MAX_CONSECUTIVE_FAILURES = 10
_last_failure_time: float = 0.0
_CIRCUIT_BREAKER_COOLDOWN = 60.0


async def _search_serpapi(query: str, max_results: int) -> list[dict]:
    """Search via SerpAPI (Google). Returns list of {title, url, snippet}."""
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
    """Search via Tavily API. Returns list of {title, url, snippet}."""
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


async def _search_ddgs_api(query: str, max_results: int) -> list[dict]:
    """DuckDuckGo via duckduckgo-search package (more reliable than scraping)."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return []

    import concurrent.futures

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        raw = await loop.run_in_executor(
            pool,
            lambda: list(DDGS().text(query, max_results=max_results, backend="html")),
        )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": (r.get("body", ""))[:300],
        }
        for r in raw
    ]


_BING_TIMEOUT = 15


def _decode_bing_url(raw_url: str) -> str | None:
    """Decode Bing's ck/a redirect URL back to the real destination."""
    clean = html.unescape(raw_url)
    if "bing.com/ck/a" not in clean:
        return clean
    u_match = re.search(r"[?&]u=a1([^&]+)", clean)
    if not u_match:
        return None
    encoded = u_match.group(1)
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except Exception:
        return urllib.parse.unquote(encoded)


def _parse_bing_results(body: str, max_results: int) -> list[dict]:
    """Extract search results from Bing HTML body."""
    algo_pattern = re.compile(r'<li class="b_algo"[^>]*>(.*?)</li>', re.DOTALL)
    algos = algo_pattern.findall(body)

    results = []
    seen_urls: set[str] = set()
    for algo_html in algos[: max_results + 5]:
        link_m = re.search(
            r'<h2[^>]*><a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            algo_html,
            re.DOTALL,
        )
        if not link_m:
            continue
        raw_url = _decode_bing_url(link_m.group(1))
        if not raw_url or raw_url in seen_urls:
            continue
        seen_urls.add(raw_url)
        title = html.unescape(re.sub(r"<[^>]+>", "", link_m.group(2)).strip())
        snippet = ""
        snip_m = re.search(r"<p[^>]*>(.*?)</p>", algo_html, re.DOTALL)
        if snip_m:
            snippet = html.unescape(re.sub(r"<[^>]+>", "", snip_m.group(1)).strip())[:300]
        results.append({"title": title, "url": raw_url, "snippet": snippet})

    if len(results) < max_results:
        h2_links = re.findall(
            r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            body,
            re.DOTALL,
        )
        for raw_url, title_html in h2_links:
            decoded = _decode_bing_url(raw_url)
            if not decoded or decoded in seen_urls:
                continue
            if any(x in decoded for x in ("bing.com", "microsoft.com", "msn.com")):
                continue
            seen_urls.add(decoded)
            title = html.unescape(re.sub(r"<[^>]+>", "", title_html).strip())
            if len(title) < 5:
                continue
            results.append({"title": title, "url": decoded, "snippet": ""})
            if len(results) >= max_results:
                break

    return results[:max_results]


async def _search_bing_scrape(query: str, max_results: int) -> list[dict]:
    """Bing web search via HTML scraping. No API key needed.

    Uses ensearch=1 + mkt=en-US to force English results (environment IP is
    geo-located to a non-English locale). Handles redirect to cn.bing.com.
    """
    await asyncio.sleep(random.uniform(0.5, 1.5))
    params = urllib.parse.urlencode(
        {
            "q": query,
            "ensearch": "1",
            "setlang": "en-us",
            "mkt": "en-US",
            "count": str(max_results + 5),
        }
    )
    url = f"https://cn.bing.com/search?{params}"
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Cookie": "ENSEARCH=BENVER=1;",
    }
    async with httpx.AsyncClient(timeout=_BING_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code in (403, 429):
            await asyncio.sleep(random.uniform(3.0, 8.0))
            resp = await client.get(url, headers=headers)
        resp.raise_for_status()

    return _parse_bing_results(resp.text, max_results)


async def _search_wikipedia(query: str, max_results: int) -> list[dict]:
    """Wikipedia search — scrape the search results page for article links."""
    search_url = (
        f"https://en.wikipedia.org/w/index.php?search={urllib.parse.quote_plus(query)}&title=Special:Search&ns0=1"
    )
    headers = {"User-Agent": _USER_AGENT, "Accept-Language": "en-US"}
    resp = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(search_url, headers=headers)
                resp.raise_for_status()
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt < 2:
                await asyncio.sleep(random.uniform(2.0, 5.0))
                continue
            raise
    body = resp.text

    if "/wiki/" in resp.url.path and "Special:Search" not in str(resp.url):
        title = ""
        m = re.search(r"<title>(.*?)</title>", body)
        if m:
            title = html.unescape(m.group(1)).replace(" - Wikipedia", "").strip()
        snippet = ""
        for tag in ("p",):
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", body, re.DOTALL)
            if m:
                text = html.unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip())
                if len(text) > 40:
                    snippet = text[:300]
                    break
        return [{"title": title, "url": str(resp.url), "snippet": snippet}]

    results = []
    link_pattern = re.compile(
        r'<div class="mw-search-result-heading">\s*<a[^>]+href="(/wiki/[^"]+)"[^>]*title="([^"]*)"',
        re.DOTALL,
    )
    snippet_pattern = re.compile(r'<div class="searchresult"[^>]*>(.*?)</div>', re.DOTALL)
    links = link_pattern.findall(body)
    snippets = snippet_pattern.findall(body)

    for i, (href, title) in enumerate(links[:max_results]):
        article_url = f"https://en.wikipedia.org{href}"
        snippet = ""
        if i < len(snippets):
            snippet = html.unescape(re.sub(r"<[^>]+>", "", snippets[i]).strip())[:300]
        results.append({"title": html.unescape(title), "url": article_url, "snippet": snippet})

    return results


async def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    """DuckDuckGo HTML scrape fallback. No API key needed."""
    await asyncio.sleep(random.uniform(2.0, 5.0))
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
    async with httpx.AsyncClient(timeout=_DDG_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
        if resp.status_code in (202, 403, 429):
            await asyncio.sleep(random.uniform(5.0, 10.0))
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
    """DuckDuckGo Lite — lighter weight fallback that's less likely to hang."""
    await asyncio.sleep(random.uniform(2.0, 5.0))
    url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote_plus(query)}"
    async with httpx.AsyncClient(timeout=_DDG_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
        if resp.status_code in (202, 403, 429):
            await asyncio.sleep(random.uniform(5.0, 10.0))
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        body = resp.text

    # DuckDuckGo Lite uses simple table layout
    link_pattern = re.compile(r'<a[^>]+href="([^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>', re.DOTALL)
    # Fallback: find any links in result rows
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
        "Uses SerpAPI / Tavily / DuckDuckGo with automatic fallback."
    ),
)
async def web_search_tool(query: str, max_results: int = 5) -> str:
    """Search the web. Returns a numbered list of results with title, URL, and snippet.

    Fallback chain: SerpAPI (Google) → Tavily → Wikipedia + Bing (parallel) → DuckDuckGo HTML → DuckDuckGo Lite.
    Each step has its own timeout to prevent hanging.
    After 5 consecutive failures, short-circuits to avoid wasting time per attempt.
    Periodically retries to recover when network comes back.
    """
    global _consecutive_failures, _last_failure_time

    query = str(query)  # guard against model returning non-string (e.g. Qwen passing int)
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 5
    _unavailable_msg = (
        f"[SEARCH UNAVAILABLE] All search providers failed for query: {query}\n"
        "Web search is temporarily not accessible. You MUST still provide a concrete "
        "answer based on your training knowledge. Do NOT answer with 'unavailable', "
        "'unknown', or 'unable to determine' — give your best factual answer. "
        "Try WebFetch to access specific URLs directly if you know the relevant page."
    )

    # Short-circuit after repeated failures, but auto-reset after cooldown.
    if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
        if time.monotonic() - _last_failure_time > _CIRCUIT_BREAKER_COOLDOWN:
            _consecutive_failures = 0
            logger.info("Web search circuit breaker reset after %.0fs cooldown", _CIRCUIT_BREAKER_COOLDOWN)
        elif _consecutive_failures % 5 != 0:
            _consecutive_failures += 1
            return _unavailable_msg

    # 1. Try SerpAPI (Google) — most reliable in proxy environments
    try:
        results = await _search_serpapi(query, max_results)
        if results:
            _consecutive_failures = 0
            return _format_results(results)
    except Exception as e:
        logger.warning("SerpAPI search failed: %s (falling back to Tavily)", e)

    # 2. Try Tavily
    try:
        results = await _search_tavily(query, max_results)
        if results:
            _consecutive_failures = 0
            return _format_results(results)
    except Exception as e:
        logger.warning("Tavily search failed: %s (falling back)", e)

    # 3. Try Wikipedia + Bing IN PARALLEL — merge results for better coverage
    async def _try_wiki():
        try:
            return await _search_wikipedia(query, max_results)
        except Exception as e:
            logger.warning("Wikipedia search failed: %s", e)
            return []

    async def _try_bing():
        try:
            return await _search_bing_scrape(query, max_results)
        except Exception as e:
            logger.warning("Bing scrape failed: %s", e)
            return []

    wiki_task = asyncio.create_task(_try_wiki())
    bing_task = asyncio.create_task(_try_bing())
    done, pending = await asyncio.wait(
        {wiki_task, bing_task},
        return_when=asyncio.ALL_COMPLETED,
        timeout=max(_BING_TIMEOUT, 20),
    )
    # Merge results from both sources, dedup by URL
    merged: list[dict] = []
    seen_urls: set[str] = set()
    for task in done:
        try:
            for r in task.result():
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    merged.append(r)
        except Exception:
            pass
    if merged:
        _consecutive_failures = 0
        return _format_results(merged[: max_results + 3])

    # 4. Try DDGS API (duckduckgo-search package)
    try:
        results = await _search_ddgs_api(query, max_results)
        if results:
            _consecutive_failures = 0
            return _format_results(results)
    except Exception as e:
        logger.warning("DDGS API failed: %s (falling back to DDG HTML)", e)

    # 5. Try DuckDuckGo HTML scrape
    try:
        results = await _search_duckduckgo(query, max_results)
        if results:
            _consecutive_failures = 0
            return _format_results(results)
    except Exception as e:
        logger.warning("DuckDuckGo HTML failed: %s (falling back to Lite)", e)

    # 6. Try DuckDuckGo Lite
    try:
        results = await _search_duckduckgo_lite(query, max_results)
        if results:
            _consecutive_failures = 0
            return _format_results(results)
    except Exception as e:
        logger.warning("DuckDuckGo Lite also failed: %s", e)

    _consecutive_failures += 1
    _last_failure_time = time.monotonic()
    if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
        logger.warning(
            "Web search circuit breaker tripped after %d failures (cooldown=%.0fs)",
            _consecutive_failures,
            _CIRCUIT_BREAKER_COOLDOWN,
        )
    return _unavailable_msg
