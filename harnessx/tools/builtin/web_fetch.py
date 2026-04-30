# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import httpx

from ..base import tool
from ._web_utils import _USER_AGENT, truncate_text

# Heuristic: static fetch returning fewer than this many chars likely got a
# JS-rendered shell (e.g. <div id="root"></div>). Fall back to Playwright.
_JS_THRESHOLD = 200


async def _fetch_static(url: str) -> str:
    """httpx GET → html2text. Returns error description on failure (not empty)."""
    try:
        import html2text
    except ImportError:
        return "[error: html2text not installed]"

    import asyncio

    last_error = ""
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if "html" not in content_type and "text" not in content_type:
                    return f"[binary content: {content_type}]"
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.body_width = 0
                return h.handle(resp.text)
        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}"
            if e.response.status_code in (429, 500, 502, 503):
                await asyncio.sleep(2)
                continue
            return f"[fetch failed: {last_error} for {url}]"
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = str(type(e).__name__)
            if attempt == 0:
                await asyncio.sleep(1)
                continue
            return f"[fetch failed: {last_error} for {url}]"
        except Exception as e:
            return f"[fetch failed: {e}]"
    return f"[fetch failed after retries: {last_error}]"


async def _fetch_with_browser(url: str) -> str:
    """Reuse the browser singleton (from browser.py) to fetch JS-rendered pages."""
    from .browser import _get_page

    page = await _get_page()
    await page.goto(url, timeout=20000, wait_until="domcontentloaded")
    await page.wait_for_timeout(1500)  # let JS settle
    return await page.inner_text("body")


@tool(
    name="WebFetch",
    description=(
        "Fetch the text content of a web page. "
        "Automatically upgrades to a headless browser for JavaScript-rendered pages."
    ),
)
async def web_fetch_tool(url: str) -> str:
    """
    Fetch web page content as plain text / markdown.
    Upgrades to Playwright if static fetch returns < 200 chars (JS-rendered page).
    """
    try:
        text = await _fetch_static(url)
        if len(text.strip()) < _JS_THRESHOLD:
            text = await _fetch_with_browser(url)
        return truncate_text(text.strip()) or f"No content retrieved from {url}"
    except Exception as e:
        return f"Fetch error for {url}: {e}"
