"""
Browser tool — headless Chromium via Playwright.

Actions: navigate, get_text, screenshot, click, type, query.
Uses a per-worker page pool (configurable via VERL_BROWSER_POOL_SIZE)
to allow concurrent browser operations across sequences.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import sys
import tempfile
import time

import logging

from .base import tool
from ._web_utils import truncate_text

logger = logging.getLogger(__name__)

_success_count: int = 0
_fail_count: int = 0

_POOL_SIZE = int(os.getenv("VERL_BROWSER_POOL_SIZE", "3"))

_playwright = None
_browser = None
_init_lock = asyncio.Lock()
_page_pool: asyncio.Queue | None = None
_pages: list = []


async def _ensure_pool():
    global _playwright, _browser, _page_pool, _pages
    async with _init_lock:
        if _page_pool is not None:
            return
        from playwright.async_api import async_playwright

        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(args=["--no-sandbox"])
        _page_pool = asyncio.Queue()
        for _ in range(_POOL_SIZE):
            page = await _browser.new_page()
            _pages.append(page)
            _page_pool.put_nowait(page)
        logger.warning("[BROWSER] Page pool initialized: %d pages", _POOL_SIZE)


@contextlib.asynccontextmanager
async def acquire_page():
    """Acquire a page from the pool; release it back when done."""
    await _ensure_pool()
    t0 = time.monotonic()
    page = await _page_pool.get()
    wait = time.monotonic() - t0
    if wait > 1.0:
        logger.warning("[BROWSER_DEBUG] pool_wait=%.1fs", wait)
    try:
        yield page
    finally:
        _page_pool.put_nowait(page)


async def _close_browser():
    global _playwright, _browser, _page_pool, _pages
    async with _init_lock:
        if _browser:
            await _browser.close()
            _browser = None
            _pages.clear()
            _page_pool = None
        if _playwright:
            await _playwright.stop()
            _playwright = None


def _atexit_close():
    global _playwright, _browser, _page_pool, _pages
    try:
        if _browser:
            try:
                _browser._impl_obj._connection._transport._proc.kill()
            except Exception:
                pass
        if _playwright:
            try:
                _playwright._impl_obj._connection._transport._proc.kill()
            except Exception:
                pass
    except Exception:
        pass
    _browser = None
    _pages = []
    _page_pool = None
    _playwright = None


_original_unraisablehook = sys.unraisablehook


def _suppress_loop_closed(unraisable):
    if isinstance(unraisable.exc_value, RuntimeError) and "Event loop is closed" in str(unraisable.exc_value):
        return
    _original_unraisablehook(unraisable)


sys.unraisablehook = _suppress_loop_closed
atexit.register(_atexit_close)


@tool(
    name="Browser",
    description=(
        "Control a headless web browser. "
        "Actions: navigate (go to URL), get_text (extract page text), "
        "screenshot (capture page image), click (click an element), "
        "type (type text into an element)."
    ),
)
async def browser_tool(
    action: str,
    url: str = "",
    selector: str = "",
    text: str = "",
    screenshot_path: str = "",
) -> str:
    """Headless browser via Playwright."""
    global _success_count, _fail_count
    action = action.lower().strip()

    try:
        async with acquire_page() as page:
            if action == "navigate":
                if not url:
                    return "Error: 'url' is required for navigate action."
                await page.goto(
                    url,
                    timeout=int(os.getenv("VERL_BROWSER_PAGE_TIMEOUT", "30000")),
                    wait_until="domcontentloaded",
                )
                await page.wait_for_timeout(500)
                _success_count += 1
                logger.warning(
                    "Browser OK action=navigate [total: %d ok, %d fail]",
                    _success_count,
                    _fail_count,
                )
                return f"Navigated to: {url}"

            elif action == "get_text":
                body_text = await page.inner_text("body")
                _success_count += 1
                logger.warning(
                    "Browser OK action=get_text [total: %d ok, %d fail]",
                    _success_count,
                    _fail_count,
                )
                return truncate_text(body_text.strip()) or "(no text content)"

            elif action == "screenshot":
                if not screenshot_path:
                    screenshot_path = os.path.join(
                        tempfile.gettempdir(),
                        f"harnessx_screenshot_{int(time.time())}.png",
                    )
                else:
                    os.makedirs(os.path.dirname(os.path.abspath(screenshot_path)), exist_ok=True)
                await page.screenshot(path=screenshot_path, full_page=False)
                _success_count += 1
                logger.warning(
                    "Browser OK action=screenshot [total: %d ok, %d fail]",
                    _success_count,
                    _fail_count,
                )
                return f"Screenshot saved to: {screenshot_path}"

            elif action == "click":
                if not selector:
                    return "Error: 'selector' is required for click action."
                await page.click(
                    selector,
                    timeout=int(os.getenv("VERL_BROWSER_CLICK_TIMEOUT", "16000")),
                )
                _success_count += 1
                logger.warning(
                    "Browser OK action=click [total: %d ok, %d fail]",
                    _success_count,
                    _fail_count,
                )
                return f"Clicked: {selector}"

            elif action == "type":
                if not selector:
                    return "Error: 'selector' is required for type action."
                if not text:
                    return "Error: 'text' is required for type action."
                await page.click(
                    selector,
                    timeout=int(os.getenv("VERL_BROWSER_CLICK_TIMEOUT", "16000")),
                )
                await page.type(selector, text)
                _success_count += 1
                logger.warning(
                    "Browser OK action=type [total: %d ok, %d fail]",
                    _success_count,
                    _fail_count,
                )
                return f"Typed into {selector}: {text!r}"

            else:
                return f"Unknown action: {action!r}. Valid actions: navigate, get_text, screenshot, click, type."
    except Exception as e:
        _fail_count += 1
        logger.warning(
            "Browser FAILED action=%s: %s [total: %d ok, %d fail]",
            action,
            e,
            _success_count,
            _fail_count,
        )
        return f"Browser error ({action}): {e}"
