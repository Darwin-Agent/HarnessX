# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import atexit
import os
import tempfile
import time

from ..base import tool
from ._web_utils import truncate_text

# ── Local singleton (LocalSandbox / no sandbox) ───────────────────────────────
_playwright = None
_browser = None
_page = None
_lock = asyncio.Lock()

# ── CDP singleton cache (DockerSandbox) ───────────────────────────────────────
# cdp_url → (playwright_instance, browser, page)
_cdp_connections: dict[str, tuple] = {}
_cdp_lock = asyncio.Lock()

_CDP_CONNECT_RETRIES = 20  # × 0.5 s = up to 10 s for chromium to start
_CDP_RETRY_INTERVAL = 0.5

_LAUNCH_TIMEOUT = 15.0  # seconds to wait for Playwright/Chromium launch


async def _get_local_page():
    """Return the shared local Playwright page, launching on first call."""
    global _playwright, _browser, _page
    async with _lock:
        if _page is None or _browser is None:
            from playwright.async_api import async_playwright

            _playwright = await asyncio.wait_for(
                async_playwright().start(),
                timeout=_LAUNCH_TIMEOUT,
            )
            _browser = await asyncio.wait_for(
                _playwright.chromium.launch(args=["--no-sandbox"]),
                timeout=_LAUNCH_TIMEOUT,
            )
            _page = await _browser.new_page()
    return _page


async def _get_cdp_page(cdp_url: str):
    """Return a Playwright page connected via CDP to a container's Chromium.

    Retries up to 10 s to allow Chromium time to start after container launch.
    The connection is cached per *cdp_url* — warm-pool containers reuse it.
    """
    async with _cdp_lock:
        if cdp_url in _cdp_connections:
            pw, browser, page = _cdp_connections[cdp_url]
            # Verify connection is still alive
            try:
                await page.evaluate("1")
                return page
            except Exception:
                # Connection lost (container restarted) — reconnect below
                try:
                    await browser.close()
                except Exception:
                    pass
                try:
                    await pw.stop()
                except Exception:
                    pass
                del _cdp_connections[cdp_url]

        from playwright.async_api import async_playwright

        pw = await async_playwright().start()
        browser = None
        last_exc: Exception | None = None
        for attempt in range(_CDP_CONNECT_RETRIES):
            try:
                browser = await pw.chromium.connect_over_cdp(cdp_url)
                break
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(_CDP_RETRY_INTERVAL)

        if browser is None:
            await pw.stop()
            raise RuntimeError(
                f"Could not connect to Chromium CDP at {cdp_url} after {_CDP_CONNECT_RETRIES} attempts: {last_exc}"
            )

        # Use existing context/page if chromium already has one, else create
        contexts = browser.contexts
        if contexts and contexts[0].pages:
            page = contexts[0].pages[0]
        else:
            page = await browser.new_page()

        _cdp_connections[cdp_url] = (pw, browser, page)
        return page


async def _get_page():
    """Dispatch to CDP or local Playwright based on the active sandbox."""
    from harnessx.sandbox.base import get_current_sandbox

    sandbox = get_current_sandbox()
    cdp_url: str | None = getattr(sandbox, "cdp_url", None)
    if cdp_url:
        return await _get_cdp_page(cdp_url)
    return await _get_local_page()


async def _close_cdp_connection(cdp_url: str) -> None:
    """Close a specific CDP connection (called by DockerSandboxProvider.release)."""
    async with _cdp_lock:
        entry = _cdp_connections.pop(cdp_url, None)
        if entry is None:
            return
        pw, browser, _ = entry
        try:
            await browser.close()
        except Exception:
            pass
        try:
            await pw.stop()
        except Exception:
            pass


async def _close_local_browser():
    """Close the local Playwright singleton.

    Uses a timeout on lock acquisition to avoid deadlocking if another
    coroutine holds the lock (e.g. Playwright launch is hanging).
    """
    global _playwright, _browser, _page

    acquired = False
    try:
        await asyncio.wait_for(_lock.acquire(), timeout=3.0)
        acquired = True
    except (asyncio.TimeoutError, Exception):
        pass

    try:
        if _browser:
            try:
                await asyncio.wait_for(_browser.close(), timeout=5.0)
            except Exception:
                pass
            _browser = None
            _page = None
        if _playwright:
            try:
                await asyncio.wait_for(_playwright.stop(), timeout=5.0)
            except Exception:
                pass
            _playwright = None
    finally:
        if acquired:
            _lock.release()


def _atexit_close():
    """Synchronous atexit hook — close all browser instances on exit.

    Uses asyncio.run() (fresh event loop) with per-operation timeouts so the
    process never hangs waiting for an unresponsive playwright subprocess.
    Catches BaseException (including KeyboardInterrupt) so a Ctrl+C pressed
    during cleanup does not produce a traceback.
    """

    async def _close_all():
        try:
            await asyncio.wait_for(_close_local_browser(), timeout=3.0)
        except BaseException:
            pass
        for cdp_url in list(_cdp_connections.keys()):
            try:
                await asyncio.wait_for(_close_cdp_connection(cdp_url), timeout=3.0)
            except BaseException:
                pass

    coro = _close_all()
    loop = None
    try:
        # Avoid asyncio.run() here: repeated Ctrl+C during interpreter shutdown
        # can interrupt loop bootstrap/teardown and leak noisy tracebacks.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        except BaseException:
            # Cleanup is best-effort; never block process exit.
            try:
                coro.close()
            except Exception:
                pass
            pass
    except BaseException:
        # If loop creation failed, make sure the coroutine does not leak and
        # trigger "was never awaited" warnings.
        try:
            coro.close()
        except Exception:
            pass
    finally:
        if loop is not None:
            shutdown_coro = None
            try:
                shutdown_coro = loop.shutdown_asyncgens()
                loop.run_until_complete(shutdown_coro)
            except BaseException:
                if shutdown_coro is not None:
                    try:
                        shutdown_coro.close()
                    except Exception:
                        pass
                pass
            try:
                loop.close()
            except BaseException:
                pass


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
    """
    Headless browser via Playwright.

    Automatically uses the Chromium inside a Docker container (via CDP) when
    ``DockerSandboxProvider(enable_browser=True)`` is active; otherwise launches
    a local Chromium process.

    Args:
        action: One of 'navigate' | 'get_text' | 'screenshot' | 'click' | 'type'.
        url: URL for 'navigate'.
        selector: CSS selector for 'click' / 'type'.
        text: Text to type for 'type' action.
        screenshot_path: Optional file path for 'screenshot'. Defaults to a tmp path.
    """
    action = action.lower().strip()

    if action == "navigate":
        if not url:
            return "Error: 'url' is required for navigate action."
        page = await _get_page()
        await page.goto(url, timeout=30000, wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
        return f"Navigated to: {url}"

    elif action == "get_text":
        page = await _get_page()
        body_text = await page.inner_text("body")
        return truncate_text(body_text.strip()) or "(no text content)"

    elif action == "screenshot":
        page = await _get_page()
        if not screenshot_path:
            screenshot_path = os.path.join(
                tempfile.gettempdir(),
                f"harnessx_screenshot_{int(time.time())}.png",
            )
        else:
            os.makedirs(os.path.dirname(os.path.abspath(screenshot_path)), exist_ok=True)
        await page.screenshot(path=screenshot_path, full_page=False)
        return f"Screenshot saved to: {screenshot_path}"

    elif action == "click":
        if not selector:
            return "Error: 'selector' is required for click action."
        page = await _get_page()
        await page.click(selector, timeout=10000)
        return f"Clicked: {selector}"

    elif action == "type":
        if not selector:
            return "Error: 'selector' is required for type action."
        if not text:
            return "Error: 'text' is required for type action."
        page = await _get_page()
        await page.click(selector, timeout=10000)
        await page.type(selector, text)
        return f"Typed into {selector}: {text!r}"

    else:
        return f"Unknown action: {action!r}. Valid actions: navigate, get_text, screenshot, click, type."
