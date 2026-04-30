"""
WebFetch tool — fetches web page content as plain text, or downloads files.

Tries static httpx+html2text first; falls back to headless Playwright
if the static response is too short (JS-rendered page).

Non-HTML/text content (PDF, images, archives, etc.) is automatically
downloaded to a local download directory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import urllib.parse

import httpx

from .base import tool
from ._web_utils import _USER_AGENT, truncate_text

logger = logging.getLogger(__name__)

_JS_THRESHOLD = 200
_RETRY_COUNT = 0
_RETRY_DELAY = float(os.getenv("VERL_WEBFETCH_RETRY_DELAY", "1.0"))

_success_count: int = 0
_fail_count: int = 0

_DOWNLOAD_DIR = os.getenv("VERL_DOWNLOAD_DIR", "/tmp/verl_downloads")
_DOWNLOAD_MAX_BYTES = int(os.getenv("VERL_WEBFETCH_DOWNLOAD_MAX_BYTES", str(500 * 1024 * 1024)))


def _clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _filename_from_url(url: str, content_type: str) -> str:
    """Extract a filename from the URL path; fall back to timestamp-based name."""
    parsed = urllib.parse.urlparse(url)
    basename = os.path.basename(urllib.parse.unquote(parsed.path)).strip()
    if basename and "." in basename:
        return basename

    ext_map = {
        "application/pdf": ".pdf",
        "application/zip": ".zip",
        "application/gzip": ".gz",
        "application/x-tar": ".tar",
        "application/json": ".json",
        "application/xml": ".xml",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "video/mp4": ".mp4",
        "text/csv": ".csv",
    }
    ct = content_type.split(";")[0].strip().lower()
    ext = ext_map.get(ct, "")
    name = basename if basename else f"download_{int(time.time())}"
    return name + ext


async def _download_file(url: str, save_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    timeout = int(os.getenv("VERL_WEBFETCH_DOWNLOAD_TIMEOUT", "240"))
    total_bytes = 0
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url, headers={"User-Agent": _USER_AGENT}) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "unknown")
                content_length = resp.headers.get("content-length")
                if content_length and int(content_length) > _DOWNLOAD_MAX_BYTES:
                    return (
                        f"Error: file too large ({_human_size(int(content_length))}). "
                        f"Max allowed: {_human_size(_DOWNLOAD_MAX_BYTES)}."
                    )
                with open(save_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        total_bytes += len(chunk)
                        if total_bytes > _DOWNLOAD_MAX_BYTES:
                            f.close()
                            os.remove(save_path)
                            return (
                                f"Error: download exceeded max size ({_human_size(_DOWNLOAD_MAX_BYTES)}) "
                                f"after {_human_size(total_bytes)}. File removed."
                            )
                        f.write(chunk)
        return (
            f"Fetched file '{os.path.basename(save_path)}' "
            f"({_human_size(total_bytes)}, {content_type}) "
            f"saved to {save_path}"
        )
    except httpx.HTTPStatusError as e:
        return f"Download failed: HTTP {e.response.status_code} for {url}"
    except httpx.TimeoutException:
        return f"Download failed: timed out after {timeout}s for {url}"
    except Exception as e:
        return f"Download failed: {e}"


async def _fetch_static(url: str) -> tuple[str, bool]:
    """Fetch URL. Returns (text, was_downloaded).

    If content is non-HTML/text, auto-downloads to _DOWNLOAD_DIR and returns
    a summary message with was_downloaded=True.
    """
    try:
        import html2text

        async with httpx.AsyncClient(
            timeout=int(os.getenv("VERL_WEBFETCH_STATIC_TIMEOUT", "30")),
            follow_redirects=True,
        ) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                filename = _filename_from_url(url, content_type)
                save_path = os.path.join(_DOWNLOAD_DIR, filename)
                result = await _download_file(url, save_path)
                return result, True
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.body_width = 0
            return h.handle(resp.text), False
    except Exception:
        return "", False


async def _fetch_with_browser(url: str) -> str:
    import time as _time
    from .browser import acquire_page

    _lock_t0 = _time.monotonic()
    async with acquire_page() as page:
        _lock_wait = _time.monotonic() - _lock_t0
        if _lock_wait > 1.0:
            logger.warning("[WEBFETCH_DEBUG] pool_wait=%.1fs url=%s", _lock_wait, url[:80])
        await page.goto(
            url,
            timeout=int(os.getenv("VERL_BROWSER_PAGE_TIMEOUT", "15000")),
            wait_until="domcontentloaded",
        )
        await page.wait_for_timeout(500)
        return await page.inner_text("body")


@tool(
    name="WebFetch",
    description=(
        "Fetch the text content of a web page. "
        "Automatically upgrades to a headless browser for JavaScript-rendered pages. "
        "Non-text content (PDF, images, archives, etc.) is automatically downloaded "
        "to a local directory and the saved file path is returned."
    ),
)
async def web_fetch_tool(url: str) -> str:
    """Fetch web page text; auto-downloads binary files to local disk."""
    global _success_count, _fail_count

    parsed_path = urllib.parse.urlparse(url).path.lower()
    if parsed_path.endswith(
        (
            ".pdf",
            ".zip",
            ".gz",
            ".tar",
            ".xlsx",
            ".xls",
            ".docx",
            ".pptx",
            ".csv",
            ".mp3",
            ".mp4",
            ".wav",
        )
    ):
        for attempt in range(1 + _RETRY_COUNT):
            result = await _download_file(
                url,
                os.path.join(_DOWNLOAD_DIR, _filename_from_url(url, "application/octet-stream")),
            )
            if "saved to" in result:
                _success_count += 1
                logger.warning(
                    "WebFetch DOWNLOAD OK (direct, attempt %d) [total: %d ok, %d fail]",
                    attempt + 1,
                    _success_count,
                    _fail_count,
                )
                return result
            if attempt < _RETRY_COUNT:
                logger.warning(
                    "WebFetch download retry %d/%d for %s",
                    attempt + 1,
                    _RETRY_COUNT,
                    url[:80],
                )
                await asyncio.sleep(_RETRY_DELAY)
        _fail_count += 1
        logger.warning(
            "WebFetch DOWNLOAD FAILED (direct) %s [total: %d ok, %d fail]",
            url[:80],
            _success_count,
            _fail_count,
        )
        return result

    for attempt in range(1 + _RETRY_COUNT):
        try:
            text, was_downloaded = await _fetch_static(url)

            if was_downloaded:
                if "saved to" in text:
                    _success_count += 1
                    logger.warning(
                        "WebFetch DOWNLOAD OK (attempt %d) [total: %d ok, %d fail]",
                        attempt + 1,
                        _success_count,
                        _fail_count,
                    )
                    return text
                if attempt < _RETRY_COUNT:
                    logger.warning(
                        "WebFetch download retry %d/%d for %s",
                        attempt + 1,
                        _RETRY_COUNT,
                        url[:80],
                    )
                    await asyncio.sleep(_RETRY_DELAY)
                    continue
                _fail_count += 1
                logger.warning(
                    "WebFetch DOWNLOAD FAILED %s [total: %d ok, %d fail]",
                    url[:80],
                    _success_count,
                    _fail_count,
                )
                return text

            if len(text.strip()) < _JS_THRESHOLD:
                try:
                    text = await asyncio.wait_for(
                        _fetch_with_browser(url),
                        timeout=int(os.getenv("VERL_WEBFETCH_BROWSER_TIMEOUT", "36")),
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning("WebFetch browser fallback failed for %s: %s", url[:80], e)

            text = _clean_text(text)
            result = truncate_text(text) or ""
            if result:
                _success_count += 1
                logger.warning(
                    "WebFetch OK (%d chars, attempt %d) [total: %d ok, %d fail]",
                    len(result),
                    attempt + 1,
                    _success_count,
                    _fail_count,
                )
                return result

            if attempt < _RETRY_COUNT:
                logger.warning(
                    "WebFetch empty, retry %d/%d for %s",
                    attempt + 1,
                    _RETRY_COUNT,
                    url[:80],
                )
                await asyncio.sleep(_RETRY_DELAY)
                continue

            _fail_count += 1
            logger.warning(
                "WebFetch EMPTY for %s [total: %d ok, %d fail]",
                url[:80],
                _success_count,
                _fail_count,
            )
            return f"No content retrieved from {url}"
        except Exception as e:
            if attempt < _RETRY_COUNT:
                logger.warning(
                    "WebFetch error retry %d/%d for %s: %s",
                    attempt + 1,
                    _RETRY_COUNT,
                    url[:80],
                    e,
                )
                await asyncio.sleep(_RETRY_DELAY)
                continue
            _fail_count += 1
            logger.warning(
                "WebFetch FAILED for %s: %s [total: %d ok, %d fail]",
                url[:80],
                e,
                _success_count,
                _fail_count,
            )
            return f"Fetch error for {url}: {e}"

    _fail_count += 1
    return f"No content retrieved from {url}"
