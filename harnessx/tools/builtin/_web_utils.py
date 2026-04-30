# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15"
)
_MAX_CONTENT_CHARS = 30000


def truncate_text(text: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    """Truncate text to max_chars and append a truncation notice with size info."""
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[... truncated, showing {max_chars}/{len(text)} chars]"
    return text
