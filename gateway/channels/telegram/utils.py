from __future__ import annotations

import re

_MD2_SPECIAL = r"\_*[]()~`>#+-=|{}.!"


def escape_md2(text: str) -> str:
    """Escape MarkdownV2 special characters."""
    return re.sub(r"([_*\[\]()~`>#+=|{}.!\-\\])", r"\\\1", text)


def utf16_len(s: str) -> int:
    """Count UTF-16 code units in s.

    Telegram's 4096 limit is measured in UTF-16 code units, not codepoints.
    Characters outside the BMP (emoji, etc.) consume 2 units each.
    """
    return len(s.encode("utf-16-le")) // 2


def _prefix_within_utf16_limit(s: str, limit: int) -> str:
    """Return the longest prefix of s whose UTF-16 length ≤ limit."""
    if utf16_len(s) <= limit:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if utf16_len(s[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return s[:lo]


def split_text(text: str, max_len: int = 4000) -> list[str]:
    """Split text into chunks whose UTF-16 length ≤ max_len, respecting code block boundaries."""
    if utf16_len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text

    while utf16_len(remaining) > max_len:
        # Find a safe split point within the UTF-16 budget
        safe = _prefix_within_utf16_limit(remaining, max_len)
        split_at = len(safe)

        candidate = remaining[:split_at]
        if candidate.count("```") % 2 == 1:
            # Mid code-block — extend to next closing ```
            close = remaining.find("```", split_at)
            split_at = close + 3 if close != -1 else split_at
        else:
            # Prefer paragraph break, then line break, then hard limit
            para = remaining.rfind("\n\n", 0, split_at)
            if para > split_at // 2:
                split_at = para
            else:
                line = remaining.rfind("\n", 0, split_at)
                if line > split_at // 2:
                    split_at = line

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks
