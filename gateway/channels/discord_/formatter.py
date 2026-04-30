from __future__ import annotations

_MAX_LEN = 2000
_EDIT_SAFE_LEN = 1900


def truncate(text: str) -> str:
    if len(text) <= _EDIT_SAFE_LEN:
        return text
    return text[: _EDIT_SAFE_LEN - 1] + "…"


def split_text(text: str, max_len: int = _EDIT_SAFE_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        candidate = remaining[:max_len]
        if candidate.count("```") % 2 == 1:
            # We're mid-code-block — extend to the next closing ```
            close = remaining.find("```", max_len)
            split_at = close + 3 if close != -1 else max_len
        else:
            split_at = remaining.rfind("\n\n", 0, max_len)
            if split_at < max_len // 2:
                split_at = remaining.rfind("\n", 0, max_len)
            if split_at < max_len // 2:
                split_at = max_len
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks
