_USER_AGENT = "Mozilla/5.0 (compatible; HarnessX/0.4)"
_MAX_CONTENT_CHARS = 20000


def truncate_text(text: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[... truncated, showing {max_chars}/{len(text)} chars]"
    return text
