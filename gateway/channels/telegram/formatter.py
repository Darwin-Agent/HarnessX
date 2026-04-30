from __future__ import annotations

import re

from .utils import escape_md2, _prefix_within_utf16_limit

_MAX_LEN = 4096


def _escape_code_content(text: str) -> str:
    """Inside MarkdownV2 code spans only backslash and backtick need escaping."""
    return text.replace("\\", "\\\\").replace("`", "\\`")


def to_markdown_v2(text: str) -> str:
    """Convert basic markdown to Telegram MarkdownV2.

    Escapes all special chars in plain text; inside code spans only escapes
    backslash and backtick as required by the Telegram Bot API spec.
    """
    if not text:
        return text
    parts = re.split(r"(```[\s\S]*?```|`[^`]+`)", text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # code span or code block
            if part.startswith("```"):
                inner = part[3:-3]
                result.append("```" + _escape_code_content(inner) + "```")
            else:
                inner = part[1:-1]
                result.append("`" + _escape_code_content(inner) + "`")
        else:
            result.append(escape_md2(part))
    out = "".join(result)
    return _prefix_within_utf16_limit(out, _MAX_LEN)
