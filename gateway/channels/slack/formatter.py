from __future__ import annotations

import re

_MAX_LEN = 39000


def to_mrkdwn(text: str) -> str:
    """
    Convert basic markdown to Slack mrkdwn format.
    Differences from standard markdown:
      - Bold: **text** → *text*
      - Italic: *text* or _text_ (same)
      - Strikethrough: ~~text~~ → ~text~
      - Link: [text](url) → <url|text>
    """
    # Code blocks: leave as-is (Slack supports ``` natively)
    # Links
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    # Bold: **text** → *text*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    # Strikethrough: ~~text~~ → ~text~
    text = re.sub(r"~~(.+?)~~", r"~\1~", text)
    return text[:_MAX_LEN]
