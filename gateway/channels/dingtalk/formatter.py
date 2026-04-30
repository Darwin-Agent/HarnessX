from __future__ import annotations


_MAX_LEN = 20000


def markdown_payload(text: str, title: str = "Reply") -> dict:
    """Build dingtalk webhook markdown payload."""
    return {
        "msgtype": "markdown",
        "markdown": {
            "title": title,
            "text": text[:_MAX_LEN],
        },
        "at": {"isAtAll": False},
    }


def text_payload(text: str) -> dict:
    """Build dingtalk webhook text payload."""
    return {
        "msgtype": "text",
        "text": {"content": text[:_MAX_LEN]},
        "at": {"isAtAll": False},
    }
