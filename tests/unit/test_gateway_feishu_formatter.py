from __future__ import annotations

import json

from gateway.channels.feishu.formatter import (
    build_text_payload_candidates,
    build_interactive_content,
    build_post_md_content,
    has_markdown_table,
)
from gateway.channels.feishu.utils import extract_post_text


def test_build_post_md_content_uses_post_md_schema() -> None:
    payload = json.loads(build_post_md_content("hello\n```python\nprint('x')\n```"))
    row = payload["zh_cn"]["content"][0][0]
    assert row["tag"] == "md"
    assert "hello" in row["text"]
    # Code fence should be separated from previous text for Feishu parsing.
    assert "\n```python" in row["text"]


def test_has_markdown_table_detects_pipe_table() -> None:
    text = "| Name | Score |\n|---|---|\n| Alice | 95 |"
    assert has_markdown_table(text) is True
    assert has_markdown_table("No table here") is False


def test_build_interactive_content_converts_table_block() -> None:
    text = "Summary\n\n| Name | Score |\n| --- | --- |\n| Alice | 95 |\n| Bob | 88 |\n"
    card = json.loads(build_interactive_content(text))
    assert "elements" in card
    tags = [e.get("tag") for e in card["elements"]]
    assert "markdown" in tags
    assert "table" in tags


def test_extract_post_text_supports_md_tag() -> None:
    body = {
        "content": [
            [{"tag": "md", "text": "**Hello**"}],
            [{"tag": "text", "text": "World"}],
        ]
    }
    text = extract_post_text(body)
    assert "**Hello**" in text
    assert "World" in text


def test_build_text_payload_candidates_fallback_order_for_table() -> None:
    text = "| A | B |\n|---|---|\n| 1 | 2 |"
    cands = build_text_payload_candidates(
        text,
        prefer_interactive_table=True,
        allow_interactive=True,
    )
    assert [c[0] for c in cands] == ["interactive", "post", "text"]


def test_build_text_payload_candidates_without_interactive() -> None:
    text = "hello"
    cands = build_text_payload_candidates(
        text,
        prefer_interactive_table=True,
        allow_interactive=False,
    )
    assert [c[0] for c in cands] == ["post", "text"]
