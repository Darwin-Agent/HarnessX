# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from harnessx.core.events import Message
from harnessx.processors.memory.strategies.extractors import (
    ContentExtractor,
    TextContentExtractor,
    extract_blocks_by_type,
    has_modality,
    message_modalities,
)


# ─── TextContentExtractor ────────────────────────────────────────────────────


class TestExtractors:
    def test_text_extractor_str_content(self):
        ext = TextContentExtractor()
        msg = Message(role="user", content="hello world")
        assert ext.extract(msg) == "hello world"

    def test_text_extractor_list_content(self):
        ext = TextContentExtractor()
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "hello"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc",
                    },
                },
                {"type": "text", "text": "world"},
            ],
        )
        assert ext.extract(msg) == "hello world"

    def test_text_extractor_empty_content(self):
        ext = TextContentExtractor()
        msg = Message(role="user", content="")
        assert ext.extract(msg) == ""

    def test_text_extractor_image_only(self):
        ext = TextContentExtractor()
        msg = Message(
            role="user",
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc",
                    },
                },
            ],
        )
        assert ext.extract(msg) == ""

    def test_text_extractor_modality(self):
        assert TextContentExtractor().modality() == "text"

    def test_text_extractor_is_content_extractor(self):
        assert isinstance(TextContentExtractor(), ContentExtractor)

    # ─── extract_blocks_by_type ──────────────────────────────────────────────────

    def test_extract_blocks_str_content(self):
        assert extract_blocks_by_type("hello", "text") == []

    def test_extract_blocks_image(self):
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image", "source": {"type": "base64", "data": "x"}},
            {
                "type": "image",
                "source": {"type": "url", "url": "http://example.com/img.png"},
            },
        ]
        images = extract_blocks_by_type(content, "image")
        assert len(images) == 2
        assert all(b["type"] == "image" for b in images)

    def test_extract_blocks_no_match(self):
        content = [{"type": "text", "text": "hi"}]
        assert extract_blocks_by_type(content, "audio") == []

    def test_extract_blocks_skips_non_dict(self):
        content = [{"type": "text", "text": "hi"}, "not a dict", 42]
        assert len(extract_blocks_by_type(content, "text")) == 1

    # ─── has_modality ─────────────────────────────────────────────────────────────

    def test_has_modality_str_text(self):
        msg = Message(role="user", content="hello")
        assert has_modality(msg, "text") is True
        assert has_modality(msg, "image") is False

    def test_has_modality_list_with_image(self):
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "hi"},
                {"type": "image", "source": {"type": "base64", "data": "x"}},
            ],
        )
        assert has_modality(msg, "text") is True
        assert has_modality(msg, "image") is True
        assert has_modality(msg, "audio") is False

    def test_has_modality_empty_list(self):
        msg = Message(role="user", content=[])
        assert has_modality(msg, "text") is False

    # ─── message_modalities ──────────────────────────────────────────────────────

    def test_message_modalities_str(self):
        msg = Message(role="user", content="hello")
        assert message_modalities(msg) == {"text"}

    def test_message_modalities_mixed(self):
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "hi"},
                {"type": "image", "source": {}},
                {"type": "audio", "data": "..."},
            ],
        )
        assert message_modalities(msg) == {"text", "image", "audio"}

    def test_message_modalities_skips_invalid(self):
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "hi"},
                {"no_type": True},  # missing "type" key
                "not a dict",
            ],
        )
        assert message_modalities(msg) == {"text"}
