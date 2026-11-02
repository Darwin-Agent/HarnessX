# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import sys
from types import SimpleNamespace

import pytest

from examples.deed_page import publish as example
from harnessx.tools.base import ToolResult


def _fake_registry(calls, output='{"ok": true}', error_on=None):
    class Registry:
        def __init__(self, **kwargs):
            assert kwargs == {
                "transport": "http",
                "url": "https://deed.page/mcp",
                "name": "deed-page",
                "headers": {"User-Agent": "HarnessX/0.1.0 (deed.page example)"},
            }

        async def connect(self):
            calls.append("connect")
            if error_on == "connect":
                raise RuntimeError("unavailable")

        async def execute(self, name, arguments):
            calls.append((name, arguments))
            if error_on == name:
                return SimpleNamespace(error="unavailable", output="")
            return ToolResult(output=output)

        async def disconnect(self):
            calls.append("disconnect")

    return Registry


@pytest.mark.parametrize(
    ("slug", "ttl", "expected"),
    [
        (None, 3600, {"files": {"index.html": "<h1>x</h1>"}, "ttl_seconds": 3600}),
        ("my-report", 0, {"files": {"index.html": "<h1>x</h1>"}, "slug": "my-report"}),
    ],
)
async def test_publish_arguments_and_close(monkeypatch, capsys, slug, ttl, expected):
    calls = []
    monkeypatch.setattr(example, "MCPToolRegistry", _fake_registry(calls, '{"url": "https://a.deed.page/"}'))
    output = await example.publish("<h1>x</h1>", slug, ttl)
    assert calls == ["connect", ("publish_site", expected), "disconnect"]
    assert output == '{"url": "https://a.deed.page/"}'
    assert "https://a.deed.page/" in capsys.readouterr().out


async def test_delete_passes_token(monkeypatch):
    calls = []
    monkeypatch.setattr(example, "MCPToolRegistry", _fake_registry(calls))
    await example.delete("my-report", "deed_tok_test")
    assert calls == ["connect", ("delete_site", {"slug": "my-report", "token": "deed_tok_test"}), "disconnect"]


@pytest.mark.parametrize("failure", ["connect", "publish_site"])
async def test_errors_are_raised_and_registry_closed(monkeypatch, failure):
    calls = []
    monkeypatch.setattr(example, "MCPToolRegistry", _fake_registry(calls, error_on=failure))
    with pytest.raises(RuntimeError, match="unavailable"):
        await example.publish("<h1>x</h1>")
    assert calls[-1] == "disconnect"


def test_cli_publishes_sample_and_shows_token_hint(monkeypatch, capsys):
    calls = []
    monkeypatch.delenv("DEEDPAGE_TOKEN", raising=False)
    monkeypatch.setattr(example, "MCPToolRegistry", _fake_registry(calls, '{"token": "deed_tok_test"}'))
    monkeypatch.setattr(sys, "argv", ["publish.py"])
    example.main()
    assert calls[1] == ("publish_site", {"files": {"index.html": example.SAMPLE_PAGE}, "ttl_seconds": 3600})
    assert "DEEDPAGE_TOKEN" in capsys.readouterr().out


def test_cli_publishes_file(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("DEEDPAGE_TOKEN", raising=False)
    page = tmp_path / "report.html"
    page.write_text("<h1>report</h1>", encoding="utf-8")
    monkeypatch.setattr(example, "MCPToolRegistry", _fake_registry(calls))
    monkeypatch.setattr(sys, "argv", ["publish.py", str(page), "--slug", "my-report", "--ttl", "0"])
    example.main()
    assert calls[1] == ("publish_site", {"files": {"index.html": "<h1>report</h1>"}, "slug": "my-report"})


def test_cli_delete_requires_token(monkeypatch):
    monkeypatch.delenv("DEEDPAGE_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["publish.py", "--delete", "my-report"])
    with pytest.raises(SystemExit):
        example.main()


def test_cli_delete_with_token(monkeypatch):
    calls = []
    monkeypatch.setenv("DEEDPAGE_TOKEN", "deed_tok_test")
    monkeypatch.setattr(example, "MCPToolRegistry", _fake_registry(calls))
    monkeypatch.setattr(sys, "argv", ["publish.py", "--delete", "my-report"])
    example.main()
    assert calls[1] == ("delete_site", {"slug": "my-report", "token": "deed_tok_test"})


def test_cli_update_forwards_token(monkeypatch, tmp_path):
    calls = []
    page = tmp_path / "report.html"
    page.write_text("<h1>v2</h1>", encoding="utf-8")
    monkeypatch.setenv("DEEDPAGE_TOKEN", "deed_tok_test")
    monkeypatch.setattr(example, "MCPToolRegistry", _fake_registry(calls))
    monkeypatch.setattr(sys, "argv", ["publish.py", str(page), "--slug", "my-report", "--ttl", "0"])
    example.main()
    assert calls[1] == (
        "publish_site",
        {"files": {"index.html": "<h1>v2</h1>"}, "slug": "my-report", "token": "deed_tok_test"},
    )
