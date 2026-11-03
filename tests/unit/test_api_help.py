from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from harnessx.api.routes import help as help_routes


@pytest.fixture
def docs_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    docs_root = tmp_path / "docs"
    guide = docs_root / "guide"
    guide.mkdir(parents=True)
    (guide / "evolution.md").write_text("# Evolution\n\nEnglish content.\n", encoding="utf-8")
    (guide / "evolution.zh-CN.md").write_text("# 进化\n\n中文内容。\n", encoding="utf-8")
    (guide / "quickstart.md").write_text("# Quickstart\n", encoding="utf-8")
    (docs_root / "pr18-remediation.md").write_text("# Internal record\n", encoding="utf-8")
    gateway = docs_root / "gateway" / "zh"
    gateway.mkdir(parents=True)
    (gateway / "README.md").write_text("# 网关\n", encoding="utf-8")
    for recipe in ("coevolution", "verl"):
        recipe_root = tmp_path / "recipe" / recipe
        recipe_root.mkdir(parents=True)
        (recipe_root / "README.md").write_text(f"# {recipe}\n", encoding="utf-8")
        (recipe_root / "README.zh-CN.md").write_text(f"# {recipe} 中文\n", encoding="utf-8")
    monkeypatch.setattr(help_routes, "DOCS_ROOT", docs_root)
    app = FastAPI()
    app.include_router(help_routes.router)
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("lang,section,title", [("en", "Guide", "Evolution"), ("zh", "指南", "进化")])
def test_tree_selects_language_without_duplicate_pages(docs_client, lang, section, title):
    response = docs_client.get("/help", params={"lang": lang})
    assert response.status_code == 200
    sections = response.json()["sections"]
    guide = next(item for item in sections if item["name"] == section)
    assert guide["items"][0] == {"path": "guide/evolution", "title": title}
    assert len(guide["items"]) == 2
    paths = [entry["path"] for item in sections for entry in item["items"]]
    assert len(paths) == len(set(paths))
    assert not any(".zh-CN" in path or "pr18" in path or "gateway/" in path for path in paths)


@pytest.mark.parametrize("lang,content", [("en", "English content."), ("zh", "中文内容。")])
def test_content_follows_selected_language(docs_client, lang, content):
    response = docs_client.get("/help/guide/evolution", params={"lang": lang})
    assert response.status_code == 200
    assert response.json()["path"] == "guide/evolution"
    assert content in response.json()["content"]


def test_untranslated_page_remains_available(docs_client):
    response = docs_client.get("/help/guide/quickstart", params={"lang": "zh"})
    assert response.status_code == 200
    assert response.json()["title"] == "Quickstart"


@pytest.mark.parametrize("recipe", ["coevolution", "verl"])
def test_recipe_pages_are_available_in_both_languages(docs_client, recipe):
    for lang, suffix in [("en", ""), ("zh", " 中文")]:
        response = docs_client.get(f"/help/recipes/{recipe}", params={"lang": lang})
        assert response.status_code == 200
        assert response.json()["title"] == f"{recipe}{suffix}"
        tree = docs_client.get("/help", params={"lang": lang}).json()
        assert any(entry["path"] == f"recipes/{recipe}" for section in tree["sections"] for entry in section["items"])


def test_explicit_translation_path_has_canonical_response(docs_client):
    response = docs_client.get("/help/guide/evolution.zh-CN.md", params={"lang": "zh"})
    assert response.status_code == 200
    assert response.json()["path"] == "guide/evolution"
    assert response.json()["title"] == "进化"


@pytest.mark.parametrize("route", ["/help", "/help/guide/evolution"])
def test_unknown_language_is_rejected(docs_client, route):
    assert docs_client.get(route, params={"lang": "unknown"}).status_code == 422


def test_missing_page_and_path_traversal(docs_client):
    assert docs_client.get("/help/guide/missing").status_code == 404
    assert docs_client.get("/help/%2e%2e/private").status_code == 400
