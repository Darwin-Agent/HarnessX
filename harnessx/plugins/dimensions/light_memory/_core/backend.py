# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import unicodedata
from pathlib import Path

from .types import GrepHit, MemoryDocument, MemoryFrontmatter, PluginConfig

# ── Tag used to wrap injected memory context ────────────────────────────────

RELEVANT_MEMORIES_TAG = "relevant-memories"

# ── Directory constants ──────────────────────────────────────────────────────

MEMORY_ROOT_DIRS = [
    "user",
    "knowledge",
    "knowledge/skill",
    "knowledge/learning",
    "entity",
    "sessions",
    "daily",
]

MEMORY_SCAN_DIRS = [
    "user",
    "knowledge",
    "entity",
    "sessions",
    "daily",
]

ORGANIZATION_SCAN_DIRS = [
    "user",
    "knowledge",
    "entity",
    "sessions",
]

# ── Repository initialisation ───────────────────────────────────────────────


def ensure_memory_repo(cfg: PluginConfig) -> None:
    os.makedirs(cfg.memory_root, exist_ok=True)
    for d in MEMORY_ROOT_DIRS:
        os.makedirs(os.path.join(cfg.memory_root, d), exist_ok=True)
    os.makedirs(os.path.join(cfg.memory_root, ".ops"), exist_ok=True)


# ── Frontmatter parsing / serialisation ──────────────────────────────────────

_KEY_VALUE_RE = re.compile(r"^([a-zA-Z_][\w-]*):\s*(.*)$")


def split_frontmatter(raw: str) -> tuple[str | None, str]:
    text = raw.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return None, text
    return text[4:end], text[end + 5 :]


def _strip_optional_quotes(value: str) -> str:
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def _parse_frontmatter_value(raw: str):
    if raw == "":
        return ""
    if raw == "true":
        return True
    if raw == "false":
        return False
    if re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
        return float(raw) if "." in raw else int(raw)
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_strip_optional_quotes(item.strip()) for item in inner.split(",") if item.strip()]
    return _strip_optional_quotes(raw)


def parse_frontmatter(block: str) -> dict:
    result: dict = {}
    current_key: str | None = None
    current_array: list[str] | None = None

    for line in block.split("\n"):
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue

        if trimmed.startswith("- ") and current_key is not None and current_array is not None:
            current_array.append(_strip_optional_quotes(trimmed[2:].strip()))
            continue

        m = _KEY_VALUE_RE.match(trimmed)
        if not m:
            continue

        if current_key is not None and current_array is not None:
            result[current_key] = current_array if current_array else ""
            current_key = None
            current_array = None

        key = m.group(1)
        raw_value = m.group(2).strip()

        if raw_value == "":
            current_key = key
            current_array = []
            continue

        result[key] = _parse_frontmatter_value(raw_value)

    if current_key is not None and current_array is not None:
        result[current_key] = current_array if current_array else ""

    return result


_VALID_TYPES = {"style", "profile", "session", "skill", "learning", "entity", "daily"}
_VALID_STATUSES = {"active", "deprecated", "archived"}
_VALID_CONFIDENCES = {"high", "medium", "low", "deprecated"}


def _clamp(value, lo: float, hi: float, fallback: float) -> float:
    if not isinstance(value, (int, float)):
        return fallback
    return min(hi, max(lo, float(value)))


def _to_string_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def to_memory_frontmatter(parsed: dict) -> MemoryFrontmatter | None:
    if (
        parsed.get("type") not in _VALID_TYPES
        or parsed.get("status") not in _VALID_STATUSES
        or not isinstance(parsed.get("id"), str)
        or not isinstance(parsed.get("title"), str)
        or not isinstance(parsed.get("summary"), str)
        or not isinstance(parsed.get("created_at"), str)
        or not isinstance(parsed.get("updated_at"), str)
    ):
        return None

    last_accessed = parsed.get("last_accessed_at")
    if not isinstance(last_accessed, str):
        last_accessed = parsed["updated_at"]

    return MemoryFrontmatter(
        id=parsed["id"],
        type=parsed["type"],
        status=parsed["status"],
        confidence=parsed.get("confidence") if parsed.get("confidence") in _VALID_CONFIDENCES else "medium",
        importance=_clamp(parsed.get("importance"), 0, 1, 0.5),
        access_count=int(parsed.get("access_count", 0)) if isinstance(parsed.get("access_count"), (int, float)) else 0,
        skip_count=int(parsed.get("skip_count", 0)) if isinstance(parsed.get("skip_count"), (int, float)) else 0,
        token_estimate=int(parsed.get("token_estimate", 0))
        if isinstance(parsed.get("token_estimate"), (int, float))
        else 0,
        created_at=parsed["created_at"],
        updated_at=parsed["updated_at"],
        last_accessed_at=last_accessed,
        title=parsed["title"],
        summary=parsed["summary"],
        keywords=_to_string_list(parsed.get("keywords")),
        related=_to_string_list(parsed.get("related")),
        source=parsed.get("source", "") if isinstance(parsed.get("source"), str) else "",
        supersedes=_to_string_list(parsed.get("supersedes")),
        user_id=parsed.get("user_id") if isinstance(parsed.get("user_id"), str) else None,
    )


def _serialize_value(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(json.dumps(v) for v in value) + "]"
    return json.dumps(value)


def serialize_frontmatter(fm: MemoryFrontmatter) -> str:
    lines = [
        "---",
        f"id: {_serialize_value(fm.id)}",
        f"type: {fm.type}",
        f"status: {fm.status}",
        f"confidence: {fm.confidence}",
        f"importance: {fm.importance}",
        f"access_count: {fm.access_count}",
        f"skip_count: {fm.skip_count}",
        f"token_estimate: {fm.token_estimate}",
        f"created_at: {_serialize_value(fm.created_at)}",
        f"updated_at: {_serialize_value(fm.updated_at)}",
        f"last_accessed_at: {_serialize_value(fm.last_accessed_at)}",
        f"title: {_serialize_value(fm.title)}",
        f"summary: {_serialize_value(fm.summary)}",
        f"keywords: {_serialize_value(fm.keywords)}",
        f"related: {_serialize_value(fm.related)}",
        f"source: {_serialize_value(fm.source)}",
        f"supersedes: {_serialize_value(fm.supersedes)}",
    ]
    if fm.user_id:
        lines.append(f"user_id: {_serialize_value(fm.user_id)}")
    lines.append("---")
    return "\n".join(lines)


# ── Memory document I/O ─────────────────────────────────────────────────────


def extract_summary_section(body: str) -> str:
    marker = "<!-- SUMMARY_END -->"
    idx = body.find(marker)
    if idx == -1:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        return truncate_text(paragraphs[0], 200) if paragraphs else ""
    return body[:idx].strip()


def parse_memory_document(file_path: str, root: str) -> MemoryDocument | None:
    try:
        raw = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return None
    fm_block, body = split_frontmatter(raw)
    if fm_block is None:
        return None
    parsed = parse_frontmatter(fm_block)
    fm = to_memory_frontmatter(parsed)
    if fm is None:
        return None
    return MemoryDocument(
        file_path=file_path,
        relative_path=_to_repo_relative(root, file_path),
        frontmatter=fm,
        body=body.strip(),
        summary_section=extract_summary_section(body),
    )


def write_memory_document(file_path: str, fm: MemoryFrontmatter, body: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp = file_path + ".tmp"
    payload = f"{serialize_frontmatter(fm)}\n\n{body.strip()}\n"
    Path(tmp).write_text(payload, encoding="utf-8")
    os.replace(tmp, file_path)


# ── Keyword extraction ───────────────────────────────────────────────────────

STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "has",
        "have",
        "been",
        "some",
        "them",
        "than",
        "its",
        "over",
        "such",
        "that",
        "this",
        "with",
        "will",
        "each",
        "from",
        "they",
        "were",
        "which",
        "their",
        "said",
        "what",
        "when",
        "where",
        "how",
        "who",
        "did",
        "does",
        "just",
        "more",
        "also",
        "about",
        "would",
        "make",
        "like",
        "been",
        "could",
        "into",
        "time",
        "very",
        "your",
        "most",
        "should",
        "other",
        "there",
        "after",
        "then",
        "only",
        "those",
        "these",
    }
)

# CJK Unified Ideographs range
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
_LATIN_RE = re.compile(r"[a-z0-9_-]+")


def extract_keywords(query: str) -> list[str]:
    normalised = unicodedata.normalize("NFKC", query).lower()
    normalised = re.sub(r"\s+", " ", normalised).strip()
    tokens: set[str] = set()

    # Latin words (>=3 chars, no stopwords)
    for m in _LATIN_RE.finditer(normalised):
        tok = m.group()
        if len(tok) > 2 and tok not in STOPWORDS:
            tokens.add(tok)

    # CJK bigrams
    for m in _CJK_RE.finditer(normalised):
        seg = m.group()
        if len(seg) == 1:
            tokens.add(seg)
        else:
            for i in range(len(seg) - 1):
                tokens.add(seg[i : i + 2])

    return list(tokens)


# ── TF-IDF keyword search ───────────────────────────────────────────────────


def grep_memory_files_with_scores(
    root: str,
    keywords: list[str],
    max_results: int = 50,
) -> list[GrepHit]:
    if not keywords:
        return []

    # Build per-keyword regex
    escaped = [re.escape(k) for k in keywords]
    keyword_patterns = [re.compile(e, re.IGNORECASE) for e in escaped]
    combined = re.compile("|".join(escaped), re.IGNORECASE)

    # First pass: collect matching files
    all_files: list[tuple[str, str]] = []  # (relative_path, content)

    for scan_dir in MEMORY_SCAN_DIRS:
        dir_path = os.path.join(root, scan_dir)
        if not os.path.isdir(dir_path):
            continue
        for fp in _walk_markdown_files(dir_path):
            rel = _to_repo_relative(root, fp)
            if rel in ("_index.md", "README.md"):
                continue
            try:
                content = Path(fp).read_text(encoding="utf-8")
            except OSError:
                continue
            if combined.search(content):
                all_files.append((rel, content))

    if not all_files:
        return []

    # Document frequency per keyword
    df = [0] * len(keywords)
    for _, content in all_files:
        for i, pat in enumerate(keyword_patterns):
            if pat.search(content):
                df[i] += 1

    # IDF weights
    total_docs = len(all_files)
    idf = [math.log(1 + total_docs / d) if d > 0 else 0.0 for d in df]

    # TF-IDF scores
    hits: list[GrepHit] = []
    for rel, content in all_files:
        score = 0.0
        for i, pat in enumerate(keyword_patterns):
            matches = pat.findall(content)
            if matches:
                tf = math.log(1 + len(matches))
                score += tf * idf[i]
        if score > 0:
            hits.append(GrepHit(relative_path=rel, match_count=score))

    hits.sort(key=lambda h: h.match_count, reverse=True)
    return hits[:max_results]


# ── Entity keyword index ────────────────────────────────────────────────────

_KW_FIELD_RE = re.compile(r"^keywords:\s*\[([^\]]*)\]", re.MULTILINE)


def get_entity_keyword_hits(
    root: str,
    entity_names: list[str],
    max_results: int = 30,
) -> list[GrepHit]:
    """Scan frontmatter ``keywords`` fields for entity-name matches.

    This is a fast-path lookup that matches capitalized proper nouns
    (person names, place names) against the curated keyword lists in
    memory files, without running a full-text search.
    """
    if not entity_names:
        return []

    lower_names = [n.lower() for n in entity_names]
    hits: list[GrepHit] = []

    for scan_dir in MEMORY_SCAN_DIRS:
        dir_path = os.path.join(root, scan_dir)
        if not os.path.isdir(dir_path):
            continue
        for fp in _walk_markdown_files(dir_path):
            rel = _to_repo_relative(root, fp)
            if rel in ("_index.md", "README.md"):
                continue
            try:
                content = Path(fp).read_text(encoding="utf-8")
            except OSError:
                continue

            m = _KW_FIELD_RE.search(content)
            if not m:
                continue

            file_keywords = m.group(1).lower()
            score = sum(1 for name in lower_names if name in file_keywords)
            if score > 0:
                hits.append(GrepHit(relative_path=rel, match_count=score))

    hits.sort(key=lambda h: h.match_count, reverse=True)
    return hits[:max_results]


# ── File traversal ───────────────────────────────────────────────────────────


def _walk_markdown_files(dir_path: str) -> list[str]:
    results: list[str] = []
    try:
        entries = os.scandir(dir_path)
    except OSError:
        return results
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir(follow_symlinks=False):
            results.extend(_walk_markdown_files(entry.path))
        elif entry.is_file() and entry.name.endswith(".md"):
            results.append(entry.path)
    return results


def get_all_memory_documents(
    cfg: PluginConfig,
    dirs: list[str] | None = None,
) -> list[MemoryDocument]:
    documents: list[MemoryDocument] = []
    for d in dirs or MEMORY_SCAN_DIRS:
        start = os.path.join(cfg.memory_root, d)
        if not os.path.isdir(start):
            continue
        for fp in _walk_markdown_files(start):
            doc = parse_memory_document(fp, cfg.memory_root)
            if doc:
                documents.append(doc)
    return documents


# ── Path / text utilities ────────────────────────────────────────────────────


def _to_repo_relative(root: str, file_path: str) -> str:
    return os.path.relpath(file_path, root).replace(os.sep, "/")


def truncate_text(value: str, max_length: int) -> str:
    text = value.strip()
    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 3)].strip() + "..."


def round_val(value: float, digits: int) -> float:
    factor = 10**digits
    return round(value * factor) / factor


# ── Injected-memory stripping ────────────────────────────────────────────────

_TAG_RE = re.compile(
    r"<relevant-memories>.*?</relevant-memories>",
    re.DOTALL | re.IGNORECASE,
)


def strip_injected_memories(text: str) -> str:
    """Remove ``<relevant-memories>…</relevant-memories>`` blocks from text."""
    return _TAG_RE.sub("", text).strip()


# ── Session-topics keyword supplement ───────────────────────────────────────

_SESSION_TOPICS_FILE = ".ops/session_topics.txt"


def merge_session_topics(cfg: "PluginConfig", prompt_keywords: list[str]) -> list[str]:
    """Read ``.ops/session_topics.txt`` and merge its keywords into *prompt_keywords*.

    The file is maintained by the agent (one keyword per line).  It improves
    recall continuity for follow-up turns such as "how is that project going?".
    Returns a deduplicated merged list.
    """
    file_path = os.path.join(cfg.memory_root, _SESSION_TOPICS_FILE)
    try:
        raw = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return prompt_keywords
    topics = [line.strip() for line in raw.splitlines() if line.strip()]
    if not topics:
        return prompt_keywords
    merged: dict[str, None] = dict.fromkeys(prompt_keywords)
    for t in topics:
        merged[t] = None
    return list(merged)


# ── Git helpers ──────────────────────────────────────────────────────────────


def ensure_git_repo(cfg: "PluginConfig") -> None:
    """Initialise a git repo inside *memory_root* if one does not exist yet.

    Silently skips when ``cfg.git_mode == "disabled"`` or when git is not
    available on the system.
    """
    if cfg.git_mode == "disabled":
        return
    git_dir = os.path.join(cfg.memory_root, ".git")
    if os.path.isdir(git_dir):
        return
    try:
        subprocess.run(
            ["git", "init", cfg.memory_root],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", cfg.memory_root, "config", "user.email", "memory@harnessx"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", cfg.memory_root, "config", "user.name", "HarnessX Memory"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def commit_files_if_needed(
    cfg: "PluginConfig",
    relative_paths: list[str],
    message: str,
) -> None:
    """Stage *relative_paths* and commit if there are any changes.

    Silently no-ops when ``cfg.auto_commit`` is False, when git is not
    available, or when there are no staged changes.
    """
    if not cfg.auto_commit or cfg.git_mode == "disabled":
        return
    if not relative_paths:
        return
    git_dir = os.path.join(cfg.memory_root, ".git")
    if not os.path.isdir(git_dir):
        return
    try:
        for rel in relative_paths:
            full = os.path.join(cfg.memory_root, rel)
            if os.path.exists(full):
                subprocess.run(
                    ["git", "-C", cfg.memory_root, "add", full],
                    check=True,
                    capture_output=True,
                )
        result = subprocess.run(
            ["git", "-C", cfg.memory_root, "diff", "--cached", "--quiet"],
            capture_output=True,
        )
        if result.returncode == 0:
            return  # nothing staged
        subprocess.run(
            ["git", "-C", cfg.memory_root, "commit", "-m", message],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


# ── Conversation extraction from messages ───────────────────────────────────


def extract_conversation_turns(
    messages: tuple | list,
    strip_memory_tag: bool = True,
) -> list[dict]:
    """Extract ``{role, text}`` dicts from a ``final_messages`` sequence.

    Strips the injected ``<relevant-memories>`` context block from assistant
    messages so the daily log contains only the actual conversation.
    """
    turns: list[dict] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role not in ("user", "assistant"):
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # ContentBlock list — extract text parts
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    parts.append(str(block.text))
            text = "".join(parts)
        else:
            text = str(content)

        if strip_memory_tag:
            text = strip_injected_memories(text)
        text = text.strip()
        if len(text) < 10:
            continue
        turns.append({"role": role, "text": text})
    return turns
