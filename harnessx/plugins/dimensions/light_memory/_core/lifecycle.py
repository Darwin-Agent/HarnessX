# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from .backend import (
    ORGANIZATION_SCAN_DIRS,
    extract_keywords,
    get_all_memory_documents,
    get_entity_keyword_hits,
    grep_memory_files_with_scores,
    parse_memory_document,
    round_val,
    truncate_text,
    write_memory_document,
)
from .types import GrepHit
from .daily import append_to_daily
from .decay import compute_decayed_importance, find_decayed_memories
from .index_file import generate_index_file
from .organize import run_organization
from .types import MemoryDocument, MemoryFrontmatter, PluginConfig

LLMCallFn = Callable[[str], Awaitable[str | None]]

# ── Synonym expansion ────────────────────────────────────────────────────────

SYNONYM_GROUPS: list[list[str]] = [
    ["sculpture", "sculpt", "sculpting", "art", "artwork", "artistic"],
    ["painting", "paint", "drawing", "draw", "canvas", "art"],
    ["music", "musical", "song", "singing", "instrument", "band", "concert"],
    ["travel", "trip", "vacation", "journey", "visit", "visiting", "toured"],
    ["cook", "cooking", "bake", "baking", "recipe", "kitchen", "chef", "meal"],
    ["exercise", "workout", "gym", "fitness", "running", "yoga", "sport"],
    ["read", "reading", "book", "novel", "literature", "author"],
    ["movie", "film", "cinema", "watch", "watching"],
    ["job", "work", "career", "profession", "occupation", "employment", "company"],
    ["school", "university", "college", "education", "study", "studying", "degree"],
    ["pet", "dog", "cat", "animal", "puppy", "kitten"],
    ["child", "children", "kid", "kids", "son", "daughter", "baby"],
    ["marry", "married", "marriage", "wedding", "spouse", "husband", "wife"],
    ["friend", "friendship", "buddy", "pal"],
    ["hobby", "hobbies", "interest", "interests", "passion", "pastime"],
    ["plan", "planning", "plans", "goal", "goals", "intend", "intention"],
    ["move", "moving", "relocate", "relocated", "relocation"],
    ["adopt", "adoption", "adopting", "foster"],
    ["volunteer", "volunteering", "charity", "nonprofit"],
    ["garden", "gardening", "plant", "plants", "botanical"],
    ["photograph", "photography", "photo", "camera"],
    ["recommend", "recommendation", "suggest", "suggestion"],
]


def expand_with_synonyms(keywords: list[str]) -> list[str]:
    expanded = set(keywords)
    lower_kws = [k.lower() for k in keywords]
    for kw in lower_kws:
        for group in SYNONYM_GROUPS:
            if kw in group:
                added = 0
                for syn in group:
                    if syn not in expanded and syn != kw and added < 3:
                        expanded.add(syn)
                        added += 1
                break
    return list(expanded)


def extract_relevant_lines(text: str, keywords: list[str], context_lines: int) -> str:
    lines = text.split("\n")
    lower_kws = [k.lower() for k in keywords]
    matched: set[int] = set()
    for i, line in enumerate(lines):
        ll = line.lower()
        if any(kw in ll for kw in lower_kws):
            matched.add(i)
    if not matched:
        return "\n".join(lines[:20]) + ("\n..." if len(lines) > 20 else "")
    included: set[int] = set()
    for idx in matched:
        for c in range(max(0, idx - context_lines), min(len(lines), idx + context_lines + 1)):
            included.add(c)
    sorted_idx = sorted(included)
    result: list[str] = []
    prev = -2
    for idx in sorted_idx:
        if idx > prev + 1 and result:
            result.append("...")
        result.append(lines[idx])
        prev = idx
    return "\n".join(result)


# ── Internal: parse daily log as MemoryDocument ──────────────────────────────


def _parse_daily_log_as_document(file_path: str, root: str) -> MemoryDocument | None:
    try:
        content = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        return None
    if not content.strip():
        return None
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})\.md$", file_path)
    date_str = date_m.group(1) if date_m else "unknown"
    rel = os.path.relpath(file_path, root).replace(os.sep, "/")
    return MemoryDocument(
        file_path=file_path,
        relative_path=rel,
        frontmatter=MemoryFrontmatter(
            id=f"daily/{date_str}",
            type="daily",
            status="active",
            confidence="high",
            importance=0.5,
            access_count=0,
            skip_count=0,
            token_estimate=max(1, len(content) // 4),
            created_at=f"{date_str}T12:00:00Z",
            updated_at=f"{date_str}T12:00:00Z",
            last_accessed_at=f"{date_str}T12:00:00Z",
            title=f"Daily conversation log: {date_str}",
            summary=f"Raw conversation transcript from {date_str}",
        ),
        body=content,
        summary_section=f"Raw conversation transcript from {date_str}",
    )


# ── Internal: shared search + rank ──────────────────────────────────────────


def _search_and_rank(
    cfg: PluginConfig,
    prompt: str,
    now: datetime | None = None,
):
    """Keyword grep → parse docs → rank by match count + decayed importance."""
    if now is None:
        now = datetime.now(timezone.utc)

    prompt_keywords = extract_keywords(prompt)
    synonyms = expand_with_synonyms(prompt_keywords)
    grep_hits = grep_memory_files_with_scores(cfg.memory_root, synonyms, cfg.top_k * 5)

    # Stage 1.5: Entity keyword index — augment grep with keyword-field entity matches
    entity_names = re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)
    unique_entities = list(dict.fromkeys(entity_names))
    if unique_entities:
        grep_paths = {h.relative_path for h in grep_hits}
        for hit in get_entity_keyword_hits(cfg.memory_root, unique_entities, 20):
            if hit.relative_path not in grep_paths:
                grep_hits.append(GrepHit(relative_path=hit.relative_path, match_count=0.5))
                grep_paths.add(hit.relative_path)

    candidates: list[dict] = []
    for hit in grep_hits:
        full_path = os.path.join(cfg.memory_root, hit.relative_path)
        is_daily = hit.relative_path.startswith("daily/")
        doc = (
            _parse_daily_log_as_document(full_path, cfg.memory_root)
            if is_daily
            else parse_memory_document(full_path, cfg.memory_root)
        )
        if doc and doc.frontmatter.status == "active":
            decay = compute_decayed_importance(
                doc.frontmatter.importance,
                doc.frontmatter.last_accessed_at,
                cfg.access_half_life_days,
                now,
            )
            candidates.append({"doc": doc, "match_count": hit.match_count, "decay": decay})

    candidates.sort(key=lambda c: (-c["match_count"], -c["decay"]))

    # Load profile/style
    profile_doc = None
    profile_path = os.path.join(cfg.memory_root, "user", "profile.md")
    if os.path.isfile(profile_path):
        d = parse_memory_document(profile_path, cfg.memory_root)
        if d and d.frontmatter.status == "active":
            profile_doc = d

    style_doc = None
    style_path = os.path.join(cfg.memory_root, "user", "style.md")
    if os.path.isfile(style_path):
        style_doc = parse_memory_document(style_path, cfg.memory_root)

    return candidates, profile_doc, style_doc, prompt_keywords


def _update_access_counters(cfg: PluginConfig, docs: list[MemoryDocument], now: datetime) -> None:
    now_iso = now.isoformat()
    for doc in docs:
        fm = MemoryFrontmatter(
            **{
                **doc.frontmatter.__dict__,
                "access_count": doc.frontmatter.access_count + 1,
                "last_accessed_at": now_iso,
            }
        )
        write_memory_document(doc.file_path, fm, doc.body)


# ── Flow 1: Memory Recall ───────────────────────────────────────────────────


def read_recalled_memories(
    cfg: PluginConfig,
    prompt: str,
    now: datetime | None = None,
) -> str:
    """Single-stage recall: search → rank → return full text."""
    if now is None:
        now = datetime.now(timezone.utc)
    candidates, profile_doc, style_doc, _ = _search_and_rank(cfg, prompt, now)
    top = candidates[: cfg.top_k]

    if top:
        _update_access_counters(cfg, [c["doc"] for c in top], now)

    parts: list[str] = []
    if profile_doc:
        parts.append(f"### Speaker Profiles\n{profile_doc.body}\n")
    if style_doc:
        parts.append(f"### User Preferences\n{style_doc.body}\n")
    for c in top:
        fm = c["doc"].frontmatter
        parts.append(f"### {fm.title} (importance: {round_val(c['decay'], 2)})")
        parts.append(c["doc"].body)
        parts.append("")
    return "\n".join(parts)


_RECALL_SELECTION_PROMPT = """\
You are a memory retrieval assistant. Given a question and a list of candidate memories (title and summary), select which memories are likely to contain the answer.

Question: {question}

Candidate memories:
{candidateTable}

Return a JSON array of the candidate numbers that are relevant to answering this question.
When in doubt, include the memory — it is better to include a marginally relevant memory than to miss a critical one.
Return ONLY a JSON array of numbers, e.g. [1, 3, 5]. No other text."""


async def read_recalled_memories_with_llm(
    cfg: PluginConfig,
    prompt: str,
    call_llm: LLMCallFn,
    now: datetime | None = None,
) -> str:
    """Two-stage LLM-assisted recall:
    1. Keyword grep → wide candidates (with LLM query expansion + synonym expansion)
    2. LLM selects relevant candidates → load full text
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Stage 0: LLM query expansion
    expansion_prompt = (
        f"Given this question about someone's life, generate 5-8 search keyword groups to find relevant memories. Include synonyms, related terms, and any person names mentioned.\n\n"
        f"Question: {prompt}\n\n"
        f"Return a JSON array of strings, each being a keyword or short phrase.\n"
        f'Example: ["Caroline job", "career", "work", "employment", "company", "Feeling-AI"]\n\n'
        f"Return ONLY the JSON array, no other text."
    )
    llm_expanded: list[str] = []
    try:
        resp = await call_llm(expansion_prompt)
        if resp:
            cleaned = resp.strip()
            cleaned = re.sub(r"```json?\s*\n?", "", cleaned)
            cleaned = cleaned.replace("```", "").strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                llm_expanded = [k for k in parsed if isinstance(k, str)]
    except Exception:
        pass

    # Stage 1: Wide grep
    prompt_keywords = extract_keywords(prompt)
    synonym_expanded = expand_with_synonyms(prompt_keywords)
    llm_tokens = []
    for k in llm_expanded:
        llm_tokens.extend(extract_keywords(k))
    all_kws = list(dict.fromkeys(synonym_expanded + llm_tokens))

    grep_hits = grep_memory_files_with_scores(cfg.memory_root, all_kws, 40)

    # Stage 1.5: Entity keyword index — augment grep with keyword-field entity matches
    entity_names = re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)
    unique_entities = list(dict.fromkeys(entity_names))
    if unique_entities:
        grep_paths = {h.relative_path for h in grep_hits}
        for hit in get_entity_keyword_hits(cfg.memory_root, unique_entities, 20):
            if hit.relative_path not in grep_paths:
                grep_hits.append(GrepHit(relative_path=hit.relative_path, match_count=0.5))
                grep_paths.add(hit.relative_path)

    all_candidates: list[dict] = []
    for hit in grep_hits:
        full_path = os.path.join(cfg.memory_root, hit.relative_path)
        is_daily = hit.relative_path.startswith("daily/")
        doc = (
            _parse_daily_log_as_document(full_path, cfg.memory_root)
            if is_daily
            else parse_memory_document(full_path, cfg.memory_root)
        )
        if doc and doc.frontmatter.status == "active":
            decay = compute_decayed_importance(
                doc.frontmatter.importance,
                doc.frontmatter.last_accessed_at,
                cfg.access_half_life_days,
                now,
            )
            all_candidates.append({"doc": doc, "match_count": hit.match_count, "decay": decay})

    all_candidates.sort(key=lambda c: (-c["match_count"], -c["decay"]))

    # Profile / style
    profile_doc = None
    profile_path = os.path.join(cfg.memory_root, "user", "profile.md")
    if os.path.isfile(profile_path):
        d = parse_memory_document(profile_path, cfg.memory_root)
        if d and d.frontmatter.status == "active":
            profile_doc = d

    style_doc = None
    style_path = os.path.join(cfg.memory_root, "user", "style.md")
    if os.path.isfile(style_path):
        style_doc = parse_memory_document(style_path, cfg.memory_root)

    if not all_candidates:
        parts: list[str] = []
        if profile_doc:
            parts.append(f"### Speaker Profiles\n{profile_doc.body}\n")
        if style_doc:
            parts.append(f"### User Preferences\n{style_doc.body}\n")
        return "\n".join(parts) or "(No relevant memories found)"

    # Stage 2: LLM candidate selection
    candidate_lines = [
        f"{i + 1}. [{c['doc'].frontmatter.title}] — {c['doc'].frontmatter.summary}"
        for i, c in enumerate(all_candidates)
    ]
    sel_prompt = _RECALL_SELECTION_PROMPT.replace("{question}", prompt).replace(
        "{candidateTable}", "\n".join(candidate_lines)
    )
    selected_indices: list[int] = []
    try:
        resp = await call_llm(sel_prompt)
        if resp:
            cleaned = resp.strip()
            cleaned = re.sub(r"```json?\s*\n?", "", cleaned)
            cleaned = cleaned.replace("```", "").strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                selected_indices = [
                    int(n) - 1 for n in parsed if isinstance(n, (int, float)) and 1 <= n <= len(all_candidates)
                ]
    except Exception:
        selected_indices = list(range(min(cfg.top_k, len(all_candidates))))

    if not selected_indices:
        selected_indices = list(range(min(cfg.top_k, len(all_candidates))))

    # Stage 2.5: Retrieval reflection (if too few selected)
    if len(selected_indices) <= 2 and len(selected_indices) < len(all_candidates):
        try:
            titles = ", ".join(all_candidates[i]["doc"].frontmatter.title for i in selected_indices)
            reflect_resp = await call_llm(
                f'I found these memories: [{titles}]. But the question "{prompt}" may need more information. '
                f"Generate 3-5 additional search keywords to find missing memories. Return ONLY a JSON array of strings."
            )
            if reflect_resp:
                cleaned = re.sub(r"```json?\s*\n?", "", reflect_resp.strip()).replace("```", "").strip()
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    supp_kws = []
                    for k in parsed:
                        if isinstance(k, str):
                            supp_kws.extend(extract_keywords(k))
                    if supp_kws:
                        existing_paths = {c["doc"].relative_path for c in all_candidates}
                        supp_hits = grep_memory_files_with_scores(cfg.memory_root, supp_kws, 20)
                        for hit in supp_hits:
                            if hit.relative_path in existing_paths:
                                continue
                            fp = os.path.join(cfg.memory_root, hit.relative_path)
                            is_daily = hit.relative_path.startswith("daily/")
                            doc = (
                                _parse_daily_log_as_document(fp, cfg.memory_root)
                                if is_daily
                                else parse_memory_document(fp, cfg.memory_root)
                            )
                            if doc and doc.frontmatter.status == "active":
                                decay = compute_decayed_importance(
                                    doc.frontmatter.importance,
                                    doc.frontmatter.last_accessed_at,
                                    cfg.access_half_life_days,
                                    now,
                                )
                                all_candidates.append(
                                    {
                                        "doc": doc,
                                        "match_count": hit.match_count,
                                        "decay": decay,
                                    }
                                )
                                selected_indices.append(len(all_candidates) - 1)
        except Exception:
            pass

    # Stage 3: Load full text of selected memories
    selected_docs = [all_candidates[i] for i in selected_indices if i < len(all_candidates)]
    if selected_docs:
        _update_access_counters(cfg, [c["doc"] for c in selected_docs], now)

    parts = []
    if profile_doc:
        parts.append(f"### Speaker Profiles\n{profile_doc.body}\n")
    if style_doc:
        parts.append(f"### User Preferences\n{style_doc.body}\n")

    for c in selected_docs:
        fm = c["doc"].frontmatter
        is_daily = fm.type == "daily"
        parts.append(f"### {fm.title} (relevance: {c['match_count']:.1f}, importance: {round_val(c['decay'], 2)})")
        if is_daily:
            parts.append(extract_relevant_lines(c["doc"].body, prompt_keywords, 3))
        else:
            parts.append(c["doc"].body)
        parts.append("")

    return "\n".join(parts)


# ── Flow 2: Daily Capture ────────────────────────────────────────────────────


def perform_capture(
    cfg: PluginConfig,
    speaker: str,
    text: str,
    date: datetime | None = None,
) -> None:
    entry = f"**{speaker}:** {truncate_text(text, 2000)}"
    append_to_daily(cfg, entry, date)


# ── Flow 3: Organization ────────────────────────────────────────────────────


async def perform_organization(
    cfg: PluginConfig,
    call_llm: LLMCallFn | None,
    now: datetime | None = None,
) -> None:
    if now is None:
        now = datetime.now(timezone.utc)

    all_docs = get_all_memory_documents(cfg, ORGANIZATION_SCAN_DIRS)
    if not all_docs:
        return

    # Step 1: Archive below decay threshold
    if cfg.decay_enabled:
        to_archive = find_decayed_memories(all_docs, cfg.access_half_life_days, 0.05, now)
        for doc in to_archive:
            fm = MemoryFrontmatter(
                **{
                    **doc.frontmatter.__dict__,
                    "status": "archived",
                    "updated_at": now.isoformat(),
                }
            )
            write_memory_document(doc.file_path, fm, doc.body)

    # Step 2: LLM consolidation
    if call_llm:
        fresh = get_all_memory_documents(cfg, ORGANIZATION_SCAN_DIRS)
        await run_organization(cfg, call_llm, fresh, now)

    # Step 3: Rebuild index
    final = get_all_memory_documents(cfg)
    generate_index_file(cfg, final)


# ── Memory Writing: rule-based ───────────────────────────────────────────────


def _compute_turn_importance(text: str) -> float:
    n = len(text)
    if n < 50:
        return 0.3
    if n < 100:
        return 0.4
    if n < 200:
        return 0.5
    if n < 500:
        return 0.6
    return 0.7


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def write_memory_from_turn(
    cfg: PluginConfig,
    speaker: str,
    text: str,
    session_id: int,
    turn_index: int,
    session_date: str,
) -> None:
    if len(text) < 30:
        return
    slug = f"session-{session_id}-turn-{turn_index}"
    file_path = os.path.join(cfg.memory_root, "sessions", cfg.user_id, f"{slug}.md")
    if os.path.isfile(file_path):
        return

    kws = extract_keywords(text)
    summary = truncate_text(text, 100)
    date_iso = f"{session_date}T12:00:00Z"

    fm = MemoryFrontmatter(
        id=f"sessions/{cfg.user_id}/{slug}",
        type="session",
        status="active",
        confidence="high",
        importance=_compute_turn_importance(text),
        access_count=0,
        skip_count=0,
        token_estimate=_estimate_tokens(text),
        created_at=date_iso,
        updated_at=date_iso,
        last_accessed_at=date_iso,
        title=f"{speaker}: {truncate_text(text, 50)}",
        summary=summary,
        keywords=kws[:10],
        source=f"daily/{session_date}.md",
        user_id=cfg.user_id,
    )
    body = (
        f"## Summary\n\n{summary}\n\n<!-- SUMMARY_END -->\n\n"
        f"**Speaker:** {speaker}\n**Session:** {session_id}\n**Date:** {session_date}\n\n"
        f"{text}\n\n## Change History\n- {session_date}: Initial creation from conversation"
    )
    write_memory_document(file_path, fm, body)


def _update_session_topics(cfg: PluginConfig, turns: list[dict]) -> None:
    all_text = " ".join(t["text"] for t in turns)
    kws = extract_keywords(all_text)
    sorted_kws = sorted(kws, key=len, reverse=True)[:10]
    topics_path = os.path.join(cfg.memory_root, ".ops", "session_topics.txt")
    os.makedirs(os.path.dirname(topics_path), exist_ok=True)
    Path(topics_path).write_text("\n".join(sorted_kws) + "\n", encoding="utf-8")


def write_memories_from_session(
    cfg: PluginConfig,
    turns: list[dict],
    session_id: int,
    session_date: str,
) -> int:
    written = 0
    for i, turn in enumerate(turns):
        if len(turn["text"]) < 20:
            continue
        write_memory_from_turn(cfg, turn["speaker"], turn["text"], session_id, i, session_date)
        written += 1
    _update_session_topics(cfg, turns)
    return written


# ── Memory Writing: LLM-based ───────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a memory management assistant. Review the following conversation between two people and extract facts worth remembering long-term.

Conversation from Session {sessionId} ({sessionDate}):
{conversationText}

Extract the most important facts, preferences, events, and relationships mentioned. For each memory, provide:
- title: A concise descriptive title (max 60 chars), including the speaker's name
- summary: One-line summary clearly stating WHO the fact is about
- keywords: 3-8 relevant keywords for future retrieval, including speaker names and key entities
- importance: 0.3 to 0.9 (higher = more important to remember)
- content: The detailed fact to store, preserving all specific names, dates, and numbers

Guidelines:
- This is a conversation between TWO PEOPLE — always attribute facts to the correct speaker
- Include speaker names in titles and summaries
- NEVER generalize specifics: "3 times a week" not "regularly", "Toyota Camry" not "a car", "Eternal Sunshine" not "a movie"
- When a speaker lists items, include EVERY item: "Paris, London, Rome" not "several European cities"
- For dates: preserve the exact expression from conversation. If relative ("last Tuesday"), also note the session date for context
- Separate facts about different speakers into different memories
- Skip only pure greetings and filler words
- A typical conversation chunk should produce 4-10 memories

Common mistakes to AVOID:
- "has several hobbies" → CORRECT: "hobbies include pottery, hiking, and reading sci-fi novels"
- "sometime in early 2023" → CORRECT: "January 29, 2023"
- "has children" → CORRECT: "has 3 children named Emma, Jack, and Lily"

Return strict JSON (no markdown fences):
{
  "memories": [
    {
      "title": "Caroline: researching adoption agencies",
      "summary": "Caroline is researching LGBTQ-friendly adoption agencies in Boston as a single parent",
      "keywords": ["Caroline", "adoption", "LGBTQ", "single parent", "Boston"],
      "importance": 0.7,
      "content": "Caroline revealed she is researching adoption agencies in Boston that support LGBTQ+ individuals. She has contacted 3 agencies so far and plans to complete her application by December 2023. [Verbatim] Caroline said: 'I've been looking into adoption agencies in Boston, specifically ones that are LGBTQ+ friendly. I've reached out to 3 so far and hope to have my application done by December.'"
    }
  ]
}
If nothing worth remembering, return { "memories": [] }"""

_LLM_CHUNK_SIZE = 10


def _parse_extraction_response(text: str | None) -> list[dict]:
    if not text:
        return []
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    idx = re.search(r"[\[{]", cleaned)
    if idx and idx.start() > 0:
        cleaned = cleaned[idx.start() :]
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("memories"), list):
            return []
        results = []
        for item in parsed["memories"]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()
            if not title or not content:
                continue
            summary = str(item.get("summary", "")).strip() or truncate_text(content, 100)
            kws = item.get("keywords", [])
            if not isinstance(kws, list):
                kws = extract_keywords(content)[:8]
            else:
                kws = [str(k) for k in kws if isinstance(k, str)]
            imp = item.get("importance", 0.5)
            if not isinstance(imp, (int, float)):
                imp = 0.5
            imp = max(0.1, min(0.9, float(imp)))
            results.append(
                {
                    "title": title,
                    "summary": summary,
                    "keywords": kws,
                    "importance": imp,
                    "content": content,
                }
            )
        return results
    except (json.JSONDecodeError, ValueError):
        return []


async def write_memories_from_session_with_llm(
    cfg: PluginConfig,
    call_llm: LLMCallFn,
    turns: list[dict],
    session_id: int,
    session_date: str,
) -> int:
    date_iso = f"{session_date}T12:00:00Z"
    total = 0

    for chunk_start in range(0, len(turns), _LLM_CHUNK_SIZE):
        chunk = turns[chunk_start : chunk_start + _LLM_CHUNK_SIZE]
        chunk_index = chunk_start // _LLM_CHUNK_SIZE
        conv_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in chunk)

        prompt = (
            _EXTRACTION_PROMPT.replace("{sessionId}", str(session_id))
            .replace("{sessionDate}", session_date)
            .replace("{conversationText}", conv_text)
        )
        resp = await call_llm(prompt)
        memories = _parse_extraction_response(resp)

        if not memories and resp:
            retry = (
                "Your last response was not valid JSON. Please return only JSON without markdown code fences.\n\n"
                f"Your previous output:\n{resp[:500]}\n\n"
                'Please correct to strict JSON: { "memories": [...] }'
            )
            retry_resp = await call_llm(retry)
            memories = _parse_extraction_response(retry_resp)

        for i, mem in enumerate(memories):
            slug = f"session-{session_id}-llm-c{chunk_index}-{i}"
            fp = os.path.join(cfg.memory_root, "sessions", cfg.user_id, f"{slug}.md")
            if os.path.isfile(fp):
                continue
            fm = MemoryFrontmatter(
                id=f"sessions/{cfg.user_id}/{slug}",
                type="session",
                status="active",
                confidence="high",
                importance=mem["importance"],
                access_count=0,
                skip_count=0,
                token_estimate=_estimate_tokens(mem["content"]),
                created_at=date_iso,
                updated_at=date_iso,
                last_accessed_at=date_iso,
                title=truncate_text(mem["title"], 60),
                summary=mem["summary"],
                keywords=mem["keywords"][:10],
                source=f"daily/{session_date}.md",
                user_id=cfg.user_id,
            )
            body = (
                f"## Summary\n\n{mem['summary']}\n\n<!-- SUMMARY_END -->\n\n"
                f"**Session:** {session_id}\n**Date:** {session_date}\n\n"
                f"{mem['content']}\n\n## Change History\n- {session_date}: Created by LLM memory extraction"
            )
            write_memory_document(fp, fm, body)
            total += 1

    _update_session_topics(cfg, turns)
    return total
