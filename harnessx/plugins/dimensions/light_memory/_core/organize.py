# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Awaitable, Callable

from .backend import (
    parse_memory_document,
    round_val,
    write_memory_document,
)
from .daily import read_recent_daily_entries
from .decay import sort_by_decayed_importance
from .index_file import remove_index_entry, update_index_entry
from .types import MemoryDocument, MemoryFrontmatter, PluginConfig

LLMCallFn = Callable[[str], Awaitable[str | None]]

# ── JSON response parsing ────────────────────────────────────────────────────


def _parse_json_response(text: str | None):
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    idx = re.search(r"[\[{]", cleaned)
    if idx and idx.start() > 0:
        cleaned = cleaned[idx.start() :]
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


# ── Organization prompt ──────────────────────────────────────────────────────

_ORG_PROMPT = """\
You are a memory consolidation engine. Review the following memory store from conversations between two people and perform three maintenance tasks.

Current memory inventory (sorted by decayed importance):
{memorySummary}

Recent conversation records:
{recentDaily}

### Task 1: Deduplication & Merge
If two memories describe the same fact about the SAME person, suggest merging into a single memory.
- Keep the more complete one as the target
- The other becomes the source and will be marked as deprecated
- Provide a merged summary
- Do NOT merge memories about different speakers
- Do NOT merge memories about different topics even if about the same person

### Task 2: Conflict Detection
If two memories genuinely contradict each other about the same person and topic, keep the newer one.
- Mark the older one as deprecated, provide the reason
- Ensure speaker attribution is correct before deprecating

### Task 3: Speaker Profile Update
Synthesize all memories to build/update profiles for each speaker in the conversation.
Generate a combined profile document covering both speakers, organized by person.
Include: personal details, interests, skills, preferences, plans, relationships, key events, and timeline.
If no meaningful new information since last update, do not generate an update_profile action.

Return strict JSON (do NOT wrap in markdown code fences):
{{
  "actions": [
    {{ "action": "merge", "source": "older file path", "target": "kept file path", "mergedSummary": "merged one-line summary" }},
    {{ "action": "deprecate", "file": "file path", "reason": "contradicts xxx, superseded by newer memory" }},
    {{ "action": "update_profile", "content": "complete profile markdown with sections for each speaker" }}
  ]
}}
If no actions needed, return {{ "actions": [] }}"""


# ── Main entry point ─────────────────────────────────────────────────────────


async def run_organization(
    cfg: PluginConfig,
    call_llm: LLMCallFn,
    all_docs: list[MemoryDocument],
    now: datetime | None = None,
) -> list[str]:
    if now is None:
        now = datetime.now(timezone.utc)

    ranked = sort_by_decayed_importance(all_docs, cfg.access_half_life_days, now)
    summary_lines = []
    for doc, decay_imp in ranked:
        fm = doc.frontmatter
        skip_info = f", access={fm.access_count}, skip={fm.skip_count}" if fm.skip_count > 0 else ""
        summary_lines.append(
            f"- [{fm.type}] {fm.title} (importance={round_val(decay_imp, 2)}, "
            f"confidence={fm.confidence}{skip_info}, path={doc.relative_path})\n"
            f"  Summary: {fm.summary}\n  keywords: {', '.join(fm.keywords)}"
        )
    memory_summary = "\n".join(summary_lines) or "(empty)"
    recent_daily = read_recent_daily_entries(cfg, 3, now) or "(no recent records)"

    prompt = _ORG_PROMPT.replace("{memorySummary}", memory_summary).replace("{recentDaily}", recent_daily)
    llm_response = await call_llm(prompt)
    org_result = _parse_org_response(llm_response)

    if org_result is None and llm_response:
        retry_prompt = (
            "Your last response was not valid JSON. Please return only JSON without "
            "markdown code fences.\n\nYour previous output:\n"
            f"{llm_response[:500]}\n\nPlease correct to strict JSON format: "
            '{ "actions": [...] }'
        )
        retry_response = await call_llm(retry_prompt)
        org_result = _parse_org_response(retry_response)

    if not org_result or not org_result.get("actions"):
        return []

    changed: list[str] = []
    for action in org_result["actions"]:
        try:
            changed.extend(_execute_action(cfg, action, now))
        except Exception:
            pass
    return changed


def _parse_org_response(text: str | None) -> dict | None:
    parsed = _parse_json_response(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("actions"), list):
        return None
    return parsed


# ── Action execution ─────────────────────────────────────────────────────────


def _execute_action(cfg: PluginConfig, action: dict, now: datetime) -> list[str]:
    kind = action.get("action")
    if kind == "merge":
        return _execute_merge(cfg, action, now)
    if kind == "deprecate":
        return _execute_deprecate(cfg, action, now)
    if kind == "update_profile":
        return _execute_update_profile(cfg, action, now)
    return []


def _resolve_path(root: str, input_path: str) -> str | None:
    if not input_path.strip():
        return None
    resolved_root = os.path.abspath(root)
    candidate = (
        os.path.abspath(os.path.join(root, input_path))
        if not os.path.isabs(input_path)
        else os.path.abspath(input_path)
    )
    if candidate == resolved_root or candidate.startswith(resolved_root + os.sep):
        return candidate
    return None


def _execute_merge(cfg: PluginConfig, action: dict, now: datetime) -> list[str]:
    source_str = action.get("source", "")
    target_str = action.get("target", "")
    if not isinstance(source_str, str) or not isinstance(target_str, str):
        return []
    source_path = _resolve_path(cfg.memory_root, source_str)
    target_path = _resolve_path(cfg.memory_root, target_str)
    if not source_path or not target_path:
        return []

    source_doc = parse_memory_document(source_path, cfg.memory_root)
    target_doc = parse_memory_document(target_path, cfg.memory_root)
    if not target_doc:
        return []

    now_iso = now.isoformat()
    merged_kw = list(
        dict.fromkeys(target_doc.frontmatter.keywords + (source_doc.frontmatter.keywords if source_doc else []))
    )
    combined_summary = (
        f"{target_doc.frontmatter.summary} | {source_doc.frontmatter.summary}"
        if source_doc
        else target_doc.frontmatter.summary
    )

    target_fm = MemoryFrontmatter(
        **{
            **target_doc.frontmatter.__dict__,
            "summary": combined_summary[:200],
            "keywords": merged_kw,
            "importance": max(
                target_doc.frontmatter.importance,
                source_doc.frontmatter.importance if source_doc else 0,
            ),
            "supersedes": list(dict.fromkeys(target_doc.frontmatter.supersedes + [source_str])),
            "updated_at": now_iso,
        }
    )
    merged_body = target_doc.body
    if source_doc:
        merge_marker = f"## Merged from {source_str}"
        if merge_marker not in merged_body:
            merged_body += f"\n\n{merge_marker} ({now_iso[:10]})\n\n{source_doc.body}"
    write_memory_document(target_path, target_fm, merged_body)
    updated_target = parse_memory_document(target_path, cfg.memory_root)
    if updated_target:
        update_index_entry(cfg, updated_target)

    paths = [target_str]
    if source_doc:
        source_fm = MemoryFrontmatter(
            **{
                **source_doc.frontmatter.__dict__,
                "status": "deprecated",
                "confidence": "deprecated",
                "updated_at": now_iso,
            }
        )
        write_memory_document(source_path, source_fm, source_doc.body)
        updated_source = parse_memory_document(source_path, cfg.memory_root)
        if updated_source:
            update_index_entry(cfg, updated_source)
        paths.append(source_str)
    elif os.path.isfile(source_path):
        os.unlink(source_path)
        remove_index_entry(cfg, source_str)
        paths.append(source_str)

    return paths


def _execute_deprecate(cfg: PluginConfig, action: dict, now: datetime) -> list[str]:
    file_str = action.get("file", "")
    if not isinstance(file_str, str):
        return []
    file_path = _resolve_path(cfg.memory_root, file_str)
    if not file_path:
        return []
    doc = parse_memory_document(file_path, cfg.memory_root)
    if not doc:
        return []

    reason = action.get("reason", "deprecated by organization")
    fm = MemoryFrontmatter(
        **{
            **doc.frontmatter.__dict__,
            "status": "deprecated",
            "confidence": "deprecated",
            "updated_at": now.isoformat(),
        }
    )
    write_memory_document(file_path, fm, f"{doc.body}\n\n## Deprecated\n{reason}")
    updated_doc = parse_memory_document(file_path, cfg.memory_root)
    if updated_doc:
        update_index_entry(cfg, updated_doc)
    return [file_str]


def _extract_speaker_names(content: str) -> list[str]:
    names: list[str] = []
    skip = {
        "Summary",
        "Profile",
        "Speaker",
        "Details",
        "Overview",
        "Background",
        "Change",
        "History",
        "Merged",
        "Deprecated",
    }
    for m in re.finditer(r"^#{1,4}\s+([A-Z][a-z]+)", content, re.MULTILINE):
        name = m.group(1)
        if name not in skip and name not in names:
            names.append(name)
    return names[:4]


def _execute_update_profile(cfg: PluginConfig, action: dict, now: datetime) -> list[str]:
    content = action.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return []
    profile_path = _resolve_path(cfg.memory_root, "user/profile.md")
    if not profile_path:
        return []
    existing = parse_memory_document(profile_path, cfg.memory_root) if os.path.isfile(profile_path) else None
    now_iso = now.isoformat()
    speaker_names = _extract_speaker_names(content)

    fm = MemoryFrontmatter(
        id="user/profile",
        type="profile",
        status="active",
        confidence="high",
        importance=existing.frontmatter.importance if existing else 0.9,
        access_count=existing.frontmatter.access_count if existing else 0,
        skip_count=existing.frontmatter.skip_count if existing else 0,
        token_estimate=max(1, len(content) // 4),
        created_at=existing.frontmatter.created_at if existing else now_iso,
        updated_at=now_iso,
        last_accessed_at=existing.frontmatter.last_accessed_at if existing else now_iso,
        title=f"Speaker Profiles: {' & '.join(speaker_names)}" if speaker_names else "Speaker Long-term Profiles",
        summary=(
            f"Combined profile of {' and '.join(speaker_names)}: interests, skills, preferences, relationships, and key events"
            if speaker_names
            else "Speaker profiles including interests, skills, preferences, and habits"
        ),
        keywords=["profile"] + [n.lower() for n in speaker_names],
        user_id=cfg.user_id,
    )

    body = content
    if existing:
        body += f"\n\n## Previous Profile Details\n\n{existing.body}"
    write_memory_document(profile_path, fm, body)
    return ["user/profile.md"]
