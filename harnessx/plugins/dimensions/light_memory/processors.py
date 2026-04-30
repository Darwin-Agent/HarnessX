# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import dataclasses
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from ....core.processor import MultiHookProcessor
from ....processors._sp_utils import sp_append
from ._core.backend import (
    RELEVANT_MEMORIES_TAG,
    commit_files_if_needed,
    extract_conversation_turns,
    extract_keywords,
    get_entity_keyword_hits,
    grep_memory_files_with_scores,
    merge_session_topics,
    parse_memory_document,
    round_val,
)
from ._core.daily import append_to_daily
from ._core.decay import sort_by_decayed_importance

if TYPE_CHECKING:
    from ....core.events import TaskEndEvent, TaskStartEvent
    from ....providers.base import BaseModelProvider
    from ._core.types import PluginConfig


# ─── Retrieval ───────────────────────────────────────────────────────────────


class LightMemoryRetrievalProcessor(MultiHookProcessor):
    """Inject recall candidate list + operation guidance into the system prompt.

    Runs keyword search + entity augmentation + decay ranking, then builds a
    compact candidate table (title | path | importance | summary) that the
    agent can use to selectively read memory files with its built-in tools.
    Operation guidance (English) is always injected so the agent knows how
    to create, update, and organise memory files during the conversation.

    Order 6 — runs after SystemPromptProcessor (1) and
    EnvironmentContextInjector (5).
    """

    _order = 6

    def __init__(self) -> None:
        self._cfg: "PluginConfig | None" = None

    def configure(self, cfg: "PluginConfig") -> None:
        self._cfg = cfg

    async def on_task_start(self, event: "TaskStartEvent") -> AsyncIterator["TaskStartEvent"]:
        cfg = self._cfg
        if cfg is None or not cfg.auto_recall:
            yield event
            return

        parts: list[str] = [f"<{RELEVANT_MEMORIES_TAG}>"]

        # Build recall content when the prompt is non-trivial
        prompt_text = event.task_description or ""
        if len(prompt_text.strip()) >= 2:
            try:
                recall = _perform_recall(cfg, prompt_text)
                if recall:
                    parts.append(recall)
                    parts.append("")
            except Exception:
                pass

        parts.append(_build_operation_guidance(cfg))
        parts.append(f"</{RELEVANT_MEMORIES_TAG}>")

        section = "\n\n" + "\n".join(parts) + "\n"
        new_sp = sp_append(event.system_prompt, section)
        yield dataclasses.replace(event, system_prompt=new_sp)


def _perform_recall(cfg: "PluginConfig", prompt: str) -> str | None:
    """Grep + entity augment + decay sort → candidate table string."""
    # Step 1: keyword extraction + session-topics supplement
    prompt_kw = extract_keywords(prompt)
    keywords = merge_session_topics(cfg, prompt_kw)

    grep_hits = grep_memory_files_with_scores(cfg.memory_root, keywords, cfg.top_k * 5)

    # Step 2: entity-name augmentation
    entity_names = list(set(re.findall(r"\b[A-Z][a-z]{2,}\b", prompt)))
    if entity_names:
        grep_paths = {h.relative_path for h in grep_hits}
        entity_hits = get_entity_keyword_hits(cfg.memory_root, entity_names, 20)
        for hit in entity_hits:
            if hit.relative_path not in grep_paths:
                grep_hits.append(hit)

    # Step 3: parse → filter active
    candidates = []
    for hit in grep_hits:
        fp = os.path.join(cfg.memory_root, hit.relative_path)
        doc = parse_memory_document(fp, cfg.memory_root)
        if doc and doc.frontmatter.status == "active":
            candidates.append(doc)

    if not candidates:
        return None

    # Step 4: decay sort, top-K
    ranked = sort_by_decayed_importance(candidates, cfg.access_half_life_days)
    top = ranked[: cfg.top_k]

    # Step 5: style.md always injected (small, high-value)
    style_content = ""
    style_path = os.path.join(cfg.memory_root, "user", "style.md")
    if os.path.isfile(style_path):
        style_doc = parse_memory_document(style_path, cfg.memory_root)
        if style_doc:
            style_content = f"## Reply style preferences\n{style_doc.body}"

    # Step 6: build candidate table
    rows: list[str] = []
    for doc, decay_imp in top:
        fm = doc.frontmatter
        rows.append(f"| {fm.title} | {doc.relative_path} | {round_val(decay_imp, 2)} | {fm.summary} |")

    if not rows and not style_content:
        return None

    out: list[str] = []
    if style_content:
        out.append(style_content)
    if rows:
        out.append("## Candidate memories (sorted by relevance)")
        out.append("")
        out.append("| Title | Path | Importance | Summary |")
        out.append("|-------|------|------------|---------|")
        out.extend(rows)
        out.append("")
        out.append("> Use the `read` tool to inspect relevant memories before deciding whether to use them.")
    return "\n".join(out)


# ─── Capture ──────────────────────────────────────────────────────────────────


class LightMemoryCaptureProcessor(MultiHookProcessor):
    """Append the conversation to the daily log and trigger background organisation.

    The agent writes semantic memories (sessions/, knowledge/, etc.) during the
    conversation using its built-in file tools.  This processor only handles:
    - Appending the raw conversation to ``daily/YYYY-MM-DD.md``
    - Committing the daily file via git (when auto_commit is True)
    - Kicking off the periodic organisation pass as a background asyncio task

    Order 60 — runs late so it sees the final conversation state.
    """

    _order = 60

    def __init__(self) -> None:
        self._cfg: "PluginConfig | None" = None
        self._provider: "BaseModelProvider | None" = None
        self._org_last_started_monotonic: float = 0.0
        self._org_task: asyncio.Task | None = None

    def configure(
        self,
        cfg: "PluginConfig",
        provider: "BaseModelProvider | None",
    ) -> None:
        self._cfg = cfg
        self._provider = provider

    async def on_task_end(self, event: "TaskEndEvent") -> AsyncIterator["TaskEndEvent"]:
        cfg = self._cfg
        if cfg is None:
            yield event
            return

        # TaskEndEvent in current core API does not expose `success`.
        # Keep compatibility with older event shapes while using exit_reason/error
        # as the canonical signal on new versions.
        is_success = getattr(event, "success", None)
        if is_success is None:
            is_success = getattr(event, "exit_reason", "done") == "done" and not bool(getattr(event, "error", ""))
        if not bool(is_success):
            yield event
            return

        if cfg.auto_capture:
            messages = getattr(event, "final_messages", None) or []
            turns = extract_conversation_turns(messages, strip_memory_tag=True)
            if turns:
                now = datetime.now(timezone.utc)

                # Daily log capture
                daily_path: str | None = None
                try:
                    entries = [f"**{t['role'].capitalize()}:** {t['text']}" for t in turns]
                    daily_entry = "\n\n---\n\n".join(entries)
                    rel_path, _line_range = append_to_daily(cfg, daily_entry, now)
                    daily_path = rel_path
                except Exception:
                    pass

                # Git commit daily file
                if daily_path:
                    try:
                        commit_files_if_needed(
                            cfg,
                            [daily_path],
                            "memory(daily): append conversation",
                        )
                    except Exception:
                        pass

        # Background organisation (decay archive + optional LLM consolidation)
        if cfg.organization_enabled:
            # In-process run lock
            if self._org_task is not None and not self._org_task.done():
                yield event
                return

            now_mono = time.monotonic()
            interval_s = max(
                60.0,
                float(getattr(cfg, "organization_interval_ms", 1_800_000)) / 1000.0,
            )
            if self._org_last_started_monotonic and (now_mono - self._org_last_started_monotonic) < interval_s:
                yield event
                return

            self._org_last_started_monotonic = now_mono
            try:
                self._org_task = asyncio.create_task(
                    _run_organization_bg(
                        cfg,
                        self._provider,
                        timeout_ms=int(getattr(cfg, "organization_timeout_ms", 30_000)),
                    )
                )
            except RuntimeError:
                # No running event loop (e.g. in unit tests) — skip background run.
                self._org_task = None

        yield event


# ─── Background organisation ──────────────────────────────────────────────────


async def _run_organization_bg(
    cfg: "PluginConfig",
    provider: "BaseModelProvider | None",
    *,
    timeout_ms: int = 30_000,
) -> None:
    """Run the full organisation pipeline in the background.

    1. Archive memories below the decay threshold (mechanical).
    2. LLM-driven consolidation when a provider is available.
    3. Rebuild the index.
    """
    from datetime import datetime, timezone

    from ._core.backend import (
        ORGANIZATION_SCAN_DIRS,
        get_all_memory_documents,
        write_memory_document,
    )
    from ._core.decay import find_decayed_memories
    from ._core.index_file import generate_index_file
    from ._core.organize import run_organization

    def _acquire_file_lock() -> Path | None:
        lock_path = Path(cfg.memory_root) / ".ops" / "organization.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stale_after_s = max(
            300.0,
            float(timeout_ms) / 1000.0 * 3.0,
            float(getattr(cfg, "organization_interval_ms", 1_800_000)) / 1000.0 * 2.0,
        )
        try:
            if lock_path.exists():
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_after_s:
                    lock_path.unlink(missing_ok=True)
        except Exception:
            pass

        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                fp.write(f"pid={os.getpid()} ts={int(time.time())}\n")
            return lock_path
        except FileExistsError:
            return None
        except Exception:
            return None

    async def _work() -> None:
        all_docs = get_all_memory_documents(cfg, ORGANIZATION_SCAN_DIRS)
        if not all_docs:
            return

        now = datetime.now(timezone.utc)

        # Step 1: archive decayed memories
        to_archive = find_decayed_memories(all_docs, cfg.access_half_life_days, now=now)
        archived: list[str] = []
        for doc in to_archive:
            import dataclasses as _dc

            updated_fm = _dc.replace(
                doc.frontmatter,
                status="archived",
                updated_at=now.isoformat(),
            )
            write_memory_document(doc.file_path, updated_fm, doc.body)
            archived.append(doc.relative_path)

        if archived:
            try:
                commit_files_if_needed(
                    cfg,
                    archived,
                    f"memory(archive): {len(archived)} memories below decay threshold",
                )
            except Exception:
                pass

        # Step 2: LLM consolidation
        if provider is not None:
            call_llm = _make_llm_caller(provider)
            fresh = get_all_memory_documents(cfg, ORGANIZATION_SCAN_DIRS)
            changed = await run_organization(cfg, call_llm, fresh, now)
            if changed:
                try:
                    commit_files_if_needed(
                        cfg,
                        changed,
                        f"memory(organize): consolidation — {len(changed)} files changed",
                    )
                except Exception:
                    pass

        # Step 3: Rebuild index
        final_docs = get_all_memory_documents(cfg)
        generate_index_file(cfg, final_docs)
        try:
            commit_files_if_needed(cfg, ["_index.md"], "memory(index): rebuild index")
        except Exception:
            pass

    lock_path = _acquire_file_lock()
    if lock_path is None:
        return
    try:
        await asyncio.wait_for(_work(), timeout=max(1.0, float(timeout_ms) / 1000.0))
    except Exception:
        # Fail-open: memory maintenance must not fail user task runs.
        pass
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def _make_llm_caller(provider: "BaseModelProvider"):
    """Wrap a provider into the simple async (prompt) -> str | None signature."""
    from ....core.events import Message

    async def _call(prompt: str) -> str | None:
        try:
            msgs: list[Message] = [Message(role="user", content=prompt)]
            resp = await provider.complete(messages=msgs, tools=[])
            return resp.content.strip() if resp.content else None
        except Exception:
            return None

    return _call


# ─── Operation guidance (English) ────────────────────────────────────────────


def _build_operation_guidance(cfg: "PluginConfig") -> str:
    root = cfg.memory_root
    uid = cfg.user_id
    return f"""\
## Memory operation guidance

You have a persistent memory system at `{root}/` (a git repository).
The system automatically injects candidate memories above and appends raw
conversations to `daily/` after the session.  **During the conversation,
use your built-in file tools (read / write / edit / bash) to manage memory
files directly.**  Deduplication, conflict detection, and profile updates are
handled by a background task — focus on recall, write, and update.

### Directory structure
- `user/style.md` — reply preferences (verbosity, language, format)
- `user/profile.md` — long-term profile (role, interests, tech stack)
- `knowledge/skill/` — skill knowledge and best practices
- `knowledge/learning/` — correction records (wrong belief → correct one)
- `entity/` — topics, projects, and named-entity aggregations
- `sessions/{uid}/` — extracted session highlights (not raw transcripts)
- `daily/` — raw daily logs (auto-generated, do not edit manually)
- `_index.md` — global memory map

### When to write / update a memory
Write or update a memory file when you observe:
- **User corrects your assumption or approach** → write `{root}/knowledge/learning/{{slug}}.md`
- **User states an explicit preference** ("I prefer", "always use", "never again") → edit `{root}/user/style.md` or `user/profile.md`
- **New skill knowledge or best practice** → write `{root}/knowledge/skill/{{slug}}.md`
- **Project / topic constraint or decision** → write `{root}/entity/{{slug}}.md`
- **Valuable session highlight worth preserving** → write `{root}/sessions/{uid}/{{slug}}.md`
  Example: "We had the project review meeting tonight at 7 pm" → extract: title, time, participants, key conclusions.
- **User expresses emotional state** ("feeling stressed", "really happy", "overwhelmed") → edit `{root}/user/profile.md`, section `## Recent emotional state`, format: `- [YYYY-MM-DD] description`, keep last 5 entries.
- **User gives positive or negative feedback about your responses** → edit `{root}/user/profile.md`, section `## Relationship notes`, format: `- [YYYY-MM-DD] feedback — your interpretation`, keep last 5 entries.
- **When replying**, check `## Recent emotional state` in `user/profile.md` and adjust your tone accordingly (warmer when user is stressed, more upbeat when positive).

### sessions/ writing rules
`sessions/` holds **extracted semantic highlights**, not raw conversation copies.
- Extract key facts, times, people, and conclusions as concise statements.
- Include specific dates and times ("2026-03-30 19:00–20:00").
- Filter out pleasantries, repetition, and content-free turns.
- Never generalise: "3×/week" not "frequently", "Toyota Camry" not "a car".
- List items must be complete: "Beijing, Shanghai, Shenzhen" not "several cities".
- If a relative date appears ("last Tuesday"), append the session date so it can be reconstructed later.

### Memory file format
Each memory file must contain YAML frontmatter + summary section + `SUMMARY_END` marker + body + change history:

```
---
id: "knowledge/skill/{{slug}}"
type: skill                   # style | profile | session | skill | learning | entity
status: active                # active | deprecated | archived
confidence: high              # high | medium | low | deprecated
importance: 0.7               # 0–1, affects recall ranking; never auto-modified
access_count: 0
skip_count: 0
token_estimate: 100
created_at: "ISO8601"
updated_at: "ISO8601"
last_accessed_at: "ISO8601"
title: "Short title"
summary: "One-sentence summary (≤100 chars)"
keywords: ["keyword1", "keyword2"]
related: []
source: ""
supersedes: []
user_id: "{uid}"
---

## Summary

One-sentence summary.

<!-- SUMMARY_END -->

Detailed content...

## Change history
- YYYY-MM-DD: Initial creation
```

### Memory recall — using the candidate list
The "Candidate memories" table above lists files likely relevant to this task.
1. Scan titles and summaries; identify which are relevant.
2. For relevant candidates, `read` the file's summary section (between frontmatter and `SUMMARY_END`).
3. If the summary is insufficient, read the full file.
4. Ignore irrelevant candidates — no need to read everything.
   **Special case**: when `user/profile.md` appears in the list, read its `## Recent emotional state` and `## Relationship notes` sections.
5. **Cross-memory association**: when multiple memories concern the same person or topic, synthesise them.  If one uses a vague label ("a colleague", "that project"), replace it with the specific name found in another memory.
6. **When you actually use a memory to inform your reply**, edit its frontmatter:
   - `access_count: +1`
   - `last_accessed_at: <current ISO8601 time>`
7. **When you read a memory's summary but find it irrelevant**, edit its frontmatter:
   - `skip_count: +1`
8. **When `skip_count > 3` and `skip_count / (access_count + skip_count) > 0.7`**, the keywords are too broad — edit `keywords` to be more specific and reset `skip_count` to 0.

### Active search beyond the candidate list
When fewer than 2 candidates appear, proactively browse:
- Global index: `read {root}/_index.md`
- Directory listing: `ls {root}/knowledge/skill/` etc., then `read` promising files.

### Updating existing memories
- Use `edit` to append content; do not overwrite existing content.
- Update `updated_at` in frontmatter.
- Append a change-history entry at the end.

### Capacity limits
- Body (excluding frontmatter) must not exceed 3 000 characters.
- If content is too long, split into related files and link them via `related`.
- Files under `sessions/` older than 30 days with `importance < 0.1` are auto-archived.

### Session topics
If new topics, projects, or key entities arise during the conversation,
append them (one per line, latest 10) to `{root}/.ops/session_topics.txt`.
The system reads this file on the next recall to improve context continuity.

### _index.md maintenance
After creating a new memory file, update `{root}/_index.md` by appending a
row to the relevant section table."""
