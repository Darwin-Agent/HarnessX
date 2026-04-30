# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Cross-round journal — structured multi-round memory for the meta-agent.

The journal is a single markdown file with one ``## Round N — <label>``
section per evolve. Each section is an HTML-commented YAML frontmatter
block (machine-parseable, agent-writable) followed by free prose
(Why / Changes / Evidence / Uncertainty suggested, not required).

Schema — required frontmatter keys::

    round: int                      # monotonic round index
    timestamp: str (ISO-8601 UTC)   # when this entry was appended
    hypothesis_id: str              # short slug, stable across rounds
    levers: list[str]               # subset of {configuration, control, action, instruction}
    predicted_affected: list[str]   # task_ids the round's author expects to flip F→T
    gating_outcome: str             # pending | accepted | reverted | noop (orchestrator fills after eval)
    gating_attribution: dict|str    # pending | {task_id: flipped|still_F|regressed|still_T|absent}
    changeset: dict                 # optional: orchestrator-computed config diff (tools/processors/templates)

Everything after the frontmatter block is free prose. Authors are
encouraged — not required — to use ``### Why`` / ``### Changes`` /
``### Evidence`` / ``### Uncertainty`` subheadings for consistency.
Aggregators ignore the prose; they only read the frontmatter.

Public API:

- ``append_entry(journal_path, entry)`` — append a new round section.
- ``read_entries(journal_path)`` — iterate parsed entries (frontmatter
  only; body kept as raw string).
- ``fill_gating(journal_path, round_idx, outcome, attribution)`` —
  mutate the prior round's frontmatter to record how it performed.
  Writes in place; preserves prose. Idempotent.
- ``build_context(journal_path, round_idx)`` — produce
  ``_meta_scratch/CONTEXT.md`` for the NEXT evolve round: recent
  hypotheses, what's been tried on each lever, attribution history.
  Orchestrator calls this before invoking ``evolve()``.

Entries are parsed forgivingly: a malformed frontmatter block is
skipped rather than aborting; ``read_entries`` returns what it can.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


_VALID_LEVERS = frozenset({"configuration", "control", "action", "instruction"})


# HTML-commented frontmatter block — comment syntax so it renders
# cleanly on GitHub / human readers and doesn't collide with markdown
# headings that start with ``---``.
_FRONTMATTER_OPEN = "<!-- journal:frontmatter"
_FRONTMATTER_CLOSE = "-->"


_SECTION_RE = re.compile(
    r"^## Round (?P<round>\d+)\s*(?:—|-)\s*(?P<label>.+?)\s*$",
    re.MULTILINE,
)
# Match a frontmatter block. Greedy for the close marker so a stray
# "-->" inside prose doesn't truncate the block.
_FRONTMATTER_RE = re.compile(
    re.escape(_FRONTMATTER_OPEN) + r"\s*\n(?P<yaml>.*?)\n\s*" + re.escape(_FRONTMATTER_CLOSE),
    re.DOTALL,
)


@dataclass
class JournalEntry:
    """One parsed ``## Round N`` section.

    ``frontmatter`` holds the machine-parseable YAML fields.
    ``label`` is the agent-authored hypothesis label (from the heading).
    ``prose`` is the full markdown body between the frontmatter close
    and the next ``## Round`` (or EOF).
    """

    round: int
    label: str
    frontmatter: dict[str, Any]
    prose: str

    @property
    def hypothesis_id(self) -> str:
        return str(self.frontmatter.get("hypothesis_id", ""))

    @property
    def levers(self) -> list[str]:
        value = self.frontmatter.get("levers") or []
        if not isinstance(value, list):
            return []
        return [str(v).lower() for v in value if isinstance(v, str)]

    @property
    def predicted_affected(self) -> list[str]:
        value = self.frontmatter.get("predicted_affected") or []
        if not isinstance(value, list):
            return []
        return [str(v) for v in value]

    @property
    def gating_outcome(self) -> str:
        return str(self.frontmatter.get("gating_outcome", "pending")).lower()

    @property
    def gating_attribution(self) -> dict[str, str]:
        value = self.frontmatter.get("gating_attribution", "pending")
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        return {}


@dataclass
class JournalEntrySpec:
    """What the agent writes to append a new round.

    ``round``, ``hypothesis_id``, ``levers``, ``predicted_affected``
    are required. ``label`` becomes the ``## Round N —`` heading text.
    ``prose`` is the free markdown body. ``extra_frontmatter`` lets
    benchmarks add fields without schema changes here.
    """

    round: int
    label: str
    hypothesis_id: str
    levers: list[str]
    predicted_affected: list[str]
    prose: str
    extra_frontmatter: dict[str, Any] = field(default_factory=dict)


# ─── Append ───────────────────────────────────────────────────────────────


def append_entry(journal_path: Path, entry: JournalEntrySpec) -> None:
    """Append one ``## Round N`` section to ``journal_path``.

    The entry's ``round`` must not already be present — duplicate
    rounds raise ``ValueError`` so a buggy orchestrator doesn't
    silently double-write and break attribution.
    """
    journal_path = Path(journal_path)

    for lever in entry.levers:
        if lever.lower() not in _VALID_LEVERS:
            raise ValueError(f"unknown lever {lever!r}; valid: {sorted(_VALID_LEVERS)}")

    if journal_path.is_file():
        existing = list(read_entries(journal_path))
        for e in existing:
            if e.round == entry.round:
                raise ValueError(f"journal already has a Round {entry.round} section")

    # Microsecond precision: same-second double appends in tests / fast
    # retries remain distinguishable in audit logs even though append
    # order is the authoritative sort key.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    fm = {
        "round": entry.round,
        "timestamp": now,
        "hypothesis_id": entry.hypothesis_id,
        "levers": [lev.lower() for lev in entry.levers],
        "predicted_affected": list(entry.predicted_affected),
        "gating_outcome": "pending",
        "gating_attribution": "pending",
    }
    fm.update(entry.extra_frontmatter)

    yaml_block = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).rstrip()

    section = (
        f"## Round {entry.round} — {entry.label}\n\n"
        f"{_FRONTMATTER_OPEN}\n{yaml_block}\n{_FRONTMATTER_CLOSE}\n\n"
        f"{entry.prose.rstrip()}\n"
    )

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if journal_path.is_file() else "w"
    with journal_path.open(mode, encoding="utf-8") as f:
        if mode == "a":
            f.write("\n")
        f.write(section)


# ─── Read ─────────────────────────────────────────────────────────────────


def read_entries(journal_path: Path) -> list[JournalEntry]:
    """Parse all ``## Round N`` sections from ``journal_path``.

    Returns a list ordered by round ascending. Malformed sections
    (unparseable frontmatter, missing required keys) are logged and
    skipped.
    """
    journal_path = Path(journal_path)
    if not journal_path.is_file():
        return []

    text = journal_path.read_text(encoding="utf-8")
    entries: list[JournalEntry] = []

    matches = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        header_end = m.end()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[header_end:next_start]

        fm_match = _FRONTMATTER_RE.search(body)
        if fm_match is None:
            logger.debug(
                "[journal] round %s: missing frontmatter block, skipped",
                m.group("round"),
            )
            continue

        try:
            fm_data = yaml.safe_load(fm_match.group("yaml")) or {}
        except yaml.YAMLError as exc:
            logger.warning(
                "[journal] round %s: frontmatter YAML parse error (%s), skipped",
                m.group("round"),
                exc,
            )
            continue
        if not isinstance(fm_data, dict):
            logger.warning(
                "[journal] round %s: frontmatter is not a mapping, skipped",
                m.group("round"),
            )
            continue

        prose = body[fm_match.end() :].strip()

        try:
            round_idx = int(m.group("round"))
        except ValueError:
            continue

        entries.append(
            JournalEntry(
                round=round_idx,
                label=m.group("label").strip(),
                frontmatter=fm_data,
                prose=prose,
            )
        )

    entries.sort(key=lambda e: e.round)
    return entries


def latest_entry(journal_path: Path) -> JournalEntry | None:
    entries = read_entries(journal_path)
    return entries[-1] if entries else None


# ─── Fill gating outcome ──────────────────────────────────────────────────


def fill_gating(
    journal_path: Path,
    round_idx: int,
    outcome: str,
    attribution: dict[str, str],
    *,
    extra_frontmatter: dict[str, Any] | None = None,
) -> bool:
    """Back-fill ``gating_outcome`` and ``gating_attribution`` on an
    existing round's frontmatter.

    ``outcome`` must be one of ``accepted``, ``reverted``, ``noop``.
    ``attribution`` is a ``{task_id: flipped|still_F|regressed|still_T|absent}``
    dict.

    ``extra_frontmatter`` lets the orchestrator attach post-eval facts
    it alone observes — e.g. ``regressed_unpredicted`` (tasks that
    regressed but weren't predicted) and ``changeset`` (config diff).
    Extra keys are merged non-destructively into the entry's
    frontmatter; caller-provided keys overwrite existing values so this
    same call can back-fill both gating and the diff in one pass.

    Returns ``True`` on successful update, ``False`` when the round
    was not found or the frontmatter was malformed. Idempotent — a
    second call with the same arguments leaves the file unchanged.
    """
    valid_outcomes = {"accepted", "reverted", "noop"}
    if outcome not in valid_outcomes:
        raise ValueError(f"outcome must be one of {sorted(valid_outcomes)}, got {outcome!r}")

    journal_path = Path(journal_path)
    if not journal_path.is_file():
        return False

    text = journal_path.read_text(encoding="utf-8")

    # Find the target round section.
    section_matches = list(_SECTION_RE.finditer(text))
    target_idx = None
    for i, m in enumerate(section_matches):
        try:
            r = int(m.group("round"))
        except ValueError:
            continue
        if r == round_idx:
            target_idx = i
            break
    if target_idx is None:
        return False

    section_start = section_matches[target_idx].end()
    section_end = section_matches[target_idx + 1].start() if target_idx + 1 < len(section_matches) else len(text)
    section_body = text[section_start:section_end]

    fm_match = _FRONTMATTER_RE.search(section_body)
    if fm_match is None:
        return False

    try:
        fm_data = yaml.safe_load(fm_match.group("yaml")) or {}
    except yaml.YAMLError:
        return False
    if not isinstance(fm_data, dict):
        return False

    # Idempotency: if every field we're about to write already matches,
    # do nothing. Extra frontmatter participates so a back-fill that
    # only adds ``regressed_unpredicted`` / ``changeset`` is still
    # no-oppable on a replay.
    existing_outcome = str(fm_data.get("gating_outcome", "")).lower()
    existing_attr = fm_data.get("gating_attribution")
    extras = dict(extra_frontmatter or {})
    extras_match = all(fm_data.get(k) == v for k, v in extras.items())
    if existing_outcome == outcome and existing_attr == attribution and extras_match:
        return True

    fm_data["gating_outcome"] = outcome
    fm_data["gating_attribution"] = dict(attribution)
    for k, v in extras.items():
        fm_data[k] = v

    new_yaml = yaml.safe_dump(fm_data, sort_keys=False, allow_unicode=True).rstrip()
    new_section_body = (
        section_body[: fm_match.start()]
        + f"{_FRONTMATTER_OPEN}\n{new_yaml}\n{_FRONTMATTER_CLOSE}"
        + section_body[fm_match.end() :]
    )
    new_text = text[:section_start] + new_section_body + text[section_end:]
    journal_path.write_text(new_text, encoding="utf-8")
    return True


# ─── Build context for the next evolve ────────────────────────────────────


def build_context(
    journal_path: Path,
    current_round: int,
    output_path: Path,
    recent_window: int = 5,
) -> Path | None:
    """Render ``output_path`` (markdown) for the next evolve to Read.

    Combines:

    - Attribution summary for the last ``recent_window`` rounds:
      per-lever precision (how often predicted tasks actually flipped,
      discounted by unpredicted regressions the same round caused) and
      the accepted/reverted count.
    - A table of recent hypothesis ids and their outcomes, so the
      agent can see what's been tried and what landed.
    - A pointer to the full journal for anything not summarised.

    Precision accounting:

    - ``precision_hits`` = number of predicted tasks that actually
      flipped F→T.
    - ``precision_denom`` = attributed tasks (flipped + still_F +
      regressed + still_T) **plus** each round's ``regressed_unpredicted``
      count. Adding side-effects to the denominator is what keeps a
      round that flipped 3 predicted tasks but broke 5 unpredicted ones
      from registering as 100% precision.
    - Tasks whose attribution is ``absent`` (did not run in one of the
      two rounds) carry no signal and are excluded from both counts.

    The rendered ratio is ``precision_hits / precision_denom`` ∈ [0, 1];
    the ``side_effects`` column is also shown separately so the agent
    can see the absolute damage, not just the discounted ratio.

    Returns ``output_path`` on success, ``None`` when no journal
    exists or it has no parseable entries yet.
    """
    entries = read_entries(journal_path)
    if not entries:
        return None

    # Only consider entries strictly before ``current_round`` — the
    # current round hasn't happened yet.
    past = [e for e in entries if e.round < current_round]
    if not past:
        return None

    recent = past[-recent_window:]

    # Per-lever counts. Raw integer fields (attempts, accepted, …) are
    # the transparent per-round tallies. ``weighted_hits`` and
    # ``weighted_misses`` are time-decayed sums used to build the Beta
    # posterior — recent rounds count more so a lever that worked twelve
    # rounds ago doesn't drown out recent evidence.
    lever_counts: dict[str, dict[str, float]] = {
        lev: {
            "attempts": 0,
            "accepted": 0,
            "reverted": 0,
            "precision_hits": 0,
            "precision_total": 0,
            "side_effects": 0,
            "weighted_hits": 0.0,
            "weighted_misses": 0.0,
        }
        for lev in _VALID_LEVERS
    }
    # Attribution values that carry attribution signal (absent does not).
    _ATTRIBUTED = {"flipped", "still_F", "regressed", "still_T"}

    # Time-decay factor: weight by ``0.9 ^ (current_round - entry.round)``.
    # Recent rounds → weight ≈ 1.0; rounds 22+ ago drop below 0.1. This
    # lets the posterior-mean column drift as the benchmark evolves
    # without throwing away historical counts entirely.
    _DECAY = 0.9

    for e in past:
        outcome = e.gating_outcome
        weight = _DECAY ** max(0, current_round - e.round - 1)
        for lever in e.levers:
            if lever not in _VALID_LEVERS:
                continue
            lever_counts[lever]["attempts"] += 1
            if outcome == "accepted":
                lever_counts[lever]["accepted"] += 1
            elif outcome == "reverted":
                lever_counts[lever]["reverted"] += 1
            attr = e.gating_attribution
            predicted = e.predicted_affected
            round_hits = 0
            round_attributed = 0
            if attr and predicted:
                for tid in predicted:
                    outcome_v = attr.get(tid)
                    if outcome_v not in _ATTRIBUTED:
                        continue
                    lever_counts[lever]["precision_total"] += 1
                    round_attributed += 1
                    if outcome_v == "flipped":
                        lever_counts[lever]["precision_hits"] += 1
                        round_hits += 1
            extras = e.frontmatter.get("regressed_unpredicted")
            round_side = 0
            if isinstance(extras, list):
                round_side = sum(1 for v in extras if isinstance(v, str))
                lever_counts[lever]["side_effects"] += round_side
            # Weighted counts for the Beta posterior: hits count toward
            # numerator, misses (attributed_non_flip + side_effects)
            # count toward the denominator complement.
            round_misses = (round_attributed - round_hits) + round_side
            lever_counts[lever]["weighted_hits"] += weight * round_hits
            lever_counts[lever]["weighted_misses"] += weight * round_misses

    lines: list[str] = []
    lines.append(f"# Journal Context for R{current_round}")
    lines.append("")
    lines.append(
        f"Aggregated from `{journal_path}` "
        f"across {len(past)} prior round(s); recent window shows "
        f"the last {len(recent)}."
    )
    lines.append("")

    # Lever scoreboard.
    lines.append("## Lever scoreboard (all-time)")
    lines.append("")
    lines.append(
        "Raw prediction hits = ``flipped / (attributed_predicted + side_effects)``"
        " — a round that flipped 3 tasks but broke 5 others registers as 3/8,"
        " not 100%. The **Posterior** column smooths this with a Beta(1+wh, 1+wm)"
        " prior where wh/wm are time-decayed hits/misses (0.9^rounds_ago): low-n"
        " levers get pulled toward 0.5, recent rounds weigh more than ancient ones."
        " ``n_eff`` is the effective weighted sample size — when it's small the"
        " posterior is uncertain even if the raw ratio looks extreme."
    )
    lines.append("")
    lines.append("| Lever | Attempts | Accepted | Reverted | Raw hits | Posterior (n_eff) | Side-effects |")
    lines.append("|-------|---------:|---------:|---------:|---------:|:------------------|-------------:|")
    for lev in sorted(_VALID_LEVERS):
        c = lever_counts[lev]
        denom = c["precision_total"] + c["side_effects"]
        prec = f"{c['precision_hits']}/{denom}" if denom else "—"
        side = c["side_effects"] or "—"
        wh = c["weighted_hits"]
        wm = c["weighted_misses"]
        n_eff = wh + wm
        if n_eff > 0:
            # Beta(1+wh, 1+wm) posterior mean = (1+wh) / (2+wh+wm)
            post_mean = (1.0 + wh) / (2.0 + wh + wm)
            posterior_cell = f"{post_mean:.2f} (n_eff={n_eff:.1f})"
        else:
            posterior_cell = "—"
        lines.append(
            f"| {lev} | {int(c['attempts'])} | {int(c['accepted'])} | {int(c['reverted'])} | "
            f"{prec} | {posterior_cell} | {side} |"
        )
    lines.append("")

    # Recent hypotheses table.
    lines.append(f"## Recent hypotheses (last {len(recent)})")
    lines.append("")
    lines.append("| Round | Label | Levers | Outcome | Predicted | Attribution |")
    lines.append("|------:|-------|--------|:-------:|:---------:|-------------|")
    for e in recent:
        levers_str = ",".join(e.levers) or "?"
        outcome = e.gating_outcome
        pred = len(e.predicted_affected)
        attr = e.gating_attribution
        if attr:
            flipped = sum(1 for v in attr.values() if v == "flipped")
            attr_str = f"{flipped}/{len(attr)} flipped"
        else:
            attr_str = "pending"
        lines.append(f"| R{e.round} | {_truncate(e.label, 40)} | {levers_str} | {outcome} | {pred} | {attr_str} |")
    lines.append("")

    # Recent changesets — show WHAT each round changed at the config
    # level, independent of what the agent's prose claimed. Paired with
    # the attribution row above, this lets the next round correlate
    # "what I changed" with "what flipped".
    changeset_lines = _render_recent_changesets(recent)
    if changeset_lines:
        lines.append(f"## Recent changesets (last {len(recent)})")
        lines.append("")
        lines.append(
            "One line per round summarising the objective config diff "
            "(independent of the agent's prose). Empty bullet = the round "
            "produced no structural change (noop or pure prompt prose)."
        )
        lines.append("")
        lines.extend(changeset_lines)
        lines.append("")

    # Per-task history matrix — tasks the agent has historically
    # predicted and how they've moved across rounds. Stuck tasks and
    # flipping tasks (stability = 0) are what the next round should
    # study first.
    matrix_lines = _render_per_task_matrix(recent)
    if matrix_lines:
        lines.append(f"## Per-task history (across last {len(recent)} rounds)")
        lines.append("")
        lines.append(
            "T = passed, F = failed, - = task didn't run / wasn't predicted. "
            "Sorted by: first, how many rounds the task has been predicted but "
            "remains failing (stuck); then, instability (times the status flips "
            "within the window). Tasks that repeatedly appear in `predicted_affected` "
            "without flipping deserve the most scrutiny."
        )
        lines.append("")
        lines.extend(matrix_lines)
        lines.append("")

    # Reverted hypotheses to avoid repeating.
    reverted = [e for e in past if e.gating_outcome == "reverted"]
    if reverted:
        lines.append(f"## Reverted hypotheses — do not re-propose without new evidence ({len(reverted)})")
        lines.append("")
        for e in reverted:
            lines.append(f"- R{e.round} `{e.hypothesis_id}`: {_truncate(e.label, 80)}")
        lines.append("")

    lines.append("## Full journal")
    lines.append("")
    lines.append(
        f"For hypothesis bodies, evidence citations, and uncertainty "
        f"notes, read `{journal_path}` directly. This context file is "
        "an index, not a replacement."
    )
    lines.append("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ─── Attribution helper ───────────────────────────────────────────────────


def compute_attribution(
    predicted: list[str],
    passed_now: set[str],
    passed_before: set[str],
    *,
    appeared_now: set[str] | None = None,
    appeared_before: set[str] | None = None,
) -> dict[str, str]:
    """Classify each predicted task id into a gating-attribution outcome.

    Given:
    - ``predicted`` — task ids the journal entry said this round would
      help (``predicted_affected``)
    - ``passed_now`` / ``passed_before`` — task ids that passed in the
      this-round / prior-round evaluation
    - ``appeared_now`` / ``appeared_before`` — task ids that *ran at all*
      in each round (``passed`` is always a subset). When provided, a
      task that did not run in a round is classified as ``absent``
      rather than silently assumed failing. When ``None``, the appeared
      sets default to the passed sets (legacy callers get the old
      behaviour, except absent is now detectable from presence in
      ``passed_before | passed_now``).

    Returns ``{task_id: outcome}`` where outcome is one of:

    - ``flipped``   — was failing, now passing (hypothesis landed)
    - ``still_F``   — was failing, still failing (hypothesis missed)
    - ``regressed`` — was passing, now failing (hypothesis hurt)
    - ``still_T``   — was passing, still passing (no signal)
    - ``absent``    — task did not appear in at least one of the two
      rounds, so the round-to-round delta is undefined

    Task ids not in ``predicted`` are not returned — the attribution
    is scoped to what the agent claimed would improve, not the round
    as a whole.
    """
    result: dict[str, str] = {}
    # Strict mode: both appeared sets provided → detect ``absent`` tasks
    # and exclude them from lever precision. Legacy mode: when either
    # set is omitted, fall back to the old (passed_* only) behaviour —
    # a task predicted but not in either passed set is still_F, as the
    # old code assumed. This keeps downstream callers that don't know
    # the appeared sets working unchanged.
    strict = appeared_now is not None and appeared_before is not None

    for tid in predicted:
        if strict:
            appeared_n = tid in appeared_now  # type: ignore[operator]
            appeared_b = tid in appeared_before  # type: ignore[operator]
            if not appeared_n or not appeared_b:
                result[tid] = "absent"
                continue
        was_p = tid in passed_before
        now_p = tid in passed_now
        if was_p and now_p:
            result[tid] = "still_T"
        elif was_p and not now_p:
            result[tid] = "regressed"
        elif not was_p and now_p:
            result[tid] = "flipped"
        else:
            result[tid] = "still_F"
    return result


# ─── Helpers ──────────────────────────────────────────────────────────────


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


_CHANGESET_KEYS_ORDER = (
    ("tools_added", "+tools"),
    ("tools_removed", "-tools"),
    ("processors_added", "+processors"),
    ("processors_removed", "-processors"),
    ("processors_config_changed", "kwargs"),
    ("templates_added", "+templates"),
    ("templates_removed", "-templates"),
    ("templates_changed", "~templates"),
)


def _render_per_task_matrix(recent: list["JournalEntry"]) -> list[str]:
    """Render a per-task × per-round pass/fail matrix across ``recent``.

    Rows are task_ids that appeared in at least one round's
    ``gating_attribution``. Columns are the N recent rounds (in
    chronological order). Cells: ``T`` / ``F`` / ``-`` (absent).

    Rows are sorted:
    1. Descending by number of F cells within the window (most-stuck
       tasks first).
    2. Descending by number of status flips T↔F within the window
       (most-unstable tasks).
    3. Ascending by task_id (stable deterministic tie-break).
    """
    # status[task_id][round] = "T" | "F" | "-"
    status: dict[str, dict[int, str]] = {}
    _ATTR_TO_CELL = {
        "flipped": "T",
        "still_T": "T",
        "still_F": "F",
        "regressed": "F",
        "absent": "-",
    }
    for e in recent:
        attr = e.gating_attribution
        if not attr:
            continue
        for tid, outcome in attr.items():
            cell = _ATTR_TO_CELL.get(outcome, "-")
            status.setdefault(tid, {})[e.round] = cell

    if not status:
        return []

    rounds_in_window = [e.round for e in recent]

    def _sort_key(tid: str) -> tuple[int, int, str]:
        cells = [status[tid].get(r, "-") for r in rounds_in_window]
        f_count = sum(1 for c in cells if c == "F")
        flips = sum(1 for a, b in zip(cells, cells[1:]) if a != b and a != "-" and b != "-")
        return (-f_count, -flips, tid)

    out: list[str] = []
    header = "| Task | " + " | ".join(f"R{r}" for r in rounds_in_window) + " |"
    sep = "|------|" + "|".join([":--:"] * len(rounds_in_window)) + "|"
    out.append(header)
    out.append(sep)
    for tid in sorted(status, key=_sort_key):
        cells = [status[tid].get(r, "-") for r in rounds_in_window]
        truncated = _truncate(tid, 40)
        out.append(f"| `{truncated}` | " + " | ".join(cells) + " |")
    return out


def _render_recent_changesets(recent: list["JournalEntry"]) -> list[str]:
    """Return bullet lines summarising each recent round's changeset.

    Reads the ``changeset`` dict from each entry's frontmatter (populated
    by the orchestrator via ``extra_frontmatter={"changeset": diff}``).
    Falls back to an empty bullet for rounds with no recorded diff.
    """
    out: list[str] = []
    any_data = False
    for e in recent:
        cs = e.frontmatter.get("changeset")
        if not isinstance(cs, dict):
            cs = {}
        parts: list[str] = []
        for key, label in _CHANGESET_KEYS_ORDER:
            vals = cs.get(key)
            if not isinstance(vals, list) or not vals:
                continue
            # Render as e.g. "+tools: [<tool_name_1>, <tool_name_2>]"
            compact = [_truncate(str(v), 40) for v in vals[:4]]
            more = f", …+{len(vals) - 4}" if len(vals) > 4 else ""
            parts.append(f"{label}: [{', '.join(compact)}{more}]")
        if parts:
            any_data = True
        levers_str = ",".join(e.levers) or "?"
        summary = "; ".join(parts) if parts else "no structural change"
        out.append(f"- R{e.round} [{levers_str}] `{e.hypothesis_id}` — {summary}")
    return out if any_data else []


__all__ = [
    "JournalEntry",
    "JournalEntrySpec",
    "append_entry",
    "read_entries",
    "latest_entry",
    "fill_gating",
    "build_context",
    "compute_attribution",
]
