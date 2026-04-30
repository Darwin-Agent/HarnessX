---
name: journal
description: Cross-round journal format — the multi-round memory that tells the next meta-agent what's been tried, what landed, and what got reverted. One `## Round N` section per evolve, with machine-parseable YAML frontmatter + free prose body. Read at the start of every evolve to avoid re-discovering doomed hypotheses; append one new section before stopping.
---

# Journal — cross-round memory

A markdown file (the caller passes its path as `memo_path`). Each
evolve appends one `## Round N — <label>` section. The file is the
meta-agent's only memory across rounds — there is no persistent
conversation, and the orchestrator fills in gating outcomes between
rounds so the next agent can see which past bets paid off.

The orchestrator pre-renders a machine index at
`_meta_scratch/CONTEXT.md` (when the journal has prior entries):
lever scoreboard, recent hypotheses, reverted bets. Read that
first — it's a cheap way to see what's been tried without scanning
the full journal prose.

## On entry — read once

1. `Read <memo_path>` — full cross-round history, prose included.
2. `Read _meta_scratch/CONTEXT.md` (when present) — compact index of
   lever usage, attribution, and reverted hypotheses.

Look for:
- Hypotheses with `gating_outcome: reverted` → do not re-propose
  without new evidence (hard invariant #4 variant).
- The lever scoreboard's prediction hit rate → if Action landed
  1/12 times while Instruction landed 4/6, update your prior.
- Outstanding `needs_from_human` items (free-prose; scan for them).

## On exit — append one section

Required shape:

```markdown
## Round N — <short label in 2-5 words>

<!-- journal:frontmatter
round: N
timestamp: 2026-04-27T15:30:00Z    # ISO-8601 UTC; orchestrator can override
hypothesis_id: h_<short slug>      # stable across related rounds
levers: [action]                   # subset of {configuration, control, action, instruction}
predicted_affected: [task_a, task_b]
gating_outcome: pending             # orchestrator back-fills next round
gating_attribution: pending         # orchestrator back-fills next round
expected_global_gain: "<one-line cluster-level upside>"
regression_risk: "<main risk outside predicted_affected>"
cost_shift: "<expected token/cost delta or direction>"
rollback_trigger: "<observable signal to revert next round>"
-->

### Why

One paragraph on the pattern that motivated the round — the
cluster of trajectories and the shape they share.

### Changes

- `tools/<name>.py` — new @tool, one-line signature description
- `config.yaml` — register above under `tools.custom`

### Evidence

- `task_a` frontmatter: `judge_missing_capability.summary: "<gap phrase>"`
- `task_a` step <n> body: `<40-char quote showing the failure shape>`
- `task_b` exit_reason=max_steps, <n>/<max_steps> spent on the same
  pattern

### Uncertainty

One-liner on what might go wrong and how you'd know.
```

## Frontmatter keys — all required

- **`round`** (int) — monotonic round index.
- **`timestamp`** (ISO-8601 UTC string) — when appended.
- **`hypothesis_id`** (string slug) — short, stable across related
  rounds. Prefer semantic ids that name the intervention shape
  (e.g. `h_<cluster>_v1`, `h_<tool_name>_retry_v2`) over
  auto-generated ones. If a later round extends the same
  hypothesis, reuse the same id with a version suffix.
- **`levers`** (list) — subset of
  `{configuration, control, action, instruction}`.
- **`predicted_affected`** (list of task_ids) — the task ids you
  claim this round's change should flip from F to T (for
  corrective / preservative-transfer candidates) or protect from
  T→F regression (for preservative-lock candidates). Lists with
  zero entries are legitimate when the claim is about cost, not
  pass-rate. Preservative-lock candidates can list their
  currently-passing cluster here to make the "don't regress these"
  claim explicit. Over-predicting hurts your attribution score;
  the orchestrator grades this.
- **`gating_outcome`** (string) — `pending` on first write. The
  orchestrator fills with `accepted` / `reverted` / `noop` after
  the benchmark round runs.
- **`gating_attribution`** (dict | string) — `pending` on first
  write. The orchestrator fills with
  `{task_id: flipped|still_F|regressed|still_T}` after evaluation.
- **`expected_global_gain`** (string) — one-line claim of global
  upside (cluster-level, not single-task anecdote). For
  corrective / preservative-transfer candidates: "flips N tasks
  in cluster X". For preservative-lock candidates: "prevents
  expected regression of N tasks if future rounds drift" — this
  is a legitimate gain even with zero flips.
- **`regression_risk`** (string) — primary collateral risk outside
  `predicted_affected`.
- **`cost_shift`** (string) — expected cost/tokens movement if this
  change lands.
- **`rollback_trigger`** (string) — concrete signal that means "revert
  this hypothesis" next round.

## Frontmatter keys — required conditionally

- **`cited_candidates`** (list of strings) — **required when the
  round's config changes** (the orchestrator's evidence gate
  enforces it). Lists the `C-NNN` IDs from `_meta_scratch/candidates.md`
  that motivated this round's changes. Example:
  `cited_candidates: [C-001, C-004]`. See the `analyze` skill for
  the `candidates.md` format. Omit entirely on noop rounds (byte-
  identical config copy) — they don't need evidence.
- **`retry_rationale`** (string) — required only if your
  `(levers, predicted_affected)` signature matches a reverted round.
  One-line explanation of what evidence is new since the last try,
  e.g. `retry_rationale: "R3 tried instruction-only; this round
  found tool_error_counts spike for WebFetch that the prior attempt
  ignored"`. Skip when your signature is novel. See the novelty
  findings file (`_meta_scratch/NOVELTY_FAIL.md`) if the gate
  rejects your round — it names the colliding prior round.

## Prose body — four suggested headings, not enforced

The parser ignores the prose. These headings exist for consistency:

- **`### Why`** — the cluster of trajectories and the pattern
  they share. For corrective rounds: the failure shape. For
  preservative-lock rounds: the habit present in the passing
  cluster that would be at risk without the encoding. For
  preservative-transfer rounds: the habit fired on the passing
  cluster + absent on the failing cluster.
- **`### Changes`** — bulleted list, one bullet per authored file
  or config edit.
- **`### Evidence`** — quoted frontmatter or body excerpts with
  `task_id` citations. Hand-wavy "several tasks did X" is not
  evidence. Preservative-transfer rounds must cite both ends
  (passing cluster body + failing cluster body), consistent with
  `candidates.md`.
- **`### Uncertainty`** — what might fail and how you'd tell.

Include the same four signals in prose when stakes are high:
`expected_global_gain`, `regression_risk`, `cost_shift`,
`rollback_trigger`. This makes Pareto-style tradeoffs explicit across
rounds instead of implicit in vague narratives.

Write more when the round is high-stakes, less when it's a small
config nudge. Don't abbreviate the frontmatter — that's what the
orchestrator reads.

## Rules

- **Append, never edit prior sections.** The orchestrator DOES edit
  prior sections — it back-fills `gating_outcome` and
  `gating_attribution` surgically without touching prose. You, as
  the agent, never rewrite an old round.
- **Cite task ids + field or body quote** in Evidence. Vague is bad.
- **Do not re-propose reverted hypotheses** without a specific
  reason the prior attempt was buggy (not "the idea was right but
  my implementation wasn't" — come back with a different shape, a
  different lever, or more evidence).
- **Multi-lever rounds are allowed** when the evidence warrants
  (see `analyze` skill → *Multi-lever rounds*). List each change
  under its own bullet in Changes and cite one candidate ID per
  change in `cited_candidates`. Attribution is per-round, so
  bundling *does* widen the ambiguity when gating — that's a
  reason to keep each change independently supported (own
  candidate, own cluster, own retroactive-check variant), not a
  reason to drop to a single change per round.

## Legacy compat

Entries without the `<!-- journal:frontmatter ... -->` block are
skipped by the machine parser but still render as prose for readers.
Early rounds that predate the journal format will be invisible in
`CONTEXT.md`'s scoreboard — that's fine, new rounds start fresh.

## Example minimal append

Small config tune, one gap, one lever. Because this changes the
config (CostGuardProcessor kwargs), the entry must cite a candidate
from `_meta_scratch/candidates.md`:

```markdown
## Round 3 — loosen cost guard

<!-- journal:frontmatter
round: 3
timestamp: 2026-04-27T10:12:00Z
hypothesis_id: h_cost_guard_v1
levers: [configuration]
predicted_affected: [task_42]
cited_candidates: [C-001]
gating_outcome: pending
gating_attribution: pending
expected_global_gain: "Should reduce budget_exceeded cluster on long-horizon tasks"
regression_risk: "May increase cost on easy tasks that already pass"
cost_shift: "+5-10% tokens expected on median run"
rollback_trigger: "If pass_rate is flat/down and cost rises >10%, revert"
-->

### Why

task_42 hit budget_exceeded at step 18/30 while still productively
calling tools. CostGuardProcessor max_usd=0.5 is too tight for
Level-3 tasks that require >20 WebFetch calls.

### Changes

- `config.yaml` — CostGuardProcessor max_usd 0.5 → 1.0

### Evidence

- `task_42` frontmatter: `exit_reason: budget_exceeded`, cost_usd: 0.51
- `task_42` step 18 body: still making progress ("Found the relevant
  Wikipedia section — let me fetch one more to confirm").

### Uncertainty

Looser cap may push the per-round bill up ~10%. If R4 pass_rate
drops while cost climbs, revert.
```
