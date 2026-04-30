---
name: analyze
description: How to read trajectories and extract useful experience — from successes and failures alike — into candidates for the 4 HarnessConfig levers. Covers the trajectory frontmatter schema (behaviour / eval / judge), the two axes of reflection (lens × lever), the three-variant retroactive check, and when to delegate batch reading to spawn_reflect_worker. Read at Step 3 of the loop.
---

# Analyze — reading trajectories, extracting experience

Your input is a directory of `<task_id>.md` files, one per task in
this round. No pre-computed summaries, no cluster files — you
investigate the markdown directly. Your output is a set of candidate
hypotheses, each naming a concrete **pattern** (either a gap to
close or a habit to preserve / generalize) and the lever you'd pull.

Passing and failing trajectories are treated as equal evidence
sources. A passing cluster can teach you a habit worth locking in
or worth lifting into the template for the failing cluster; a
failing cluster can teach you a blocker worth closing. Both feed
the same candidate pipeline — same template, same retroactive
check, same global-first ordering. The bias to read only failures
is the single biggest cause of "right diagnosis, still-net-
negative round": fixes land but previously-working habits silently
regress because no candidate preserved them.

## Trajectory file shape

YAML frontmatter + markdown body. `Read <path> limit=30` returns the
frontmatter plus the first few body lines — usually enough to
classify the task. When the frontmatter hints at a deeper cause,
read the last ~40 lines of the body to see what the agent actually
had in context at its decisive step.

### Frontmatter tiers (render order)

**Behaviour — always present**

- `exit_reason` — `done` / `max_steps` / `budget_exceeded` /
  `loop_detected` / `error`
- `steps`, `total_tokens`, `cost_usd`, `elapsed_s`
- `tool_call_counts` — `{tool_name: n}`
- `tool_error_counts` — `{tool_name: n}`
- `error_count`, `pivotal_tool`
- `final_output_length`

**Eval — always present, authoritative**

- `eval_passed` (bool) — the external pipeline evaluator's verdict
- `eval_score` (float) — numeric score (0.0 / 1.0 for exact-match)

`eval_*` is the ground-truth *correctness signal*. When Eval and
Judge disagree, believe Eval. The dataset's expected answer text
is intentionally withheld from the frontmatter, and so is the
evaluator's textual reason — you see only pass/fail and score,
never the target string. Evolve on correctness signals, not on a
known target. For the *why* behind a failure, read the
trajectory body and the optional `judge_*` fields below.

**Judge — optional**

Present only when `LLMJudgeProcessor` ran. An LLM's opinion, not
ground truth. Useful for *why* a task failed, not *whether*.

- `judge_verdict` — `plausible` / `unsupported` / `hedging` /
  `format_wrong` / `refused` / `no_answer` / `judge_error`
- `judge_cause`, `judge_missing`, `judge_lesson` — one-line strings
- `judge_missing_capability` — `{present: bool, summary: "..."}`;
  primary signal for Action-lever decisions
- `extracted_answer` — what the agent actually committed (the
  expected answer text is never exposed; do not try to reconstruct
  it)

`judge_error` is the judge failing (timeout, bad JSON), not the
agent. Treat as "no opinion" and lean on behaviour + body.

## The two axes of reflection

Every candidate hypothesis is pinned on two axes. Keep them
independent — don't collapse the lens into the lever.

### Axis A — Lens (what is the pattern?)

Three reading passes over the same batch, not three buckets. A
single trajectory can answer more than one question — you are
sweeping for signals, not assigning labels. Every pass anchors on
the authoritative `eval_passed` field; behaviour / judge signals
describe the *shape*, never the verdict.

| Lens | Question | Signals (primary first) |
|---|---|---|
| **Failure** | What blocker kept this cluster from finishing? | `eval_passed=False` identifies the cluster; then read `exit_reason ∈ {budget_exceeded, error, loop_detected}`, high `tool_error_counts`, `judge_verdict ∈ {unsupported, hedging, format_wrong, refused, no_answer}` for the failure shape |
| **Capability gap** | What did the agent try but couldn't do? | Non-empty `judge_missing_capability` / `judge_missing`; sentinel strings in tool output (`[SEARCH UNAVAILABLE]`, `ModuleNotFoundError`, redirect-HTML, raw bytes). Cuts across `eval_passed` — the agent sometimes works around the wall and still passes; don't pre-filter by Failure |
| **Success** | What habit made this cluster finish cleanly? | `eval_passed=True` identifies the cluster; then `exit_reason=done` with short step count relative to the task class, consistent tool sequence across the passing cluster, non-empty `judge_lesson` for transferable insight |

`exit_reason=done` alone is not evidence of success — the agent
merely called `end_turn`. Always gate on `eval_passed` before
reading a trajectory as a positive example.

The three lenses feed the **same** candidate pipeline. A Success-
lens observation turns into a candidate in one of two shapes:
- **Lock-in** — encode the habit as Configuration / Control /
  Instruction so a future round's edit cannot silently regress
  it. Justification: the cited passing tasks would fail if the
  habit were removed.
- **Transfer** — lift the habit into a rule applied to a failing
  cluster. Justification: the cited failing tasks would pass if
  the habit were applied.

Failure and Capability-gap lenses produce **corrective**
candidates (close a gap). Both polarities go through the
same retroactive check, same global-first ordering, same
`candidates.md` schema. What changes is which variant of the
retroactive check the candidate has to answer.

Rushing to Failure and skipping the other two leaves Capability-gap
bets (blocked-but-sometimes-recoverable shapes) and Success-pattern
lessons (what to preserve or generalise) on the table.

### Axis B — Lever (where does the fix belong?)

| Lever | What lives here | Pick it when |
|---|---|---|
| **Configuration** | Kwargs on existing processors | An existing component's tuning is off (loop-detection too tight, sliding window too small, a builtin's `max_usd` too stingy) |
| **Control** | New `MultiHookProcessor` — mechanical hooks around the loop | A mechanical step misfires or is missing: tool output needs parsing, input needs sanitising, the same guard needs to fire across tasks, final output needs reformatting |
| **Action** | New `@tool` — expands the agent's action space | The agent has no way to take a class of action; or an existing tool fires but returns data the caller has no reasonable way to use |
| **Instruction** | Jinja template edit, SOUL.md edit, or a new skill file | Capability and control are fine, but the agent does not know *when / in what order / under what condition* to use them |

The axes are orthogonal. A Failure symptom can be fixed at any of
the four levers. A Capability-gap lens does not always mean a new
tool — if the existing tool fires but returns unusable data, the
lever is often Control (an `on_after_tool` processor that parses),
not Action.

### Worked examples — lens × lever × intent

The three axes cover 3 × 4 × 3 combinations; most are rare. Below
are representative shapes you'll actually see. Use them to
pressure-test your own candidate — if yours doesn't resemble any
row here, either you've found something genuinely new or the lens
/ lever / intent tag is wrong.

| Lens | Lever | Intent | Shape |
|---|---|---|---|
| Failure | Configuration | corrective | `exit_reason=budget_exceeded` recurs while agent still making progress → raise `CostGuardProcessor.max_usd`. |
| Failure | Control | corrective | Tool returns raw bytes on PDF tasks; agent paraphrases and fails → `on_after_tool` processor that pipes bytes through a text extractor. |
| Failure | Instruction | corrective | Reasoning correct but `judge_verdict=format_wrong` across tasks → add explicit `FINAL ANSWER:` rule to the template. |
| Capability-gap | Action | corrective | `judge_missing_capability.summary` names "JS-rendered browser" on 3+ tasks; `WebFetch` returns redirect HTML → new `browser(...)` tool. |
| Capability-gap | Control | corrective | Tool *does* fetch the data but returns it in an unusable shape (binary / redirect chain) → normalising `on_after_tool` hook, not a parallel fetch tool. |
| Success | Instruction | preservative-lock | Passing cluster all emit `## Plan` block; if removed the habit evaporates under a future prompt edit → encode "emit plan block before first tool call" as an explicit template rule. |
| Success | Instruction | preservative-transfer | Passing cluster plans, failing cluster thrashes → lift the plan-block rule into the template applied to all tasks. |
| Success | Control | preservative-lock | Passing cluster consistently retries 3× on 5xx; removing the retry loop would lose these passes → harden with a `tool_failure_guard`-style retry policy. |

Rows omitted on purpose: `Capability-gap / Instruction` is almost
always wrong (if the capability isn't there, no prompt rule
conjures it — covered in "Action vs Instruction" below); `Success
/ Configuration` is rare because knob values don't encode a habit;
`Success / Action` is rare because authoring a new tool is not a
natural way to preserve a habit the existing tools already enable.

If your candidate lands in an omitted cell, that's a signal to
challenge the lens / lever / intent tag before the retroactive
check.

## Read strategy

Your output is labelled clusters with cited evidence, not a fixed
procedure run. The trajectories live under `_meta_scratch/traj/`
as `<task_id>.md` files. Available tools: `Read` / `Grep` for
direct inspection, `spawn_reflect_worker(kind="trajectory-
digester", files=[...])` for batching when the task count would
blow the token budget from `TASK.md`. Pacing — how many
frontmatters to scan before clustering, how many representatives
to body-read per cluster, when delegation is worth it — is your
judgement call. Record the reasoning in `candidates.md`.

Two constraints are non-negotiable:

- **Systemic vs idiosyncratic.** A cluster (failing OR passing)
  earns a config change only when the same shape recurs across
  at least two distinct tasks with different inputs. One-off
  observations — however obvious the fix or however clean the
  habit feels — go to `NEEDS_FROM_HUMAN` and are skipped. Two
  tasks that look different but share the same root cause still
  count as one "gap" (test: if you fix one, does the other
  follow?); two tasks with coincidentally similar frontmatter but
  different mechanisms do not count as a cluster.
- **Frontmatter shows the ceiling shape; the body shows what
  actually happened at the decisive step.** Don't propose Action
  or Control on frontmatter signals alone — for failure
  candidates a body must name the actual failure mode (raw bytes,
  empty results, redirect HTML, cycling URL, silent parse
  failure); for success candidates a body must show the habit
  actually firing (the `## Plan` block, the retry loop, the tool
  sequence) rather than just inferring it from short step count.
  A candidate whose evidence cites only a worker digest, with no
  trajectory body quoted, is not grounded.

## The retroactive check — three variants, one per candidate intent

Before writing any code, for each candidate, ask the variant that
matches the candidate's **intent**. A candidate has exactly one
intent; one that seems to need more than one variant is two
candidates glued together — split them.

### Variant A — Corrective (Failure / Capability-gap lens)

> If this fix had already been in place on the cited failing
> trajectories, would the task have succeeded?

- A commit-nudge requires an answer to commit.
- A template rule requires data to apply.
- A tighter loop detector requires the loop to be the actual
  blocker, not the consequence of the real blocker.
- A new tool requires the task to have needed that tool's output
  shape, not just the tool's topic.

When the answer is "no — the agent would still have failed", the
symptom is downstream of an earlier failure. Trace one step back:
what did the agent *lack*, not what did it fail to *do*? That
upstream gap is the real lever. Drop or re-scope.

### Variant B — Preservative, lock-in (Success lens)

> If this pattern were **removed** from the current config, would
> the cited passing trajectories have failed?

- A habit that's already reliable "by accident" of the current
  prompt / tool / model combo is not worth encoding — it costs
  complexity and pays nothing. Drop.
- A habit that was visibly load-bearing (the body shows the
  failing task in an adjacent cluster *not* doing this habit
  and breaking) earns a lock-in candidate — typically Instruction
  (promote the habit to an explicit rule) or Control (a guard
  that enforces it). Regression-probe candidates live here.

### Variant C — Preservative, transfer (Success → failing cluster)

> If this pattern had been **applied** to the cited failing
> trajectories, would they have passed?

- The generalization must pass both ends: the passing cluster's
  body must show the habit firing, AND the failing cluster's
  body must show the habit *absent at a moment where applying
  it would have unblocked the decisive step*.
- If only the first end is grounded, the habit is domain-specific
  — you're about to force a passing-cluster quirk onto tasks it
  won't help. Drop or re-scope.

### Why the variants matter

The single-question form ("would the fix have worked") silently
excludes Success-lens candidates — they can't "work" on the tasks
they came from, because those tasks already passed. Without
Variant B/C, Success-lens evidence has no path into
`candidates.md`, and the round ships fixes while quietly
regressing what was already working.

## Disambiguating adjacent levers

The Axis B table tells you what each lever *is*; the sections
below help when two lever choices could explain the same evidence.

### The first question — tool layer or caller?

Before disambiguating adjacent levers, ask where the blame sits:
on the **tool layer's return** or on the **caller's use of the
return**. This single split collapses most lever choices.

**Points at the tool layer** (Action or Control territory —
weigh together, no single signal decisive):
- `tool_error_counts[T]` recurs across distinct tasks on tool `T`.
- `judge_missing` / `judge_missing_capability` keeps naming the
  same shape (missing client, absent parser, no fallback).
- An existing tool fires successfully but returns data the caller
  has no reasonable way to use (raw bytes, truncated page,
  aggregated when itemized is needed, unparseable response).

**Points at the caller** (Instruction or Configuration territory):
- Tool returned correct data; agent misread, didn't reach for it,
  or committed before using it.
- Failure reads as budget / commitment / loop behaviour rather
  than data availability.

Once you know which side is to blame, the two sections below
disambiguate within that side. Configuration vs Instruction rarely
cross-contaminate and don't need a section; they split naturally
(existing-knob tune vs prompt-level rule).

**Perceived difficulty gradient isn't real.** Template < processor
< tool feels true but once you've written one of each, all three
are "write one file, wire it into config, verify". Pick by
evidence, not by ease.

### Action vs Instruction — when the gap is capability, not knowledge

Signals that a missing **capability** (not a missing prompt rule)
is the gap:

- `judge_missing_capability.summary` keeps naming the same shape
  across distinct tasks.
- Recurring cross-task `tool_error_counts` on the same tool.
- An existing tool fires but returns data the caller has no
  reasonable way to use (raw bytes, truncated page, unparseable
  response).
- Your own failure-bucket phrasing becomes "can't access X",
  "can't parse Y", "doesn't reach Z" for a recurring input shape.

In this regime, reaching for another template rule is usually
wrong — instruction patches around a tool that returns nothing
useful tend to compound into more failure. Prefer Action. For the
authoring mechanics (`TOOL_SPEC.md`, `@tool` signature, scoping a
tool to a class of tasks) see the `reference` skill — come back
here for lever choice, go there for how to build it.

Counter-case — when Instruction is still right:

- Evidence shows the capability is actually present (the answer was
  in a tool result the agent ignored, or judge is confused).
- Failure reads as commitment / format / timing, not data
  availability (e.g. `judge_verdict=format_wrong` on tasks whose
  reasoning was otherwise correct).

Document *why not Action* in the journal so the next round can
reuse the reasoning.

### Control vs Action — when post-processing beats a new tool

Tool fires and returns correct data, but in a shape the caller
can't use directly (binary bytes instead of parsed text, raw HTML
after a redirect chain, unstructured JSON blob). An
`on_after_tool` processor that normalises the return is usually
stronger than a new tool that does the same fetch-and-parse —
duplicating the fetch loses the existing tool's robustness and
budget accounting.

Control also covers input sanitisation (`on_before_tool`) and
cross-task guards (`on_step_start` / `on_before_model`) that must
fire uniformly across every task — shapes a per-call tool can't
express at all.

Counter-case — when Control isn't enough: if the tool's fetch
itself is wrong (wrong endpoint, missing auth, truncates before
returning), no post-processor recovers what was never retrieved.
Quick test: *can the information the agent needs be reconstructed
from what the tool currently returns?* Yes → Control. No → Action.

## When a prior round failed the replay gate

If the last round was rejected by the replay gate
(`_meta_scratch/REPLAY.md` exists with `ok: false`), treat it as
evidence not punishment. The JSON lists which tasks tripped (exit
reason, first failing step, brief traceback). Before re-proposing
anything:

- If the failure was structural (`exit_reason=error`, tool
  ImportError, Jinja render error) — the previous round's code
  was wrong, not its reasoning. The *hypothesis* may still be
  sound; re-scope the candidate and fix the implementation.
- If the failure was semantic (candidate ran cleanly but didn't
  flip the target task) — the retroactive check returned `yes`
  on paper but the real run disagreed. Treat the retroactive
  check as having failed retrospectively; don't re-ship the
  same shape without new body evidence.
- If the failure was a regression on a previously-passing task
  you didn't predict — that's a preservative-lock signal the
  prior round missed. Consider a Variant-B candidate next round
  to lock in whatever got disrupted.

The replay gate report is the highest-quality signal in the
journal — it's from an actual run, not a judge opinion.

## Cross-round signals

The journal CONTEXT.md renders a lever scoreboard. Use it as a
prior on what's worth trying, not as a rule:

- A lever that has landed consistently is worth revisiting.
- A lever that has been tried repeatedly on the same cluster
  without flipping any task is a signal to look elsewhere, not a
  reason to try again harder.
- A reverted `hypothesis_id` cannot be re-proposed in the same
  shape without new evidence; different shape at a different lever
  is fair game.

## Candidate ordering: global first, local second

Before choosing what to ship, rank candidates by global net value,
not by the most dramatic local anecdote.

Apply this ordering:

1. **Global gain first**: prefer candidates that plausibly flip a
   recurring failing cluster (>=2 tasks) or that prevent a
   likely regression on a currently-passing cluster of similar
   size. A preservative-lock candidate with a plausible 2-task
   regression saved beats a corrective candidate with a 1-task
   speculative flip.
2. **Regression surface second**: demote candidates with
   meaningful risk to already-passing or unrelated clusters
   unless evidence is unusually strong. (Preservative-lock
   candidates are specifically *reducing* this risk — they
   should be weighted accordingly, not dismissed as "no flips".)
3. **Cost shift third**: among similar gain/risk profiles, prefer
   lower expected token/cost inflation.

Every shipped candidate should make the tradeoff explicit:
- `expected_global_gain`
- `regression_risk`
- `cost_shift`

If a candidate helps one target task but likely harms global pass-rate,
drop it or re-scope it.

## Multi-lever rounds

A round can ship several independent changes when the evidence
warrants — one per distinct pattern (gap or habit). "Independent"
means each change works on its own; no same-round dependency
chains. Ship the number the evidence supports, not a target.
Breadth without focus regresses pass_rate; depth without breadth
misses cheap wins. A good round typically pairs 1-2 corrective
candidates (closing gaps) with 0-1 preservative candidates
(locking in or generalizing habits) — the exact mix is evidence-
driven, not a quota.

## `_meta_scratch/candidates.md` — required when the config changes

Before writing the config, draft your candidates in
``_meta_scratch/candidates.md``. Each candidate is a section with a
stable **ID** (``C-001``, ``C-002``, …) that the journal entry's
``cited_candidates`` frontmatter will reference.

**Header format** — the orchestrator parses `^## Candidate C-\d+`
headers, so every section header must match this shape. Anything
else (numbered bullets, "Candidate 1", etc.) is invisible to the
evidence gate and the round is rejected.

**Required fields per candidate** — each candidate must include all
of the fields below. A candidate missing any of them is treated as
under-grounded; the lever choice will be challenged and the round
may be rejected at review even if the header format parses.

| Field | Why it's required |
|---|---|
| `[lens: ... \| lever: ... \| intent: ...]` tag | Pins the candidate on all three axes so the journal's scoreboard can attribute outcomes. `intent ∈ {corrective, preservative-lock, preservative-transfer}`; it selects which retroactive-check variant the candidate answers. |
| One-line description | Forces the change to fit in one sentence; unclusterable changes are usually two candidates glued together. Phrase it as the encoding, not the observation (e.g. "promote `## Plan` block to an explicit template rule", not "passing tasks wrote `## Plan` blocks"). |
| `Signal` | Names the concrete frontmatter field or log shape that identifies the cluster. |
| `Verified` (body-quoted) | At least one body excerpt per cited task. Frontmatter alone doesn't satisfy this — structural levers need the body, not the ceiling. For preservative-transfer candidates, body excerpts are required from **both ends**: the passing cluster (habit firing) and the failing cluster (habit absent at the decisive step). |
| **`Why <chosen lever> not <adjacent lever>`** | Required on every candidate. Action candidates must argue against Instruction; Control against Action; Instruction against Action (or Control when post-processing was viable). Configuration candidates must argue why a bigger lever wasn't warranted. Preservative candidates still owe this: locking in a habit as Instruction vs. Control is a real choice. |
| **`Retroactive check`** | `yes` / `no` + a one-line justification against the variant selected by `intent` (A / B / C — see the retroactive-check section). A `no` means the cited evidence is downstream of the real pattern — either re-scope the candidate or drop it. |
| `Tasks affected` | For corrective candidates: >=2 distinct failing task_ids with the same mechanism. For preservative-lock: >=2 distinct passing task_ids where the habit fired. For preservative-transfer: >=2 passing + >=2 failing task_ids (two distinct ends). Single-task observations hit the idiosyncratic filter and belong in `NEEDS_FROM_HUMAN.md`. |
| `expected_global_gain` / `regression_risk` / `cost_shift` | Forces Pareto-style global reasoning: do not ship a candidate that is locally good but globally harmful. For preservative-lock, `expected_global_gain` can legitimately be "0 flips; prevents N expected regressions" — that's still a positive Pareto move when regression risk elsewhere is material. |

```markdown
# Candidates

## Candidate C-001
[lens: capability-gap | lever: control | intent: corrective]

One-line fix description.

- Tasks affected: task-011, task-042, task-108
- Signal: `judge_missing` names output-parsing capability on all three
  tasks; `tool_error_counts[fetch_tool]=0` (fetch is fine — shape is
  not).
- Verified (Read): task-011 step 7 — tool returned raw unstructured
  output, agent described content vaguely. task-042 step
  5 — same shape. task-108 step 9 — same shape.
- Why Control not Action: the tool already fetches the data —
  the gap is processing, not capability. A parallel fetch
  tool would duplicate the existing tool.
- Retroactive check (A-corrective): yes — if parsed output had been
  in context at the decisive step, the agent had enough to answer.

## Candidate C-002
[lens: failure | lever: instruction | intent: corrective]

One-line fix description.

- Tasks affected: task-017, task-033
- Signal: `judge_verdict=format_wrong` on both; reasoning in the
  body is otherwise correct.
- Verified: task-017 step 15 body ends mid-paragraph with no
  answer marker. task-033 step 12 — same shape.
- Why Instruction not Control: the reasoning was correct — the
  agent just didn't know to emit the marker. A mechanical
  post-hook can't extract the answer from narrative prose, and
  there's no tool output to post-process.
- Retroactive check (A-corrective): yes — an explicit format rule
  produces the correct commit path.

## Candidate C-003
[lens: success | lever: instruction | intent: preservative-transfer]

Promote the "emit a `## Plan` block before the first tool call"
habit from an implicit pattern to an explicit template rule.

- Tasks affected:
  - passing (habit fired): task-024, task-055, task-071
  - failing (habit absent at decisive step): task-038, task-093
- Signal: all three passing tasks write a `## Plan` block in step
  1-2 and keep tool calls under 6 (short `tool_call_counts`); the
  two failing tasks skip the plan, fan out to 14-18 tool calls,
  and `exit_reason=budget_exceeded`.
- Verified: task-024 step 1 body quotes the `## Plan` block;
  task-055 step 2 — same shape; task-071 step 1 — same shape.
  task-038 steps 1-3 — no planning block, tool calls thrash on
  adjacent queries; task-093 step 1 — same shape.
- Why Instruction not Control: a Control hook would inject the
  plan mechanically, bypassing the agent's own scoping — the
  planning *content* is what made the passing cluster short, not
  the presence of a header string. An Instruction rule keeps
  the scoping agent-side.
- Retroactive check (C-preservative-transfer): yes — task-038 and
  task-093's thrashing begins before step 4; a planning rule
  applied up-front would have forced query scoping and kept the
  step count under budget.
```

### The "Why X not Y" field is the main failure mode to avoid

A candidate that skips this field almost always picks the
lowest-friction lever (usually Instruction — "add a rule to the
prompt") on evidence that points at a heavier lever (Action /
Control). Writing the "Why X not Y" line forces you to argue
against the adjacent alternative, not just in favour of the one
you picked. Examples of the mistake this catches:

- Capability-gap signal, Instruction lever, no "Why not Action"
  → usually means the agent has no way to do the thing, but the
  author defaulted to a prompt patch because it was easier to
  write. Re-pick as Action unless the counter-case in "Action vs
  Instruction" applies *and you can write it out*.
- Control-shaped signal (tool returns unusable data), Action
  lever, no "Why not Control" → usually means an `on_after_tool`
  normaliser would have been the narrower fix.
- Success habit, Instruction lever, no "Why not Control" → a
  Control hook mechanically injecting the habit is tempting
  ("always write a `## Plan` block") but bypasses the agent's
  own scoping; argue why the habit needs to stay agent-authored.
  Or the opposite: a Control lever with no "Why not Instruction"
  → you're hard-coding a habit the agent could have learned to
  apply selectively.

If you can't write the "Why X not Y" in one sentence, the lever
choice isn't ready and the candidate needs more investigation.

Concrete, benchmark-specific examples (e.g. answer-format markers,
role-play protocols, verifier scripts, domain-specific tool specs)
live in the `<bench>-playbook` skill, not here — this skill is
benchmark-agnostic.

Rank candidates by ``evidence strength × expected leverage``. Ship
the ones you actually implement — a candidate that isn't implemented
shouldn't appear in ``cited_candidates`` on the journal entry.

### Cross-reference in the journal

When this round's config changes, the latest journal entry's
frontmatter must cite at least one of the candidate IDs:

```yaml
cited_candidates: [C-001, C-002]
```

See ``journal`` skill for the full frontmatter schema. The evidence
gate rejects any non-noop round where:

- ``candidates.md`` is missing, OR
- it has no ``## Candidate C-N`` sections, OR
- ``cited_candidates`` is empty or missing, OR
- a cited ID doesn't have a section in ``candidates.md``.

Noop rounds (``cp current_config output/config.yaml``) don't need
``candidates.md`` — the orchestrator only enforces evidence when the
structural diff is non-empty.

## Pitfalls

- **Jumping to an encoding before the retroactive check.** The
  first plausible story is often downstream of the real pattern
  — whether you're closing a gap (Variant A) or transferring a
  habit (Variant C).
- **Reading only failures.** Shipping fixes while the successes
  that made the baseline work silently regress is the most
  common way a round lands net-negative. Sweep passing clusters
  with the same seriousness as failing ones.
- **Confusing judge signal with eval signal.** The judge is an
  opinion; when `eval_passed` disagrees, believe `eval_passed`.
- **Single-task optimisation.** Idiosyncratic observations —
  failing OR passing — shouldn't shape the config. Log them
  under `NEEDS_FROM_HUMAN` in the journal and move on.
- **Generic tool replacement.** New tools should close a specific
  class; a "better WebFetch" competes with WebFetch in the general
  case and loses. Scope to the cluster.
- **Layer-by-lens reflex.** "Failure → prompt fix", "capability gap
  → new tool", "success → new processor". All three are often
  wrong. Always re-check the lever against the evidence.
- **Encoding a habit that was already free.** A preservative-lock
  candidate whose Variant-B check returns `no` (the passing tasks
  would still pass with the habit removed) is dead weight — it
  adds config surface area without paying for it. Drop, don't
  ship "to be safe".
