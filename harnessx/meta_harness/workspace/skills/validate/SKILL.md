---
name: validate
description: Self-validation CLIs for the artifacts you write, plus the post-flight workflow the orchestrator runs after `end_turn`. Three categories — validity (canonicalize / dry_fire / contract / synthetic replay) blocks; policy (novelty / evidence) blocks on non-noop rounds; advisory (literals) never blocks. Read when you've just written or edited `config.yaml`, `tools/*.py`, `processors/*.py`, or `templates/*.j2`.
---

# Self-validation — run it before you stop

## What runs when, and why

After your `end_turn`, the orchestrator executes the
`EvolveValidator` workflow in three phases:

| Phase | Checks | Blocks round? |
|---|---|---|
| **validity** | `canonicalize` → `dry_fire` (implicit via canonicalize + replay) → `contract` (for processors that mutate messages) → `synthetic_task` replay | yes — first failure fails the round |
| **policy** | `novelty` + `evidence` (candidates.md + cited_candidates cross-ref) — only when the structural changeset is non-empty | yes on non-noop rounds |
| **advisory** | `literals` — scan authored tools / processors for task-id-shaped literals | **no** — writes `_meta_scratch/LITERALS_WARNING.md` with findings; next round's agent can see it |

There is **no retry loop** at the orchestrator stage — a validity
or policy failure hard-fails the round. Running the validators
yourself before `end_turn` is the only way to fix a broken
artifact inside the same session, while your context still has
the reasoning that produced it.

Every validator prints a **JSON report** to stdout and exits with:
- `0` → `{"ok": true, ...}` — clean
- non-zero → `{"ok": false, "error": "..."}` — findings; read the
  artifact file the validator wrote under `_meta_scratch/`.

Run them via `Bash`. All paths below are absolute.

## Canonicalize (validity)

**What it catches**: YAML shape errors, unresolvable `_target_`
entries, missing / empty / broken-Jinja `template_path` files.

```bash
python -m harnessx.meta_harness.validate_workflow canonicalize \
    <output_dir>/config.yaml
```

Success: `{"ok": true, "checked_templates": N}` — the config loads
AND every `TemplateSystemPromptBuilder` eagerly renders.

Failure: read the `error_type` + `error` fields in the JSON. Common
shapes:
- `TemplateSyntaxError` → fix the Jinja in the `.j2` file named.
- `FileNotFoundError` on `template_path` → you referenced a template
  you forgot to write. Write it, or repoint to the existing one.
- `ModuleNotFoundError: No module named 'tools.foo'` → a `file://`
  import in the config points at a Python file that doesn't exist
  at the path given, or has a typo in the `::fn_name` suffix.

## Dry-fire — processors + tools (validity)

**What it catches**: field-name typos on event dataclasses, stale
kwargs in message / event constructors, ImportErrors / NameErrors /
SyntaxErrors inside authored tool fn bodies.

```bash
python -m harnessx.meta_harness.validate_workflow dry_fire \
    <output_dir>/config.yaml <output_dir>/_meta_scratch
```

Success: `{"ok": true, "processors": {"likely_bugs": 0, ...}, "tools": {"likely_bugs": 0, ...}}`.

Failure: the JSON tells you which side (processors or tools) has
`likely_bugs > 0` and writes the detail to:
- `_meta_scratch/DRY_FIRE_WARNINGS.md` — processors
- `_meta_scratch/DRY_FIRE_TOOL_WARNINGS.md` — tools

Read the file, fix the body, re-run.

**"Notes" are not bugs.** Dry-fire uses minimal dummy inputs, so any
real tool that hits the network on a dummy query will return an error
— the validator downgrades these to notes. Only `likely_bugs` block.

## Contract check (validity — processors only)

**What it catches**: custom processors that mutate `event.messages`
in ways that violate the HarnessX hook contract (empty messages,
+2 insertions per chain, mutating the system prompt from
`on_step_start`, etc.).

```bash
python -m harnessx.meta_harness.validate_workflow contract \
    <output_dir>/config.yaml <output_dir>/_meta_scratch
```

Success: `{"ok": true, "violations": 0}`.

Failure: `_meta_scratch/CONTRACT_VIOLATIONS.md` lists each violation
with `hook`, `violation_type`, `fixture`, and an English message.
Rules live in
`harnessx/core/processor.py::_validate_messages_contract` — open it
when you need to understand what a specific `violation_type` means.

## Synthetic replay (validity — orchestrator only)

Runs one tiny synthetic task through the real run loop after your
`end_turn`. Verifies the config binds, the run loop boots, and no
`exit_reason=error` is produced. Catches bugs the static validators
miss — provider-level issues (empty end_turn, tool schema
mismatch), templates that parse but produce malformed tool calls,
processors that dry-fire fine but crash on real events.

Not exposed as a CLI (needs a model binding only the recipe layer
provides). If you need to sanity-check locally before `end_turn`,
trust `canonicalize` + `dry_fire` + `contract` — they catch 95% of
what replay would catch, for zero model cost.

On failure: `_meta_scratch/REPLAY_FAIL.md` contains the exit
reason, first failing step, and traceback. The `analyze` skill's
"When a prior round failed the replay gate" section tells you how
to interpret different failure shapes next round.

## Literals scan (advisory — never blocks)

**What it catches**: UUIDs and other task-id-shaped literals baked
into authored `tools/*.py` or `processors/*.py` — the "works only on
the one task the author memorised, silently degrades every other"
anti-pattern.

```bash
python -m harnessx.meta_harness.validate_workflow literals \
    <output_dir> <output_dir>/_meta_scratch
```

Output: `{"ok": true, "findings": N}`. Always exits 0. Findings
are written to `_meta_scratch/TASK_SPECIFIC_LITERALS.md` regardless.
The post-flight workflow also emits `LITERALS_WARNING.md` when
`findings > 0` so the next round's agent surfaces the warning.

This used to be a blocking validator; it has been demoted because
hardcoded task-id patterns are best caught by "the next round
fails to generalise on the broader task set", not by a syntactic
heuristic. Fix what it finds anyway — such code almost never
survives.

## Suggested self-validation loop

Before `end_turn`, after you've written everything:

```bash
# 1. config.yaml loads and all templates render
python -m harnessx.meta_harness.validate_workflow canonicalize \
    <output_dir>/config.yaml

# 2. Custom processors + tools at least import and survive a dummy call
python -m harnessx.meta_harness.validate_workflow dry_fire \
    <output_dir>/config.yaml <output_dir>/_meta_scratch

# 3. Processor contract (only matters if you wrote a processor that
#    touches event.messages)
python -m harnessx.meta_harness.validate_workflow contract \
    <output_dir>/config.yaml <output_dir>/_meta_scratch

# 4. Advisory: task-id literal scan (does not block, but fix what it finds)
python -m harnessx.meta_harness.validate_workflow literals \
    <output_dir> <output_dir>/_meta_scratch
```

If 1-3 print `"ok": true`, the static bugs are out. 4 is advisory
regardless. Synthetic replay still runs after you stop — "static
clean" is not the same as "the round passes". If 1-3 print
`"ok": false`, read the artifact they name, fix, re-run.

Running these is cheap (seconds). The alternative — ending your
turn with a broken artifact and hard-failing the round — is
expensive: there is no retry, and the reasoning that produced the
broken artifact is no longer in context by the time the orchestrator
sees the failure. Prefer the self-check.
