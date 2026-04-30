---
name: reference
description: Component-authoring reference for HarnessX evolve rounds. Covers the @tool signature and TOOL_SPEC workflow, MultiHookProcessor (hook dispatch table, event fields, messages-mutation contract), system-prompt template editing (Cases A/B/C), config.yaml shape, and the signal→knob guide for the Configuration lever. Read when implementing any lever — Action, Control, Instruction, or Configuration.
---

# Reference — HarnessX component authoring

## Where things live

| File                                  | Purpose                                      |
| ------------------------------------- | -------------------------------------------- |
| `output_dir/config.yaml`              | The new HarnessConfig the caller loads       |
| `output_dir/tools/<name>.py`          | One `Tool` object per file                   |
| `output_dir/processors/<name>.py`     | One `MultiHookProcessor` subclass per file   |
| `output_dir/prompt/<name>`            | System prompt for `SystemPromptProcessor`    |

All authored files are referenced from `config.yaml` with **absolute
paths**. Relative paths are rejected by the loader.

---

## Authoring a new `@tool`

This section assumes you've already decided Action is the right
lever. For **whether** to author a new tool at all (vs. retune an
existing one, write a Control hook, or edit the prompt), see the
`analyze` skill — lever choice lives there, authoring mechanics
live here.

### Before you type — write the shape spec

For anything larger than a short stdlib helper, write the spec in
`_meta_scratch/TOOL_SPEC.md` *before* any code. Short sections, no
prose padding:

```markdown
# TOOL_SPEC — <one-line capability name>

## Capability name
What the agent literally cannot do right now.

## Class, not single task
Which class of failing tasks would this help? If only one exotic
task — abandon, this isn't a capability gap.

## Inputs the agent will pass
Real examples from failing trajectories — URLs, file tokens,
queries, record ids. Synthetic examples are disallowed; they would
pass dry-fire but tell you nothing about how the tool behaves on the
inputs the benchmark keeps presenting.

## Output shape the caller needs (not what you want to produce)
From the caller's POV: structured string? count? page-indexed text?
When the agent gets this back, what can it do with it?

## Failure modes
Named modes — auth denied, timeout, unsupported format, empty
result, rate-limited. What does the tool return in each? Never
return `""` silently — always a diagnostic string the caller can
detect.
```

A spec you cannot write honestly is a spec that is too broad —
narrow it. The spec's value is framing the inputs concretely and
pinning anti-patterns (no generic replacements, no hardcoded task
ids) at the design stage.

### Pre-flight

For pure-stdlib utilities, the pre-flight is small:

1. **Dump the `Tool` dataclass fields** — learn what the decorator
   builds before you write a call site for it:
   ```bash
   python -c "from dataclasses import fields; from harnessx.tools.base import Tool; \
       [print(f.name, '=', f.type) for f in fields(Tool)]"
   ```
2. **Read the nearest builtin** in `harnessx/tools/builtin/` that is
   structurally similar (same return shape, same env access pattern,
   same failure mode). Match its signature and error-handling style;
   don't invent a new convention.

If your tool touches the network or imports a third-party package:

3. **Write the imports as usual.** If dry-fire reports
   `ImportError`, either fall back to stdlib or record the dep in
   `_meta_scratch/NEEDS_FROM_HUMAN.md` and try a different approach.
4. **Check prior art** under `harnessx/tools/builtin/` for the
   keyword in your diagnosis — but decide by observed behaviour on
   this benchmark's inputs, not by whether a similarly named builtin
   exists (see [Prior-art rule](#prior-art-rule)).

### Minimal shape

```python
# output_dir/tools/<name>.py
from harnessx.tools.base import tool


@tool(description="One-line capability sentence — what can the agent now do that it couldn't before.")
async def <fn_name>(<typed args>) -> str:
    # Import third-party deps at module top-level when possible;
    # inline imports are fine for narrow capabilities that don't
    # justify package-level visibility.
    #
    # Return a diagnostic string on every failure mode named in
    # TOOL_SPEC.md. Never return "" — empty output gets absorbed by
    # the caller; a tagged message like "[<tool> failed: <reason>]"
    # lets the agent fall through deliberately.
    ...
```

Concrete tool examples (search fallbacks, file-format parsers,
domain-specific clients) belong in the recipe's `<bench>-playbook`
skill — they encode the benchmark's input distribution and failure
classes, which this skill deliberately does not.

Rules:

- Exactly **one** module-level `Tool` object per file (produced by
  the `@tool` decorator on a function).
- `async def` or regular `def` — both are supported.
- Input args must have type hints; they become the JSON Schema.
- Return a string, or something stringifiable. Tool outputs feed
  back to the model as text.
- **Third-party imports**: just import at module top-level. Dry-fire
  + replay-gate handle env uncertainty downstream.
- **No hardcoded task ids, URLs, or answers** in fn bodies — the
  literals validator rejects these (R4).

Reference from `config.yaml`:

```yaml
tools:
  builtin: [Read, Write, Edit, Glob, Grep, Bash, WebSearch, WebFetch]
  custom:
    - file:///abs/path/output_dir/tools/<name>.py::<fn_name>
```

The string after `::` is the function name (what the tool will be
called as). If the inner agent should use your tool **instead** of
a builtin, remove the builtin's PascalCase name from
`tools.builtin`.

---

## Authoring a new `MultiHookProcessor`

### Mandatory pre-flight

Most contract violations in evolve rounds come from skipping one of
these. Walk them in order every time, regardless of how trivial the
hook looks:

1. **Pick the hook deliberately.** Cross-check against the
   [hook dispatch table](#hook-dispatch-table) — an `on_*` method
   whose suffix isn't a known event becomes an import-time
   `TypeError`.
2. **Dump the event dataclass** (see
   [Event fields](#event-fields--look-them-up-never-guess))
   before typing any `event.<field>` access. Field names drift;
   don't trust memory or pattern-match from another processor.
3. **If your hook mutates `event.messages`** (only `on_step_start`
   and `on_before_model` are allowed to), read
   `harnessx/core/processor.py::_validate_messages_contract`
   **end-to-end**, plus `check_post_hook_invariants` in the same
   file. Then re-check your hook body against **every** violation
   category listed in [Messages-mutation contract](#messages-mutation-contract).
4. **Grep prior art** under `harnessx/processors/control/`,
   `context/`, `memory/`. If a builtin's observed behaviour is
   close to what you want, a config-layer retune beats a new file.

### Minimal shape

```python
# output_dir/processors/tool_error_tail_injector.py
from __future__ import annotations
import dataclasses

from harnessx.core.events import ToolResultEvent, Message
from harnessx.core.processor import MultiHookProcessor


class ToolErrorTailInjector(MultiHookProcessor):
    """When a tool errors, append a short tail note to its result so
    the agent sees the failure mode without re-reading the full
    stderr block."""

    _singleton_group = "tool_error_tail_injector"
    _order = 25  # after loop_detection(20), before otel(30)

    def __init__(self, tail_chars: int = 200):
        self.tail_chars = int(tail_chars)

    async def on_after_tool(self, event: ToolResultEvent):
        result = event.result or ""
        if not isinstance(result, str) or not event.is_error:
            yield event
            return
        tail = result[-self.tail_chars:].strip()
        annotated = f"{result}\n\n[tail] {tail}"
        yield dataclasses.replace(event, result=annotated)
```

Zero-arg-friendly constructor: the loader may instantiate with no
kwargs if the YAML entry doesn't pass any.

(Benchmark-specific processors — for example, one that enforces a
particular answer-format marker — belong in the recipe, authored by
the meta-agent when the benchmark's playbook identifies a grader
requirement that the current config isn't meeting.)

### Hook dispatch table

| Event class          | Method name          |
| -------------------- | -------------------- |
| `TaskStartEvent`     | `on_task_start`      |
| `StepStartEvent`     | `on_step_start`      |
| `BeforeModelEvent`   | `on_before_model`    |
| `ModelResponseEvent` | `on_after_model`     |
| `ToolCallEvent`      | `on_before_tool`     |
| `ToolResultEvent`    | `on_after_tool`      |
| `StepEndEvent`       | `on_step_end`        |
| `TaskEndEvent`       | `on_task_end`        |

Name a method `on_*` but not one of those → import-time `TypeError`.
Use `@on(CustomEventClass)` to handle a custom event under a free
name.

### Event fields — look them up, never guess

```bash
python -c "
from dataclasses import fields
from harnessx.core.events import BeforeModelEvent  # swap in your event
for f in fields(BeforeModelEvent):
    print(f.name, '=', f.type)
"
```

Referencing a non-existent field (`event.message` when the field is
`messages`) passes canonicalize but explodes on the next benchmark
run. Copy-paste field names from the dataclass.

### Messages-mutation contract

Only two hooks may touch `event.messages`: `on_step_start` and
`on_before_model`. Violations surface at runtime as `CONTRACT
[<type>] hook=... processor=...` warnings, and raise
`ContractViolationError` under `HARNESSX_CONTRACT_MODE=strict`.

**Single source of truth is
`harnessx/core/processor.py::_validate_messages_contract`** (plus
`check_post_hook_invariants` for chain-level rules). The list below
is a scanning checklist.

Violation categories:

- **Empty messages** (`empty_messages`) — `before_model` output must
  be non-empty.
- **Length delta out of range** (`length_exceeded`) — `before_model`:
  a processor may not remove messages, and may add at most +1 per
  step.
- **Tail role & insertion rules** — when `before_model` adds +1:
  the pre-hook tail must not already be `user` (content-edit only
  in that case), and the inserted message itself must have
  `role="user"`.
- **Chain-level insertion cap** — across *all* processors in a
  single `before_model` chain, at most +1 user insertion total.
- **`system_window` invariance** — `step_start` may not change
  `messages[0]` when its role is `"system"`.
- **Structural-vs-content edits** — `step_start` may not ONLY change
  the tail (that's `before_model`'s job). `before_model` with
  delta==0: if tail is user, only its `content` may change and the
  prefix must be identical; if tail is not user, no change allowed.

Chain-level invariants in `check_post_hook_invariants`:

- `system` count ≤ 1 and must sit at index 0.
- `raw_track` and `effective_track` must have equal length, matching
  roles at every index, and matching `tool_call_id` for
  `role=='tool'`.

Safest default when unsure: a pure **content rewrite of the existing
tail user message** (delta==0, prefix untouched) — valid in both
`step_start` and `before_model`'s user-tail branch.

### Yielding

A hook is an `async def` generator. Yield one or more events:

- `yield event` — pass through unchanged
- `yield dataclasses.replace(event, messages=new_tuple)` — mutate a
  frozen dataclass field
- `yield ProcessorTriggerEvent(...)` — emit an observability event
  alongside the main one

Yielding nothing blocks the hook; don't do this unless you
intentionally want to suppress a tool call or model turn.

### Ordering

`_singleton_group` prevents two processors of the same kind
stacking. `_order`: 10 = pre, 20 = normal, 30 = post. Builtins use
`token_budget=10`, `cost_guard=10`, `checkpoint=10`,
`loop_detection=20`, `otel=30`. Slot new guards at 15–25 unless you
specifically need to run before or after a named one.

Reference from `config.yaml`:

```yaml
processors:
  - _target_: file:///abs/path/output_dir/processors/tool_error_tail_injector.py::ToolErrorTailInjector
    tail_chars: 200
```

Kwargs in the YAML map directly to the processor's `__init__` args.

---

## Editing the inner agent's system prompt (instruction layer)

The inner agent's system prompt is produced by a ``SystemPromptProcessor``
entry in ``config.yaml``. How you edit it depends on which **builder**
the processor currently uses — **first read ``current_config``** and
look at the ``system_builder._target_`` of the ``SystemPromptProcessor``
entry before you plan a change. The three shapes you'll see in practice:

### Case A — the builder is ``TemplateSystemPromptBuilder``

```yaml
- _target_: harnessx.processors.context.system_prompt.SystemPromptProcessor
  system_builder:
    _target_: harnessx.processors.context.strategies.system_prompt.template.TemplateSystemPromptBuilder
    template_path: /abs/path/to/baseline.j2
```

A ``.j2`` file exists and is the source of truth for the prompt. Do
**not** edit it in place. Instead:

1. ``Read`` the current template at ``template_path``.
2. ``Write`` a modified copy to ``output_dir/templates/<name>.j2``.
3. In your new ``config.yaml``, update the ``SystemPromptProcessor``
   entry's ``template_path`` to the absolute path of your copy.

### Case B — the builder is a static Python class (``_StaticSystemPromptBuilder``, ``DefaultSystemPromptBuilder``, or a benchmark-provided subclass that returns a hardcoded string)

```yaml
- _target_: harnessx.processors.context.system_prompt.SystemPromptProcessor
  system_builder:
    _target_: <some_package>.<SomeStaticBuilder>
```

There is no ``template_path`` — the prompt is baked into a Python class.
Two legal paths:

- **B1 (preferred — promote to a template).** ``Read`` the builder's
  source to extract the literal prompt string it returns. ``Write``
  that string (with your edits) to ``output_dir/templates/<name>.j2``.
  Then swap the builder in ``config.yaml``:

  ```yaml
  system_builder:
    _target_: harnessx.processors.context.strategies.system_prompt.template.TemplateSystemPromptBuilder
    template_path: /abs/output_dir/templates/<name>.j2
    extra_context: {}
  ```

  This is the smallest diff that gives you edit power. The baseline
  class no longer runs — your template is now authoritative.

- **B2 (overlay).** Author a ``MultiHookProcessor`` with an
  ``on_before_model`` hook that inserts a supplementary user message
  before the model call. The original static builder keeps running; your
  processor adds to its output. Right choice when the static prompt is
  load-bearing (it encodes a protocol the harness depends on) and you
  only want to append a nudge, not replace. See the MultiHookProcessor
  section above for the shape.

### Case C — the builder is ``NullSystemPromptBuilder``

```yaml
- _target_: harnessx.processors.context.system_prompt.SystemPromptProcessor
  system_builder:
    _target_: harnessx.processors.context.strategies.system_prompt.null.NullSystemPromptBuilder
```

The benchmark intentionally owns the prompt through a different
mechanism (e.g. role-play turns injected per step). Editing the system
prompt usually **isn't the right lever** — the prompt is empty on
purpose, and the benchmark feeds its own instructions in-turn. Prefer
Control (a processor that edits the messages the benchmark injects) or
Instruction-via-a-new-skill-file over trying to stuff content into the
system slot.

### Rules for all instruction-layer edits

- **Check the benchmark's grader before touching answer-format
  rules.** Graders vary across benchmarks — some parse a marker
  string, some run a judge LLM, some run verifier scripts inside a
  sandbox, some use a simulated user. Removing format requirements
  that feed the grader silently breaks the round. Consult the
  ``<bench>-playbook`` skill for what the grader actually reads
  before editing any answer-format block.
- **Keep core tool descriptions** that the inner agent needs to know
  what's available. Which tool descriptions are load-bearing varies
  by benchmark — check the playbook.
- **Make edits surgical.** Add or modify specific sections; don't
  rewrite from scratch. A small targeted change is easier to attribute
  when it lands or reverts.
- **Bundle by evidence, not by reflex.** Multiple instruction changes
  in the same round are fine when each is independently grounded —
  its own cited candidate, its own cluster of affected tasks, and
  Pareto-positive against regression risk + cost shift. See `analyze`
  skill → *Multi-lever rounds* for the general rule. What to avoid
  is the opposite reflex: stacking two weakly-supported edits to
  "cover more ground", which makes attribution impossible when the
  round reverts. Rule of thumb: if you can write a separate
  ``## Candidate C-N`` section (with its own Signal / Verified /
  Retroactive check / Why X not Y) for each edit, they can ship
  together; if two edits collapse into one candidate with the same
  evidence, pick the stronger framing and keep the other as a
  follow-up in the journal.

---

## `config.yaml` shape — non-negotiable

```yaml
processors:
  # Flat list. NEVER nest under `processors.'*':` or any hook bucket —
  # the loader rejects that shape.
  - _target_: harnessx.processors.context.system_prompt.SystemPromptProcessor
    system_builder:
      _target_: harnessx.processors.context.strategies.system_prompt.template.TemplateSystemPromptBuilder
      template_path: /abs/path/to/template.j2
      extra_context: {}
  - _target_: harnessx.processors.control.token_budget.TokenBudgetProcessor
    ratio: 0.85
  - _target_: harnessx.processors.control.cost_guard.CostGuardProcessor
    max_usd: 2.0
  - _target_: file:///abs/path/output_dir/processors/tool_error_tail_injector.py::ToolErrorTailInjector
    tail_chars: 200

tools:
  builtin: [WebSearch, WebFetch, Browser, Read, Write, Edit, Glob, Grep, Bash]
  custom:
    - file:///abs/path/output_dir/tools/<name>.py::<fn_name>
```

Key rules:

- `processors:` is a **flat list**, not a dict keyed by hook bucket.
- `tools.custom` entries are **bare strings** (`file://...::fn_name`),
  not `_target_` dicts.
- All `file://` paths are **absolute**; relative paths are rejected.
- Kwargs in each entry map directly to the target's `__init__` args.

### Starting point for edits

Copy the current config, then edit the copy — smaller diff = fewer
ways to break canonicalize:

```bash
cp <current_config_path> <output_dir>/config.yaml
```

### No-op case

When no change is warranted, a valid no-op `config.yaml` is a byte-
for-byte copy of `<current_config_path>`. Still write it; the caller
treats absence of `config.yaml` as a failure.

### Common canonicalize failures

| Error | Likely cause |
|---|---|
| `'processors:' must be a flat list` | Legacy hook-bucket form. Remove the `'*':` wrapper; emit a flat list of `_target_` dicts. |
| `'tool_registry:' is not a recognised key` | Legacy format. Put tools under `tools.builtin` / `tools.custom`. |
| `tools.custom entries must be '...' strings` | You wrote `{_target_: file://...}` instead of the bare string. Drop the `_target_:` wrapper. |
| `TemplateSyntaxError` | Fix the Jinja in the `.j2` file named in the error. |
| `FileNotFoundError` on `template_path` | You referenced a template you forgot to write. |
| `ModuleNotFoundError` from `file://` import | Authored file path typo, or `::symbol` doesn't match the exported name. |
| `singleton_group ... conflicting registrations` | Same processor `_singleton_group` listed twice. |

Verify before `end_turn`:

```bash
python -c "from harnessx.core.harness import HarnessConfig; HarnessConfig.from_yaml_file('<output_dir>/config.yaml').canonicalize(); print('OK')"
```

Or use the `validate` skill's CLI suite.

---

## Tuning existing builtins (Configuration layer)

When the lever is Configuration rather than Action/Control/Instruction,
edit kwargs on an existing processor in place. The common tunables:

```yaml
# Cost ceiling per task
- _target_: harnessx.processors.control.cost_guard.CostGuardProcessor
  max_usd: 2.0

# Loop detection — fingerprints recent steps
- _target_: harnessx.processors.control.loop_detection.LoopDetectionProcessor
  window_size: 10       # how many steps to compare
  threshold: 4          # consecutive matching fingerprints before kill

# Token budget — compaction trigger
- _target_: harnessx.processors.context.token_budget.TokenBudgetProcessor
  ratio: 0.85           # fraction of model context window to use

# Checkpoint cadence
- _target_: harnessx.processors.control.checkpoint.CheckpointProcessor
  every_n: 5            # save state every N steps
```

Judgement on when to move each knob is trajectory-driven. Use the
signal table below as a starting point, read the trajectory evidence,
pick the direction, retune, verify with canonicalize. The
orchestrator's gating layer will revert a miscalibration.

### When to move which knob

| Signal in trajectories | Knob to move | Direction |
|---|---|---|
| Recurring `exit_reason: budget_exceeded` while agent was still making progress | `CostGuardProcessor.max_usd` | raise |
| Most failures hitting the step limit (steps = max_steps) | **Not a config knob** — record as `NEEDS_FROM_HUMAN: raise max_steps in runner` | — |
| `exit_reason: loop_detected` on tasks where the agent was visibly making progress | `LoopDetectionProcessor.threshold` | raise |
| Agent repeats the same action many times before loop detection fires | `LoopDetectionProcessor.threshold` or `window_size` | lower |
| Compaction artifacts visible (lost context mid-task, agent re-asking already-answered questions) | `TokenBudgetProcessor.ratio` | lower |

The signal column describes a pattern across multiple tasks, not a single incident. A one-off is
idiosyncratic — log it as `NEEDS_FROM_HUMAN` rather than tuning a knob.

---

## Prior-art grep

Before authoring anything new, check what's already there:

- `harnessx/processors/control/` for Control-lever components
  (`token_budget.py`, `cost_guard.py`, `loop_detection.py`,
  `parse_retry.py`, `repeated_edit_detector.py`,
  `self_verification.py`, `tool_failure_guard.py`, …)
- `harnessx/tools/builtin/` for Action-lever components
  (local file reads, web search, HTTP fetch, shell commands).

The goal of this grep is **mechanical orientation** — match naming
conventions, signature style, error-handling idioms of the nearest
sibling. The decision of *whether* a similarly named builtin
covers your need lives in `analyze` skill → *Disambiguating
adjacent levers*, and turns on observed behaviour on this
benchmark's trajectories, not on filename similarity.

---

## Integration check — orchestrator post-flight

After your `end_turn`, the orchestrator runs:

1. **Validity** (always): `canonicalize` + `dry_fire` + `contract`
   + `synthetic_task` replay. Each in turn; first failure stops
   and fails the round.
2. **Policy** (when config diff is non-empty): `novelty` (no
   re-proposing reverted hypotheses) + `evidence` (candidates.md
   + cited_candidates cross-ref).
3. **Advisory** (never blocks): `literals` writes to
   `_meta_scratch/LITERALS_WARNING.md` when hardcoded task-shaped
   literals appear in authored code. Fix them when you see the
   warning — they're not fatal but they signal a candidate that
   likely won't generalize.

Run the validity checks yourself before `end_turn` via the
`validate` skill — that catches the common bugs without paying
the orchestrator-replay round cost.

---

## Anti-patterns — apply to tools, processors, and templates alike

1. **Generic replacement.** "A better WebFetch that handles
   everything." Scope to the cluster. Existing tools compete in the
   general case; new tools compete on a specific class of inputs.
2. **Lifting an implementation from outside this repo.** Other
   teams' blog posts / papers describe shapes that worked on their
   input distribution. You have not seen their failing trajectories.
   Use outside work as shape inspiration, not as code.
3. **Skipping the shape spec (TOOL_SPEC.md).** Typing `@tool`
   without the spec produces components that don't match the
   caller's actual need.
4. **Multi-responsibility component.** "fetch-and-extract-and-
   summarize" → split into separate `@tool`s or processor hooks.
   One responsibility per file. The agent composes; you don't.
5. **Sprawling third-party deps.** `pdfplumber` + `pandas` +
   `openpyxl` + `sympy` in one tool. Pick one axis per tool.
6. **Silently suppressing failure with `""` / `None`.** Return a
   diagnostic string (`[<tool> failed: <reason>]`), not empty.
   Empty gets absorbed; diagnostics let the agent fall through.
7. **Hardcoded task ids, URLs, or answers in fn bodies** — R4 rule.
   Triggers a `literals` advisory warning; not fatal on its own,
   but such components almost never survive the next benchmark
   round (they only work on the one task the author memorised).
8. **For processors: mutating `event.messages` outside
   `on_step_start` / `on_before_model`** — contract violation,
   automatically rejected.
9. **For templates: editing the target's source `.j2` in place** —
   always copy to `output_dir/templates/<name>.j2` and point
   `template_path` at the copy.

---

## End-of-turn checklist

- [ ] `_meta_scratch/TOOL_SPEC.md` exists when the change is
  non-trivial (any new tool / processor beyond a stdlib helper)
- [ ] New files under `output_dir/tools/` or `output_dir/processors/`
  are referenced from `config.yaml` with absolute `file://` paths
- [ ] `canonicalize` validator → `{"ok": true}`
- [ ] `dry_fire` / `contract` validators → `"ok": true` when you
  authored new tools / processors / templates (see `validate` skill)
- [ ] `literals` validator run (advisory, non-blocking): if it
  reports findings, fix them before `end_turn` — they don't fail
  the round but the component likely won't survive the next benchmark
  round
- [ ] `memo_path` has a new `## Round N` entry with YAML frontmatter
  (see `journal` skill): `round`, `hypothesis_id`, `levers`,
  `predicted_affected`, plus prose with Why / Changes / Evidence
