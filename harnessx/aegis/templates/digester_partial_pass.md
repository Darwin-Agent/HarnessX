# Digester — PARTIAL_PASS

Task `{{ task_id }}` passed some rollouts and failed others. This is the
highest-signal case: compare pass-trace vs fail-trace, find the decisive
divergence, and isolate the causal step — not a vibe.

**Evidence discipline is the point.** Every claim you make in Layer B
and Layer C MUST cite a trajectory anchor and quote a ≤200-char snippet
from the raw event. Claims without anchors are treated as hallucination.

**Anchor mechanics — read carefully:**

1. The exact relative paths to use are listed in Layer A's "Tool calls"
   section (the `**rN** → trajectories/<file>` lines). **Copy them
   verbatim — do not retype filenames or guess UUIDs.** A digest whose
   anchor target does not exist is rejected by the structure gate.
2. Anchor format: `trajectories/<exact-filename>#step_<N>` for a
   specific step, or `trajectories/<exact-filename>#summary` when the
   evidence is the rollout as a whole (no single step is the locus).
3. **If Layer A's "Tool burst" row is non-empty, you MUST list at least
   one `tool_burst_loop` (severity=high) entry in Layer B citing the
   peak step.** Burst is the strongest behavioural signal we extract;
   ignoring it produces a misleading digest.

## Trajectories to read

{% for path in trajectory_paths -%}
- `{{ path }}`
{% endfor %}

The pass/fail split is embedded in the `_rN` suffix — Layer A's "Exits"
table tells you which rollout passed. Compare the pass-trace and the
fail-trace step by step; the decisive divergence is usually one tool
call, one prompt rule hit, or one missing step.

## Layer A — Pre-computed trace facts (inject verbatim)

When you write your digest, the Layer A block below MUST be the FIRST
section of the file. Do not paraphrase, reorder, or omit it.

```
{{ trace_facts_md }}
```

## Output contract

Call `write_tool` exactly once on `{{ digest_out_path }}` with this shape:

```
pattern: PARTIAL_PASS
failure_mode: <short_snake_case_tag>

{{ '{{ trace_facts_md — pasted verbatim }}' }}

## Pathology signals (Layer B — structured)

```yaml
- type: <see vocabulary below>
  rollout: <rollout where this pathology appeared (the failing one, usually)>
  anchor: trajectories/<file>#step_<N>
  snippet: "<≤200-char quote from that exact event>"
  observation: "<one sentence>"
  severity: <low|medium|high>
```

## Decisive Divergence (Layer C — diagnosis)

ONE paragraph. Name the single step where pass-trace and fail-trace
diverged. Cite BOTH rollouts:
- `trajectories/<task>_r0.jsonl#step_N`
- `trajectories/<task>_r1.jsonl#step_M`

Quote the decisive snippet from each (≤ 100 chars each) so the reader
can see the divergence without opening the trace.

## Why one passed / Why the other failed

2 short paragraphs. Each claim has a `(anchor)` at its end. No
"this step seemed better" without an anchor.
```

## Pathology type vocabulary

- `tool_effect_missing` — tool returned content; next model step did not use it.
- `repeat_without_progress` — same `(tool, args_sha)` fired ≥2× consecutively.
- `error_ignored` — tool returned error and model continued without acknowledging.
- `multimodal_silent_drop` — tool returned multimodal; text marker reached
  the model instead (the `render_pdf_page` failure mode).
- `hallucinated_reference` — model cited a fact or URL not present in trace.
- `missing_capability` — model explicitly said it lacked a tool or data source.
- `budget_starvation` — exit != done and ceiling hit while still progressing.
- `prompt_rule_violation` — model broke a rule stated in its system prompt.
- `final_answer_brittle` — final answer rests on one unverified chain.
- `other:<tag>` — only when nothing above fits.

## What Critic will verify

1. Layer A section verbatim.
2. Every Layer B entry has `anchor`, `snippet`, `observation`, `severity`.
3. Every `anchor` matches an actual trajectory step.
4. Every `snippet` is findable in the referenced event.
5. Layer C cites BOTH pass and fail rollouts at the divergence point.

Anchor references to `trajectories/...#step_N` — relative paths only.
Final_output without a `write_tool` call discards the digest.
