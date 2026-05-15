# Digester — ALL_FAIL

Task `{{ task_id }}` failed in every rollout. You write a 3-layer digest:
Layer A is already supplied (read-only facts). Layer B is structured
pathology signals with evidence. Layer C is your narrowed diagnosis.

**Evidence discipline is the point.** Every claim you make in Layer B
and Layer C MUST cite a trajectory anchor and quote a short snippet from
the raw event that supports it. A claim without an anchor is indistinguishable
from hallucination; Critic will reject downstream candidates that rely on it.

## Trajectories to read

{% for path in trajectory_paths -%}
- `{{ path }}`
{% endfor %}

## Layer A — Pre-computed trace facts (inject verbatim)

When you write your digest, the Layer A block below MUST be the FIRST
section of the file. Do not paraphrase, summarise, reorder, or omit it.
Just paste it as-is at the top, then append Layers B and C below it.

```
{{ trace_facts_md }}
```

## Output contract

Call `write_tool` exactly once on `{{ digest_out_path }}` with this shape:

```
pattern: ALL_FAIL
failure_mode: <short_snake_case_tag>

{{ '{{ trace_facts_md — pasted verbatim }}' }}

## Pathology signals (Layer B — structured)

```yaml
- type: <see vocabulary below>
  rollout: <r0|r1|...>
  anchor: trajectories/<file>#step_<N>
  snippet: "<≤200-char quote from that exact event — the reader must be able
            to open the trace and see this string>"
  observation: "<one sentence linking the snippet to the pathology type>"
  severity: <low|medium|high>

- type: ...
  ...
```

## Common Root Cause (Layer C — diagnosis)

1–2 paragraphs. Every factual claim MUST end with `(anchor)` — either
`trajectories/<file>#step_N` or a Layer-A table row reference like
`(Layer A · Tool calls · r0 step_4)`.

## Harness Bias Hypothesis

1–3 bullets. Each bullet names ONE bucket (prompt / tools / config /
processor) and the specific element (template line, tool name, processor
class, kwarg key). Each bullet MUST cite evidence from Layer A or a trace
anchor — no naked "the prompt is too verbose"-style claims.
```

## Pathology type vocabulary

Prefer these types; use `other:<tag>` only if nothing fits:

- `tool_effect_missing` — tool returned something, but next model step did
  not reference the return value (see Layer A's "next_uses_result = NO" column).
- `repeat_without_progress` — same `(tool, args_sha)` fired ≥2× consecutively.
  Layer A lists these under "Repeated tool calls".
- `error_ignored` — tool returned an error (return_type=error) and model
  continued without acknowledging.
- `multimodal_silent_drop` — tool returned multimodal content but only a
  text marker reached the model (return_type=multimodal_coerced in Layer A).
  This is the `render_pdf_page` failure mode.
- `hallucinated_reference` — model cited a fact, tool output, or URL that
  appears nowhere in the trace.
- `missing_capability` — model explicitly said it lacked a tool / data
  source (search the assistant thinking / content for phrases like
  "I don't have access to", "cannot retrieve").
- `budget_starvation` — exit_reason != done AND total_steps hit ceiling AND
  last few tool calls were making progress (would have worked with more steps).
- `prompt_rule_violation` — model broke a rule stated in its system prompt
  (cite the rule line + the violating step).
- `final_answer_brittle` — final answer rests on one unverified inference
  chain of length ≥ 2 with no independent check.

## What Critic will verify (don't ship a digest that fails these)

1. Layer A section is present and verbatim.
2. Every Layer B entry has `anchor`, `snippet`, `observation`, `severity`.
3. Every `anchor` matches an actual trajectory file + step number.
4. Every `snippet` is findable in the referenced event (spot-checked).
5. Layer C citations are concrete, not wave-of-the-hand.

Anchor references to `trajectories/...#step_N` — relative paths only.
Final_output without a `write_tool` call discards the digest.
