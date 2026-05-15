# Digester — ALL_PASS

Task `{{ task_id }}` passed in every rollout. Your job is NOT to celebrate;
it is to identify the reusable strategy and any latent fragility that could
break this on related tasks. Layer A tells you what happened; Layers B/C
tell future-Evolver what to preserve and what still worries you.

**Evidence discipline is the point.** Every claim in Layer B and Layer C
MUST cite a trajectory anchor and quote a ≤200-char snippet from the raw
event. No anchor = hallucination.

## Trajectories to read

{% for path in trajectory_paths -%}
- `{{ path }}`
{% endfor %}

## Layer A — Pre-computed trace facts (inject verbatim)

When you write your digest, the Layer A block below MUST be the FIRST
section of the file. Do not paraphrase, reorder, or omit it.

```
{{ trace_facts_md }}
```

## Output contract

Call `write_tool` exactly once on `{{ digest_out_path }}` with this shape:

```
pattern: ALL_PASS
strategy: <short_snake_case_tag>
failure_mode: none

{{ '{{ trace_facts_md — pasted verbatim }}' }}

## Pathology signals (Layer B — structured)

```yaml
# ALL_PASS digests often have few or no pathologies. If you find none,
# write `[]`. Do NOT invent entries — empty is truthful here. BUT: any
# `tool_effect_missing` / `repeat_without_progress` / `multimodal_silent_drop`
# visible in Layer A MUST be listed even on a pass — they indicate latent
# fragility this task got lucky around.

- type: <vocabulary below>
  rollout: r0
  anchor: trajectories/<file>#step_<N>
  snippet: "<≤200-char quote>"
  observation: "<one sentence>"
  severity: <low|medium|high>
```

## Reusable Strategy (Layer C)

1–3 short paragraphs. The decision pattern or tool-use order that worked.
Each factual claim ends with an anchor `(trajectories/<file>#step_N)` or
Layer-A table reference `(Layer A · Tool calls · r0 step_3)`.

## Latent Fragility

1–2 bullets. Each names an assumption that could make this strategy break
on related tasks — e.g. "relies on WebSearch returning exact string X"
(anchor), "would fail if tool output exceeded 8k chars" (anchor). No anchor
= not a real fragility, don't list it.
```

## Pathology type vocabulary

Same list as other patterns:
- `tool_effect_missing`, `repeat_without_progress`, `error_ignored`,
  `multimodal_silent_drop`, `hallucinated_reference`, `missing_capability`,
  `budget_starvation`, `prompt_rule_violation`, `final_answer_brittle`,
  `other:<tag>`.

## What Critic will verify

1. Layer A section verbatim.
2. If Layer B is `[]`, Layer A shows no repeats / next_uses_result=NO /
   multimodal_coerced. If those DO appear in Layer A, Layer B must mention
   them even on ALL_PASS.
3. Every Layer B entry has `anchor`, `snippet`, `observation`, `severity`.
4. Every Latent Fragility bullet has an anchor.

Anchor references to `trajectories/...#step_N` — relative paths only.
Final_output without a `write_tool` call discards the digest.
