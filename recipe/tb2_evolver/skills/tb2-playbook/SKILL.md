---
name: tb2-playbook
description: Terminal-Bench 2 benchmark-specific structural facts for meta-agent evolution rounds. Covers sandbox topology, tool constraints, evolvable config surface, and trajectory signal layout. Read before authoring any candidate that touches the system prompt, processor pipeline, or tool registry.
---

# TB2 Playbook

Structural facts about Terminal-Bench 2 that are not visible in trajectories
and are required to author valid evolution candidates.

---

## Sandbox topology

The task agent and the verifier run in **separate phases** and share
no live filesystem access:

1. **Agent phase** — agent runs inside an isolated Docker container.
   Internet access is **blocked** by default; do not author candidates
   that assume outbound package fetches succeed unless the task itself
   stages the package first.
2. **Verifier phase** — after the agent exits, an external verifier
   runs against the container's final state. The verifier's test
   files are injected **after** the agent session ends and are
   **not present during execution**.

Implication: any instruction in the system prompt telling the agent
to read verifier test files (e.g. `/tests/test_outputs.py`) will
fail silently — those files do not exist during the agent phase.
The agent must infer correctness criteria from the task description
alone.

---

## Tool constraint

The task agent has **exactly one tool: `Bash`**. No `Read`, `Write`,
`Edit`, `Glob`, `Grep`, or browser tools exist. Every filesystem
interaction, dependency install, and verification step goes through
`Bash`.

This is a hard limit set by the benchmark evaluator and cannot be
changed via `config.yaml`. Do not add tools to `tool_registry` —
the harbor environment only exposes `Bash` to the agent regardless.

---

## What `config.yaml` can evolve

The harness config controls the **processor pipeline and system
prompt**, not the benchmark infrastructure. Evolvable surface:

| Lever | How |
|---|---|
| System prompt | Swap the static prompt builder for a `TemplateSystemPromptBuilder` pointing to a `.j2` file you author under `output_dir/templates/` |
| Processor pipeline | Add, remove, reorder, or re-parameterise any `MultiHookProcessor` in the pipeline |
| Custom processors | Author a new class under `output_dir/processors/<name>.py` and reference it via `file://` absolute path |
| Processor knobs | Tune compaction thresholds, edit-loop guard sensitivity, time-reminder fractions — all via constructor parameters in the existing pipeline |

Read the current `config.yaml` to see which processors are already
in the pipeline and what parameters they expose before proposing
changes.

---

## Trajectory signal layout

Each task trial directory under `trajectories_dir/` contains:

```
<task_id>__<hash>/
  agent/
    oh_runs/
      <session>.json          # top-level session summary
      <session_id>/
        <episode>.jsonl       # full event log (tool calls, results, messages)
  verifier/
    test-stdout.txt           # pytest / test runner stdout
    ctrf.json                 # CTRF test results (passed/failed/skipped per test)
  result.json                 # reward, elapsed_s, exit_reason, n_steps, total_cost
```

Priority read order for diagnosis:
1. `result.json` — `reward` (1=pass, 0=fail), `exit_reason`, `n_steps`
2. `verifier/ctrf.json` — which specific tests failed and why
3. `verifier/test-stdout.txt` — full assertion messages
4. Last ~40 lines of the episode JSONL — what the agent was doing
   at its final steps

`exit_reason` values: `done` (natural stop), `max_steps`, `budget_exceeded`,
`loop_detected`, `error`.

---

## Known effective patterns

Prior runs on terminal-bench style tasks have found these interventions
to reliably move the score. Treat as starting hypotheses to validate
against current trajectories — not fixed rules. Better evidence from
your round overrides these priors.

- **Upfront environment survey** — agent surveys the working
  directory layout and available toolchain before writing any code;
  reduces wrong-path and missing-dependency failures
- **Explicit plan before implementation** — agent writes and
  maintains a step-by-step plan; biggest single lever observed,
  especially for multi-file or multi-service tasks
- **Non-interactive discipline** — agent never pauses to ask
  clarifying questions; acts on best interpretation and self-corrects
- **Double-confirmation before exit** — agent re-reads the task
  description and verifies each required output exists and is
  non-empty before declaring done; catches path-mismatch failures
- **Structured reasoning in tool schema** — adding `analysis` and
  `plan` fields to the `Bash` tool definition forces the model to
  articulate intent before executing; improves coherence on complex tasks
- **Effort shaping** — high reasoning effort for early steps (problem
  understanding, design), lower effort for mechanical steps (file
  writes, installs); reduces token waste without harming solution quality

---

## What trajectories will NOT tell you

These failure modes look like ordinary agent failures but have
structural causes:

- **Background process dies after agent exits** — the container
  stays alive for the verifier; `nohup`/`&` processes persist, but
  only if the agent's final Bash call does not kill them
- **Correct logic, wrong path** — the verifier checks exact output
  paths from the task description; a file written to the wrong
  location or with a different extension is a hard failure regardless
  of content correctness
