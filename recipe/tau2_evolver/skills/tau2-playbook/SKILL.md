---
name: tau2-playbook
description: τ²-bench-specific benchmark guidance (Sierra Research; retail / airline / telecom). Multi-turn dialogue eval with multiplicative reward (DB × action × NL-assertion) and pass^k reliability metric. Catalogues tau2 failure modes and techniques from public writeups, mapped to HarnessX's four levers (config / control / action / instruction). Use when forming a hypothesis or scanning for which intervention fits a pattern seen in tau2 trajectories.
---

# τ²-bench: Benchmark-specific Guidance

## Current Benchmark

- **What**: τ²-bench (Barres et al., Sierra, [arXiv 2506.07982](https://arxiv.org/abs/2506.07982)) — multi-turn
  dialogue eval where an LLM-driven user converses with the agent, the
  agent calls domain tools that mutate a shared database, and a
  three-component evaluator scores the run. Evolves out of τ-bench
  (ICLR'25, retail+airline) → τ²-bench (adds telecom as dual-control
  Dec-POMDP) → τ³-bench (Oct'25, adds banking+voice). Repo:
  [github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench). HarnessX targets τ².

- **Domains used by this evolver**:

  | Domain    | Tasks (HarnessX split) | Agent tools        | Grading                     |
  | --------- | ---------------------- | ------------------ | --------------------------- |
  | `retail`  | 114                    | 7 W + 6 R          | DB-state equality (strict)  |
  | `airline` | 50                     | 6 W + 6 R          | DB-state equality (strict)  |
  | `telecom` | 114 (`base` split)     | 6 W + 7 R + user-side tools | outcome-state (lenient) |

  Full telecom pool is 2285 tasks; the runner always uses `base`.

- **Reward ∈ [0, 1]** = product of three components; any zero zeros everything:
  - **DB check** — final database matches ground truth.
  - **Action checks** — per-action `action_match` booleans.
  - **NL assertions** — LLM judge rates claims about the dialogue (temp 0.3).

- **pass^k vs pass@k** — Sierra's signature framing. `pass@k` = at least
  one of k runs passes; **`pass^k`** = *all* k runs pass. A 70 % `pass^1`
  agent collapses to ~25 % `pass^8`. We currently run `pass^1`
  (`defaults.NUM_TRIALS = 1`); raise `NUM_TRIALS` to surface reliability.

- **Realistic baselines** (τ²-verified `avg@3`, Nova-2 report Mar 2026 —
  citable public numbers). Airline is uniformly hardest (brittle
  single-solution grading); telecom is highest (multiple valid paths):

  | Model                | Telecom | Retail | Airline |
  | -------------------- | ------- | ------ | ------- |
  | GPT-5                | 86.5    | 78.3   | 72.0    |
  | Claude Sonnet 4.5    | 78.1    | 77.2   | 66.8    |
  | GPT-5-mini           | 71.1    | 73.7   | 68.8    |
  | Claude Haiku 4.5     | 54.7    | 69.1   | 54.0    |

  HarnessX's own runs are in `benchmarks/tau2/README.md`.

## Task Structure

Each task is a multi-turn dialogue built around one customer-service
goal: user opens with an ambiguous request → agent info-gathers via read
tools → diagnoses and proposes → user confirms → agent executes write
tools → user acknowledges or adds a follow-up, then stops (`###STOP###`).

Tools split into two classes by name convention (basis of
`PhaseAwareToolFilter` in `benchmarks/tau2/tool_filter.py`):

| Class  | Prefixes                                                         | Examples                                        |
| ------ | ---------------------------------------------------------------- | ----------------------------------------------- |
| Read   | `get_`, `list_`, `search_`, `find_`, `check_`, `look_`, `think*` | `get_order_details`, `list_orders`              |
| Action | everything else                                                  | `refuel_data`, `enable_roaming`, `toggle_*`     |

## Per-Domain Cheat-Sheets

Cross-domain guidance is usually wrong in subtle ways. Anchor any
instruction-layer fix to the specific domain that surfaced the failure.

**retail** — order *status* drives tool choice: `modify_pending_order_items`
and `cancel_pending_order` need `pending`; `exchange_delivered_order_items`
needs `delivered`. Returns must use the **original payment method** (no
swapping). Exchanges must be same product, different option. Courtesy
credits often require escalation, not silent issue.

**airline** (hardest by pass^1) — **basic-economy cannot be modified**
(only cancellable via insurance); strict baggage rules per cabin; user
ID and reservation ID must be *elicited*, never fabricated. Hidden user
constraints ("I only accept EWR, not JFK") live in `user_instruction`
and surface only when the agent asks. Single-solution ground truth means
reward is brittle — a *correct* alternative can still score zero.

**telecom** (dual-control) — task-class hierarchy: `service_issue` <
`mobile_data_issue` < `mms_issue`; harder classes require resolving
underlying issues first. The agent **cannot toggle user-side settings**
(airplane mode, cellular data, roaming toggle, APN); it must instruct
the user in natural language, wait for confirmation, then re-read
state. `PolicyHintProcessor` (`benchmarks/tau2/policy_hint.py`) is an
off-by-default telecom companion that injects positive remediation
hints when conditions like `data_exceeded`, `roaming_disabled_abroad`,
`check_correct_line` are detected in tool results.

## Analyzing Failures

Frontmatter fields emitted by the evolver (always present):

| Field                | What it tells you                                                  |
| -------------------- | ------------------------------------------------------------------ |
| `task_id`            | τ² task identifier                                                 |
| `exit_reason`        | `done` / `max_steps` / `max_errors` / `too_many_errors` / `agent_stop` |
| `reward`             | Float [0,1]; 1.0 = full pass                                       |
| `eval_passed` / `eval_score` / `eval_reason` | SOUL.md-generic aliases for `reward >= 1.0`, `reward`, and the `termination_reason` + failed-action names |
| `num_messages`       | Total messages exchanged                                           |
| `cost_usd` / `elapsed_s` | Cost + wall clock                                              |
| `tools_used` / `tool_call_counts` / `tool_error_counts` | Tool histograms (≥30% error rate on one tool = red flag) |

Cross-reference with the trajectory body's `## Result` block, which
prints raw `reward_info`: `db_match`, `db_reward`, per-action
`action_checks`, and `nl_assertions` outputs. Those are the most direct
evidence for which component let you down.

## Known Techniques That Improve tau2 Scores

**Already wired in `harness_config.yaml`:**

| Technique | Layer | Knob / file | Approx. lift |
| --- | --- | --- | --- |
| **Phase-aware tool gating** — restrict action tools in first N turns | Configuration | `PhaseAwareToolFilter.read_only_steps` (default 2); bump to 3 for policy-heavy domains | moderate (closes failure mode A) |
| **Loop-detection** — catch repeated identical tool calls | Configuration | `LoopDetectionProcessor`; tune threshold | moderate on confirmation-heavy tasks |
| **Hardened baseline toolkit** — `ParseRetryProcessor`, `ToolCallCorrectionLayer`, `ToolFailureGuard`, `TokenBudgetProcessor` | Configuration | all in `harness_config.yaml` | small each, meaningful in aggregate |
| **SystemAppendProcessor** — append domain guidance without replacing tau2's policy prompt (see pattern below) | Instruction | `recipe/tau2_evolver/system_append_processor.py` + `guidance_<domain>.md` | moderate-large on policy-sensitive tasks |
| **PolicyHintProcessor** — condition-triggered positive directives for telecom `mobile_data_issue` | Instruction | `benchmarks/tau2/policy_hint.py` (off by default) | large on the mobile-data subset |
| **Extended Thinking + budget** — Anthropic reasoning on Sonnet/Opus | Configuration | `--extended-thinking --thinking-budget-tokens 62976` | substantial; a Sierra-highlighted axis |

**Public-writeup techniques worth authoring as new candidates:**

| Technique | Layer | What to write | Approx. lift |
| --- | --- | --- | --- |
| **Policy rewrite → decision tree** — recast domain policy into numbered Check→If→Then branches with explicit tool names and args | Instruction (`guidance_<domain>.md`) | Replace verbatim policy text with decision-tree form via `SystemAppendProcessor` | **+15-22 pp documented** — Quesma 2025 on GPT-5-mini; Barres replicated ~+20 pp on GPT-4.1 |
| **Input-reformulation pre-pass (IRMA)** — before each agent turn, rewrite the latest user message with the relevant policy snippet + shortlisted tools | Control (`on_before_model` processor) | Ship as `output_dir/processors/irma.py` | +12-19 pp on pass^5 (arXiv 2508.20931) |
| **Plan-confirm-execute** — planner emits JSON plan, asks one disambiguating question, executor issues the write | Control + Instruction | Plan-schema guidance rule + processor that blocks writes until plan is in `state.slots` | designed for irreversible writes; catches hallucinated IDs and basic-economy-modify attempts pre-execution |
| **Trust-score message revision** — score each outgoing assistant message and rewrite low-trust ones | Control (`on_after_model` processor) | Call cheap scorer or main model at low temp; rewrite below threshold | Cleanlab 2025: all-domain lift on GPT-5 |
| **Dependency-aware reordering** — when policy requires a different execution order than the user stated | Instruction | Guidance rule: "if X requires precondition Y, do Y first regardless of user order" | large on multi-request retail tasks |
| **Case-specific control processor** — e.g. `on_before_tool` refusing writes when order-status vs verb contradict | Control | `output_dir/processors/<name>.py` | case-dependent (targets D or E) |

## SystemAppendProcessor pattern

tau2 injects its own system prompt (the full policy document + tool
descriptions). **Never replace it with `TemplateSystemPromptBuilder`** —
that deletes the policy text and scores collapse. `SystemAppendProcessor`
reads tau2's system message on `on_task_start`, appends a markdown
guidance file, and sets `task_system_prompt` to the combined text.

```yaml
# output_dir/config.yaml
processors:
  - _target_: recipe.tau2_evolver.system_append_processor.SystemAppendProcessor
    append_path: /abs/path/to/recipe/tau2_evolver/guidance_retail.md
```

The shipped `guidance_retail.md` encodes execute-on-stated-intent,
exchange-vs-modify disambiguation, and attempt-tool-calls discipline.
For airline/telecom, author a new file drawn from patterns seen in
*that domain's* trajectories — this is where the Quesma/Barres
decision-tree rewrite lives.

## Common Failure Modes

### A. Premature action call

Agent fires a write on turn 0 or 1 from the user's opening message
alone, before any read tool.

- **Signal**: first entry of `tool_call_counts` is an action tool;
  small `num_messages`; `db_match=False`.
- **Fix**: raise `PhaseAwareToolFilter.read_only_steps`; or guidance
  rule "always call at least one read before any action".

### B. Over-confirmation / re-asking loop

Agent proposes a plan, user confirms, agent asks for re-confirmation,
exchange loops until `max_steps`.

- **Signal**: `num_messages ≥ 15` on a 4-6 turn task; many reads, no
  writes; `exit_reason = max_steps` or `user_stop`.
- **Fix**: instruction rule "execute on stated intent, do not ask for
  re-confirmation" (already in `guidance_retail.md`); tighter
  `LoopDetectionProcessor`.

### C. Hallucinated arguments or unsupported claims

Two shapes:
- **In a tool call**: invented order ID / customer ID / parameter not
  present in dialogue or read-tool output. Signal: `tool_error_counts`
  spikes on a mutation tool; body contains `Order X not found`.
- **In an assistant message (no tool call)**: confident numeric or
  policy claim the agent didn't look up. Signal: `nl_reward < 1.0`
  with `db_match=True`; no corresponding read tool in the turn.

Fix: guidance rule "every ID, number, or date must appear verbatim in
a prior tool result or user message"; or a trust-score / before-tool
processor that pattern-matches arguments against recent read outputs.

### D. Exchange-vs-modify tool confusion (retail)

`modify_pending_order_items` on a delivered order, or
`exchange_delivered_order_items` on a pending order.

- **Signal**: `failed_actions` names one of those tools; dialogue has
  "change / update / exchange / swap / replace" without disambiguating.
- **Fix**: verb → tool + order status mapping (already in
  `guidance_retail.md`); or a control processor refusing the wrong tool
  when status contradicts.

### E. Policy violation

Agent does something forbidden — swaps payment methods on a return,
refunds without auth, modifies basic-economy airline reservation,
issues a courtesy credit without escalation.

- **Signal**: `action_checks` has an *extra* action or a missing
  required auth step; `db_reward < 1.0` even with `db_match=True`.
- **Fix**: surface the specific rule in `guidance_<domain>.md` as an
  explicit shortcut ("when X and policy Y, do Z, do NOT W"). The
  decision-tree rewrite is the scalable version.

### F. Stop-guard / user-sim boundary collisions

User sim sends "Yes, proceed.###STOP###" on the confirming turn; the
orchestrator terminates before the agent executes the write.
`StopGuardUserSimulator` (`benchmarks/tau2/stop_guard.py`) strips this
when the stripped content starts with `yes\b` — zero false positives on
a 114-task retail run, ~+8 pp on retail Sonnet-4.5.

- **Signal**: `exit_reason = user_stop`, small `num_messages`, no
  write in `tool_call_counts`, DB unchanged.
- **Fix**: **pipeline issue, not agent bug.** If a new stop-token
  variant slips through, log to `_meta_scratch/NEEDS_FROM_HUMAN.md`;
  do not build an agent-layer fix for this.

### G. Dependency-graph literalism

Agent follows the user's stated sequence literally when policy requires
a different order — e.g. closing an account before redirecting its
balance, when "no CLOSED accounts" is a precondition.

- **Signal**: `failed_actions` names a write that's individually
  correct but arrived in the wrong order; trajectory shows exact
  user-dictation order; `db_match=False` on a multi-step request.
- **Fix**: guidance rule "derive the dependency graph from the policy,
  execute preconditions first regardless of stated order"; or a
  plan-confirm-execute structure that exposes the full plan first.

## Red Flags — Pipeline Bugs, Not Agent Bugs

Record in `_meta_scratch/NEEDS_FROM_HUMAN.md`; do NOT build an
agent-layer candidate:

- **`reward=0.0` with all `action_checks.action_match=True` and `db_match=True`** → NL-assertion judge failed or timed out.
- **`exit_reason=error` with no traceback** → simulation harness swallowed an exception.
- **`tool_error_counts` dominated by one tool across *most* tasks** → that tool is broken upstream, not miscalled.
- **`cost_usd=0.0` on a non-zero-message run** → cost-accounting hook lost data; run valid but budget signals unusable.
- **All three domains drop on the same commit** → upstream tau2 regression (`~/tau2-bench` git log); uniform *rises* may be the upstream audit, not your intervention.
- **An airline task with a correct-looking trajectory scoring 0.0** → single-solution brittleness (τ² §5). One ≠ systemic; >3 in a round = escalate.

## References

- τ²-bench paper — [arXiv 2506.07982](https://arxiv.org/abs/2506.07982)
- Sierra τ²-bench blog — [sierra.ai/resources/research/tau-squared-bench](https://sierra.ai/resources/research/tau-squared-bench)
- Quesma 2025 — "Improving τ²-bench with smaller models" (decision-tree policy rewrite, +22 pp on GPT-5-mini)
- IRMA — [arXiv 2508.20931](https://arxiv.org/abs/2508.20931)
- Cleanlab 2025 — trust-score message revision on τ-bench
- Artificial Analysis telecom leaderboard — [artificialanalysis.ai/evaluations/tau2-bench](https://artificialanalysis.ai/evaluations/tau2-bench)

## Pointers

- `benchmarks/tau2/README.md` — architecture, env vars, install, HarnessX scores.
- `benchmarks/tau2/tool_filter.py` — `PhaseAwareToolFilter`.
- `benchmarks/tau2/stop_guard.py` — premature-STOP regex + false-positive analysis.
- `benchmarks/tau2/policy_hint.py` — telecom condition detectors + affirmative-hint templates.
- `benchmarks/tau2/harness_config.yaml` — hardened default pipeline.
- `recipe/tau2_evolver/system_append_processor.py` — instruction-layer injection.
- `recipe/tau2_evolver/guidance_retail.md` — reference domain-guidance file.
- `recipe/tau2_evolver/defaults.py` — `NUM_TRIALS`, split, model tiers, evolve-loop constants.
