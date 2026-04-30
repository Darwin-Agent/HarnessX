---
name: gaia-playbook
description: GAIA-specific benchmark guidance. Public-writeup-sourced techniques (markdown browser, file inspector, code-action agent, planning, multi-agent decomposition, query refinement, answer-format guard, early-commit nudge, prompt caching, majority voting) mapped to HarnessX's four levers (config / control / action / instruction), plus a catalogue of common GAIA failure modes (A-H). Use when forming a hypothesis or scanning for which intervention fits a pattern you see in trajectories.
---

# GAIA: Benchmark-specific Guidance

## Task Structure

Per the GAIA paper's own frequency table, in decreasing order of
how often each capability is needed:

1. **Web browsing** — visit a specific page, not just read a search
   snippet
2. **Code execution** — compute from retrieved data (pandas, math)
3. **File handling** — parse `.pdf`, `.xlsx`, `.csv`, `.docx`,
   `.mp3`, `.mp4`, `.jpg`, `.zip`
4. **Multi-modal understanding** — images, audio, video, PDFs with
   figures
5. **Long-horizon planning** — Level 3 tasks can require 20-50 steps

Common question shapes: "According to <source>, what is <value>?",
"How many X in Y between Z1-Z2?", riddles, reversed/encoded text,
embedded-file computations, YouTube video questions.

## Analyzing Failures

Primary-evidence frontmatter fields (always reliable):

| Field                | What it tells you                                 |
| -------------------- | ------------------------------------------------- |
| `exit_reason`        | `done`, `budget_exceeded`, `loop_detected`, `error` |
| `steps`              | Loop iterations used                              |
| `total_tokens`       | Cost proxy                                        |
| `tool_call_counts`   | Which tools fired, how often                      |
| `tool_error_counts`  | Tool-level failures (≥30% of calls is a red flag)|
| `final_output_length`| Zero means the agent never produced text          |
| `eval_passed`        | Authoritative pass/fail from the external pipeline evaluator |
| `eval_score`         | Authoritative numeric score (0.0 / 1.0 for exact-match) |

The dataset's expected answer and the evaluator's textual reason are both intentionally withheld from the frontmatter — the meta-agent sees only pass/fail and the numeric score. For the *why* behind a failure, read the trajectory body and the optional `judge_*` fields below.

Opinion-only frontmatter fields (present only when the LLM judge processor ran):

| Field                | What it tells you                                 |
| -------------------- | ------------------------------------------------- |
| `extracted_answer`   | What the judge's extractor captured               |
| `judge_verdict`      | Opinion only: `plausible` / `unsupported` /       |
|                      | `hedging` / `format_wrong` / `refused` /          |
|                      | `no_answer` / `judge_error`                       |
| `judge_cause`        | Judge's hypothesis for why the run went wrong     |
| `judge_missing`      | Capability the judge thinks was absent            |
| `judge_lesson`       | One-line takeaway suggested by the judge          |

When `eval_passed` and `judge_verdict` disagree, trust `eval_*` for
pass/fail accounting and use `judge_*` to probe *why*.
`judge_error` is the judge itself failing (timeout, bad JSON), not
the agent — do not build a candidate off those rows.

(Generic trajectory-reading methodology — lens × lever × intent,
three-variant retroactive check, systemic-vs-idiosyncratic filter,
candidate schema — all live in the `analyze` skill. This playbook
only layers GAIA-specific patterns on top. In particular the
"would it pass with unlimited steps + perfect discipline?"
question is just Variant-A retroactive check with GAIA framing.)

### When diagnosis points at a missing capability

GAIA is hard on purpose: many failing tasks would still fail with
more steps, a better prompt, or a tighter loop guard, because the
agent literally does not have the capability the task requires.
These are your highest-leverage opportunities — answer the second
question above honestly ("with unlimited steps and perfect
discipline, would the final state still lack the answer?") and when
the answer is yes, **the default response is to author a new tool**,
not to nudge the model.

Signals that you are in a capability gap (not a behaviour gap):

- `judge_missing_capability.present = true` with a non-empty
  `summary` — the judge is naming the shape of the missing tool for
  you. Use `summary` as a starting sketch, not as a spec.
- `judge_missing` is non-empty and repeats across a cluster of
  failing tasks (e.g. "PDF table parser" on 3+ rows, "JS-rendered
  page browser" on 4+ rows). Repetition is evidence the gap is a
  class, not a one-off.
- Failure modes E (raw bytes), F (missing modality), G (Wayback /
  non-HTML endpoints), and most H (oversized context) post-mortems
  resolve to new tools rather than processors. When you see these,
  go straight to Action-layer candidates.
- The trajectory shows many retries against the same endpoint, or
  the agent paraphrases instead of quoting — both are "tool returned
  nothing usable", not "agent chose badly".

How to act on it:

1. `Read _meta_scratch/CONTEXT.md` when present — it's the machine-
   rendered lever scoreboard from the journal, so you can see at a
   glance whether instruction/control has already been tried on each
   task without lift. Use its hypothesis history as the seed for
   steps 2-4 below.
2. `Read reference` skill in the meta-harness workspace — it has the
   authoring mechanics (shape spec in `TOOL_SPEC.md` before code,
   stdlib-first impl, config wiring, end-of-turn checklist). The
   replay gate that runs after `end_turn` is the integration oracle —
   no self-written probe.
3. Consult the "GAIA capability classes" table later in this playbook
   — match your cluster's `judge_missing_capability` summaries to a
   row, use its "Shape hint" as a framing for your `TOOL_SPEC.md`.
   Do not copy the hint verbatim; your inputs differ from past teams'.
4. Follow the recipe's checklist. Re-run the cluster on the new tool
   and check the gap closed. If `judge_missing_capability.present`
   still fires on those rows, the tool shape was wrong — revise
   before declaring the candidate.

Conversely, do **not** reach for a new tool when:

- `judge_missing_capability.present = false` — judge already said
  the problem is behavioural.
- The answer is present in the trajectory body but the `FINAL
  ANSWER:` marker / format is wrong (mode C). That is a Control
  layer fix.
- A single existing tool already covers the capability and only the
  *instruction* to use it is missing. That is an Instruction lever.

Authoring a tool to dodge a prompt fix inflates the action surface
without lifting the score. Authoring a prompt fix to dodge an
obvious capability gap keeps the benchmark locked at the ceiling of
what the current tools can reach.

## Known Techniques That Improve GAIA Scores

These are generic GAIA wisdom from public writeups by teams scoring
40-75% on the leaderboard. Each is mapped to the lever you'd use in
HarnessX.

| Technique | Layer in HarnessX | Evidence source | Approx. lift |
| --- | --- | --- | --- |
| **Markdown-view web browser** with `visit_page` / `page_down` / `find_in_page` / `scroll_to` (agent sees the rendered page as paginated structured markdown) instead of raw-HTML WebFetch that pollutes context | Action — new tool | HF "beating GAIA" (Autogen browser), H2O | big — was a main lever taking Transformers Agents from mid-pack to #1; see capability class **Stateful browser automation** |
| **Specialized file inspector** with per-extension handlers: `.pdf` → text extract, `.xlsx/.csv` → pandas, `.mp3/.wav` → ASR, `.jpg/.png` → captioner/OCR | Action — new tool | HF "beating GAIA", H2O, hetline | large — many questions depend on attached files that default `read_file` returns as unparsed bytes; see capability classes **File-format parsers** / **Structured spreadsheet reader** / **Audio transcription** / **Vision QA** |
| **Code-Agent style** — agent writes Python snippets (not JSON tool calls) executed in a sandboxed interpreter; loops, math, and dataframe ops compose in one step | Instruction — new template + optional Action tool | HF "beating GAIA", smolagents, `Executable Code Actions Elicit Better LLM Agents` (arXiv 2402.01030) | Python-action agents beat JSON-action on the same model |
| **Planning / fact-list regeneration every N steps** — summarize known facts, rewrite plan; **crucially drop the previous plan from context** so the model re-evaluates rather than rubber-stamping | Control — new `before_model` processor | HF "beating GAIA" (N=2 manager, N=5 sub-agent) | moderate; author note: including the prior plan in context *lowered* the score |
| **Manager + research sub-agent** — manager agent delegates research to a sub-agent that returns only clean summaries; keeps manager context clean | Control (via existing `spawn_subagent`) or Action | HF "beating GAIA", JoyAgent 4-role split | large on multi-hop questions where context cleanliness matters |
| **Multiple search providers wired in** (Google CSE + Bing + DuckDuckGo + Tavily) so one outage doesn't brick the agent | Action — new tool(s) | LinkedIn empirical study, H2O | moderate — `WebSearch` outages are a common failure signal |
| **LLM query refinement** — rewrite the user question into 2-3 optimized search queries before hitting the retrieval backend | Instruction + Control | LinkedIn empirical study | modest; compounds with multi-provider |
| **Forced `FINAL ANSWER:` format guard** — `on_task_end` processor that refuses to stop until marker is present; if missing, triggers one more model turn | Control — new processor | H2O (output verification), GAIA grading contract | +2-4pp on "body has answer, no marker" failures |
| **Early-commit nudge near step budget** — prompt section or `before_model` processor injecting "commit now" when `remaining_steps ≤ 3` | Instruction or Control | well-known fix for budget_exceeded → no_answer | moderate on Level 2-3 tasks with long chains |
| **Prompt caching** on the provider side — GAIA agents re-read the same system prompt every step; caching cuts cost 2-10× without behavior change | Configuration — provider kwargs | H2O (Sonnet 3.5 with prompt caching) | cost-only; no accuracy lift |
| **Majority voting over N samples** (typically 3) — run each task multiple times, take the mode | Control — orchestration | H2O (N=3 at max accuracy) | +3-5pp at 3× cost |
| **Right-size the model per task** — reasoning models (o1/o3, `reasoning_effort=high`) are often overkill for GAIA's light-reasoning questions; non-reasoning variants or `effort=low/minimal` can beat them on latency, cost, and loops | Configuration | hetline 2025-10 (GPT-4o beat GPT-5 on GAIA via fewer graph-depth loops) | context-dependent; worth testing |

**Don't just map failures to existing tools.** The table is a menu
of techniques with known lifts — but GAIA's ceiling on any one
scaffold is set by the tools the scaffold has. Teams that broke
through 60%+ did it by *writing new tools that did not exist before*
(HuggingFace's markdown browser, their per-extension file inspector).
When a cluster of failures points at a missing capability your
current Action list does not cover, the right move is to author a
new tool sized to that cluster — see
`### When diagnosis points at a missing capability` above. Pattern-
matching a hard task to the closest-fitting existing tool is how
benchmarks stall.

## GAIA capability classes that have unblocked past systems

When the journal flags a cluster and you write `_meta_scratch/TOOL_SPEC.md`
(per the `reference` skill's "Before you type — write the shape spec"
section), it helps to know **which tool classes have historically
unblocked top teams on GAIA**. This is
**vocabulary**, not a catalogue to copy from.

The pattern is: match your cluster's `judge_missing_capability`
summaries to one of the classes below, then write a SHAPE from YOUR
failing trajectory inputs. Do not copy implementations from elsewhere
— those were shaped for different input distributions than this run's.

| Capability class | Shape hint (sketch, not spec) | Failure modes it tends to close | Known signal on GAIA |
|------------------|-------------------------------|---------------------------------|----------------------|
| **Code execution sandbox** | `execute(code: str) -> str` — python-only, timeout, captured stdout/stderr | computations from retrieved data; multi-step math; dataframe ops in one tool call; loops over fetched pages | HF codeact / smolagents: large lift on compute-heavy questions |
| **Structured spreadsheet reader** | `read_xlsx(url_or_path: str) -> str` — per-sheet, header-aware, pandas-style | mode E when the attached / fetched file is `.xlsx` / `.csv` — byte-returning reader can't see headers or cells | HF file-inspector: large lift on attachment questions |
| **Symbolic / exact-math calculator** | `compute(expr: str) -> str` — sympy-safe or stdlib Decimal; NOT raw `eval` | rounding errors in LLM-internal math; unit/scale arithmetic; physics formulas | hetline 2025-10: modest but consistent |
| **Stateful browser automation** | `browser(action, selector, …) -> str` — Playwright-backed click / type / screenshot / scroll | JS-rendered pages, session-bound content, multi-page flows — what plain `WebFetch` cannot reach | h2o / HF: mid-single-digit pp |
| **Audio transcription + processing** | `transcribe(url_or_path: str) -> str` — FFmpeg + Whisper, or a hosted ASR API | mode F when the question references audio / video / YouTube URL | HF: a few pp on multimodal Level-2 tasks |
| **Vision QA / VLM reader** | `ask_image(url_or_path, question) -> str` — calls a vision model; returns the answer, not a caption | mode F on chart / figure / scanned questions; image attachments | HF: a few pp on multimodal |
| **Structured web search** | `search(query, provider?) -> str` — multi-provider fallback (Google CSE / Tavily / Bing); ranked URLs + snippets | `WebSearch` outages, unstable snippet extraction, thin top-5 | LinkedIn empirical study: ~2pp + stability |
| **Archive / snapshot resolver** | `snapshot(url: str, date: str) -> str` — Wayback CDX lookup → body of closest capture | mode G when questions reference "the page as of date X" or historical content | h2o: ~2pp on history/archive tasks |
| **Filesystem + shell helper** | (already covered by builtin `Bash` — retune / loosen allowlist before authoring) | one-off text extraction, ad-hoc `curl` to REST endpoints, file format conversion | none — retune existing `Bash` |
| **File-format parsers (per extension)** | `pdf_text(url) -> str` / `docx_text` / `mp3_text` / … — one tool per format, scoped | mode E when answer is inside a specific binary/semi-structured format | HF per-extension file inspector: the single largest observed lift on attachment-heavy tasks |

### How to use this table when writing TOOL_SPEC.md

1. Read your cluster's `judge_missing_capability` summaries from the
   trajectory frontmatter.
2. Skim this table — which row's "Shape hint" most closely matches what
   the summaries describe?
3. Use the matched row's shape as a **starting frame** for your
   `TOOL_SPEC.md`. Fill the spec from your cluster's real input
   examples (per the `reference` skill §"Before you type — write
   the shape spec"), not from this table's hints.
4. If no row matches cleanly — author anyway. This table is not
   exhaustive. GAIA surfaces novel capability gaps regularly; that's
   what evolution is for.

### What this table is NOT

- Not a list of tools to install (do not author all 10)
- Not a checklist ("GAIA needs all classes, let's ship them")
- Not an implementation reference (the "Shape hint" is a sketch, not
  a spec — your real inputs complete it)
- Not a promise the listed lifts will apply to your specific run (past
  teams had different input distributions)

## Common GAIA Failure Modes (from public post-mortems)

Patterns that appear across many different scaffolds and teams —
not tied to any one round's evidence.

### A. Skips the tool, answers from memory

Agent reads the question, believes it knows the answer from training
data, emits it without calling any tool. Often wrong on recent or
long-tail facts.

- **Signal**: very low step count + wrong answer
- **Fix direction**: `before_model` processor requiring ≥1
  information-gathering tool call before an answer is accepted;
  or instruction-layer rule "never answer a factual question without
  a tool call"
- **Not this mode when**: the agent's single tool call returned
  nothing useful and the answer came from training-data recall as
  a fallback — that is a capability or retrieval gap, not an
  over-confidence pattern. A forced tool call won't help if the
  tool it forces has no data to return.

### B. Search-snippet answer instead of primary source

Google/Bing snippet contains a non-canonical form (diminutive,
truncated, translated, abbreviated); agent commits it verbatim.

- **Signal**: passes on easy questions but misses exact-match on
  named entities
- **Fix direction**: instruction rule "proper-noun answers must come
  from a WebFetch of the primary page, not from a snippet"; or a
  control-layer guard that blocks commit when the extracted answer
  is a proper noun and no WebFetch to the canonical source was made

### C. Deliberation without commitment

Agent finds a plausible answer mid-run, keeps searching for
confirmation, hits `max_steps` with no `FINAL ANSWER:` emitted.

- **Signal**: `exit_reason=budget_exceeded`, `extracted_answer` ends
  mid-sentence or is a question
- **Fix direction**: late-step commit nudge (see table above)
- **Not this mode when**: the trajectory's final output has no
  answer at all because retrieved data was unparseable or absent
  (raw bytes, empty results, HTTP errors, endless redirects). A
  commit-nudge cannot help if there is nothing to commit — that
  is an upstream capability or retrieval failure. The tell: scan
  the last 30-40 lines of the body; if the agent is still saying
  "let me try X" / "the page shows..." without any concrete
  answer-candidate, do not apply this mode.

### D. Recursion / graph-depth loops

Reasoning models especially tend to re-enter the same subproblem,
looping until a depth limit terminates them without output.

- **Signal**: many repeated tool calls with tiny variations;
  `exit_reason=loop_detected` or `budget_exceeded` with no committed
  answer
- **Fix direction**: tighter `LoopDetectionProcessor` matching on
  normalized tool input; or `reasoning_effort=minimal` / swap to a
  non-reasoning model
- **Not this mode when**: the loop is the consequence of an
  earlier tool returning no useful output. The agent keeps
  retrying because it has no progress signal to move past — a
  tighter loop detector will just end the run sooner without the
  answer. The lever is one step upstream: whatever the retried
  tool was meant to return, it did not.

### E. Raw bytes from file / PDF

`read_file` / `WebFetch` returns bytes or raw HTML; agent
"describes" the content vaguely ("the document discusses...")
without actually parsing it.

- **Signal**: the task expects a specific value found inside a
  binary or semi-structured document (PDF, XLSX, image, audio).
  The agent's body shows paraphrase rather than quoted content,
  or many retries against the same URL. This does NOT require
  the question to name a file attachment — a question that ends
  up pointing at a document still qualifies.
- **Fix direction**: `after_tool` processor that parses PDF/XLSX
  bytes transparently; or a specialized file-inspector tool
- **Not this mode when**: the document was read successfully and
  the error is in how the agent interpreted clearly-parsed text
  (wrong section, arithmetic error, misread table). That is a
  reasoning problem, not a parsing one.

### F. Missing modality

Question needs video or audio understanding; agent tries
transcript-only; misses visual info.

- **Signal**: YouTube / video tasks fail with empty or
  confident-wrong answers
- **Fix direction**: download + frame-extraction tool, or explicit
  fallback rule ("if transcript unavailable, report it and stop; do
  not guess")

### G. Wayback Machine / paywalled content

Playwright-based `WebFetch` fails on non-HTML endpoints like the
Wayback CDX API or paywalled paper PDFs. Agent retries then guesses.

- **Signal**: `tool_error_counts` includes `WebFetch`; body shows
  archive.org or paywalled URLs
- **Fix direction**: use `curl` via `Bash` for REST/API endpoints
  (bypasses Playwright); or dedicated `wayback_fetch` tool
- **Not this mode when**: `WebFetch` succeeded and returned
  content, but the content is a CAPTCHA, an access-wall page, or
  a partial snippet rather than the actual document. That is
  mode E (needs a parser / different fetch path), not a tooling
  mismatch with Playwright.

### H. Oversized context pollution

Agent fetches a 200KB page, stuffs the full text into context,
loses focus.

- **Signal**: `total_tokens` spikes on 1-2 tasks without
  proportional result
- **Fix direction**: markdown-view browser that paginates, or a
  sub-agent that summarizes before returning
- **Not this mode when**: the large fetch was the agent's only
  path to the data and the content is legitimately relevant. A
  markdown browser or sub-agent summarizer is the real fix, not
  context compaction — compressing the only useful context makes
  the failure silent rather than preventing it.

## Red Flags — These Are Pipeline Bugs, Not Agent Bugs

Do not build an agent-layer candidate off these; record in
`_meta_scratch/NEEDS_FROM_HUMAN.md` instead:

- `extracted_answer == ""` but body contains non-trivial text →
  answer-extractor regex bug
- `judge_verdict: judge_error` → judge failed (timeout, bad JSON)
- `exit_reason: error` with no traceback anywhere in the body → run
  loop swallowed an exception
