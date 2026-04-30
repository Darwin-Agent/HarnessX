# LoCoMo

Integration of [LoCoMo](https://github.com/snap-research/locomo) (SNAP Research) into HarnessX.

LoCoMo is a **long-term conversational memory benchmark**. Each sample is a multi-session
dialogue (up to 35 sessions, spanning months) between two speakers. The evaluation asks
500+ QA questions that require the agent to recall facts from the full conversation history,
across five question types: single-hop, multi-hop, temporal reasoning, summarization, and
adversarial (unanswerable).

## Architecture

```
LoCoMo dataset (locomo10.json or HuggingFace snap-research/locomo)
│
├── SessionIngester ──► compressor ──► memory backend
│     Verbatim │ Summary │ Facts │ LightMemory
│
└── QA loop
      ├── light-memory + LLM  →  read_recalled_memories_with_llm()
      │                           + v7f single-message format
      │                           (bypasses Harness pipeline)
      └── other compressors   →  make_locomo_harness()
                                  └── Harness pipeline
                                       ├── SystemPromptProcessor
                                       ├── MemoryRetrievalProcessor
                                       ├── TokenBudgetProcessor
                                       └── UserWrapperProcessor
```

## File structure

```
benchmarks/locomo/
├── __init__.py
├── task.py        # LoCoMoTask, LoCoMoSample, LoCoMoEvaluator (ROUGE-L / token-F1 / exact match)
├── ingester.py    # SessionIngester + compressors (Verbatim, Summary, Facts, LightMemory)
├── harness.py     # make_locomo_harness(), LightMemoryBackend, v7f SYSTEM_PROMPT
├── judge.py       # EverMemOS-aligned LLM-as-Judge (judge_accuracy, compute_aligned_accuracy)
├── run_eval.py    # batch evaluation CLI
└── run.sh         # example run command
```

## Setup

### Prerequisites

- Python >= 3.10
- LLM API credentials (Anthropic or OpenAI-compatible)
- Dataset: local `locomo10.json` or HuggingFace `snap-research/locomo`

### Install dependencies

```bash
uv sync  # or: pip install -e .
```

### Set API credentials

```bash
# Option A: Anthropic proxy (supports extended thinking)
export ANTHROPIC_BASE_URL=https://...
export ANTHROPIC_API_KEY=sk-ant-...

# Option B: LiteLLM / Azure OpenAI
export OPENAI_API_KEY=sk-...
```

## Usage

Run from the repo root with `PYTHONPATH=.`:

```bash
# Verbatim compressor (baseline), first 10 conversations
python benchmarks/locomo/run_eval.py \
    --compressor verbatim \
    --data-path benchmarks/locomo/locomo10.json \
    --max-samples 10 \
    --output reports/locomo_verbatim.jsonl

# Light-memory + LLM ingestion + retrieval (v7f-aligned)
python benchmarks/locomo/run_eval.py \
    --compressor light-memory --llm-memory \
    --model azure_openai/gpt-4.1-mini \
    --memory-model azure_openai/gpt-4.1-mini \
    --top-k 40 --half-life 9999 \
    --data-path benchmarks/locomo/locomo10.json \
    --max-samples 10 \
    --persist-dir .locomo_lm \
    --output reports/locomo_lm.jsonl \
    --resume --workers 4

# With EverMemOS-aligned LLM-as-Judge scoring
python benchmarks/locomo/run_eval.py \
    --compressor light-memory --llm-memory \
    --model azure_openai/gpt-4.1-mini \
    --judge --judge-model azure_openai/gpt-4o-mini --judge-runs 3 \
    --output reports/locomo_judged.jsonl
```

## Compressor strategies

| Strategy | Description | Memory size |
|---|---|---|
| `verbatim` | Every turn stored as `[DATE SN] Speaker: text` | Large |
| `summary` | LLM-generated 3–5 sentence summary per session | Medium |
| `facts` | LLM-extracted bullet facts per session | Small |
| `light-memory` | HarnessX light-memory file store, rule-based | Compact |
| `light-memory` + `--llm-memory` | LLM ingestion + retrieval, org per session | Best quality |

## Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--compressor` | `verbatim` | Session compression strategy |
| `--model` | `openai/pa/claude-haiku-4-5` | QA model |
| `--llm-memory` | off | Use LLM for ingestion and retrieval (light-memory only) |
| `--memory-model` | same as `--model` | Separate model for light-memory LLM paths |
| `--top-k` | `40` | light-memory retrieval candidate count |
| `--half-life` | `365` | Decay half-life in days (`9999` = disabled) |
| `--no-org` | off | Disable per-session organization pass |
| `--persist-dir` | `.locomo_lm` | Directory for light-memory files |
| `--workers` | `1` | Parallel sample workers |
| `--resume` | off | Resume from existing `.jsonl` output |
| `--judge` | off | Enable EverMemOS-aligned LLM-as-Judge |
| `--judge-model` | `azure_openai/gpt-4o-mini` | Judge model |
| `--judge-runs` | `3` | Independent judge invocations per question (majority vote) |
| `--extended-thinking` | off | Enable extended thinking (Anthropic proxy only) |
| `--max-samples` | all | Limit number of conversations |
| `--categories` | all | Comma-separated subset: `single_hop_qa,multi_hop_qa,temporal_reasoning,summarization,adversarial_qa` |

## Scoring

Per-category metrics:

| Category | Metric |
|---|---|
| `single_hop_qa` | Token F1 |
| `multi_hop_qa` | Token F1 |
| `temporal_reasoning` | Token F1 |
| `summarization` | ROUGE-L |
| `adversarial_qa` | Exact match against "I don't know" / "unknown" phrases |

`LoCoMoEvaluator.aggregate()` returns per-category means and an overall average.

With `--judge`, `compute_aligned_accuracy()` additionally reports EverMemOS-aligned accuracy
(LLM-as-Judge, adversarial excluded by default).
