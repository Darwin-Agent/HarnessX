# GAIA Benchmark

[GAIA](https://arxiv.org/abs/2311.12983) (General AI Assistants) is a deep-research benchmark that
tests multi-step reasoning combined with web retrieval. Every question has a single, deterministic answer.

## Quick start

```bash
# Run the built-in curated cases (no HuggingFace download required)
python -m benchmarks.gaia.run_gaia --model claude-sonnet-4-6

# Level 1 only
python -m benchmarks.gaia.run_gaia --level 1

# First 5 tasks
python -m benchmarks.gaia.run_gaia --max-tasks 5

# Load the full validation set from HuggingFace
python -m benchmarks.gaia.run_gaia --from-hf --level 1 --max-tasks 10

# Custom output file
python -m benchmarks.gaia.run_gaia --output results/gaia_run1.jsonl
```

## Curated cases

8 built-in curated test cases (Level 1–2) covering:

- **Web lookup** — direct factual search
- **Multi-hop reasoning** — multiple search steps chained together
- **Cross-source synthesis** — information gathered from multiple sources
- **Temporal reasoning** — time-sensitive facts

## Evaluation

- Cases with ground truth: exact match (after normalization)
- Complex cases without ground truth: scored by `LLMJudgeEvaluator`

## Output

Results are written as JSONL, one record per task:

```json
{
  "task_id": "curated-01",
  "level": 1,
  "question": "...",
  "expected": "...",
  "agent_output": "...",
  "passed": true,
  "score": 1.0,
  "steps": 5,
  "tokens": 3200,
  "cost_usd": 0.012,
  "elapsed_s": 8.3
}
```

## Dependencies

```bash
# Curated cases — no extra dependencies beyond harnessx itself
pip install harnessx

# Full dataset from HuggingFace
pip install datasets huggingface_hub
```
