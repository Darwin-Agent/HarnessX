# HarnessX End-to-End Tests

E2E tests run the full agent loop against a real LLM. They are intentionally excluded from `pytest` default discovery (`tests/unit/`, `tests/integration/`) to avoid incurring API cost on every CI run.

## Prerequisites

Copy the example file and fill in your credentials:

```bash
cp tests/e2e/.env.example tests/e2e/.env
# then edit tests/e2e/.env
```

Minimal setup (pick one provider):

```bash
# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_DEFAULT_MAIN_MODEL=claude-sonnet-4-6

# OpenAI / compatible endpoint
OPENAI_API_BASE=http://your-endpoint/v1
OPENAI_API_KEY=sk-...
OPENAI_DEFAULT_MAIN_MODEL=openai/your-model
```

`OPENAI_DEFAULT_MAIN_MODEL` requires a LiteLLM provider prefix (e.g. `openai/gpt-4o`). `ANTHROPIC_DEFAULT_MAIN_MODEL` uses the native Claude model name directly (e.g. `claude-sonnet-4-6`). `EXTRA_HEADERS` is a `"Name: Value"` string (or comma-separated pairs) forwarded on every API call.

### Test workspace (`HXE2E_TEST_HOME`)

`HXE2E_TEST_HOME` sets the `agent_home` used by all e2e tests. Run trajectories, session journals, and config files are written here. The default is `/tmp/hx_test_e2e`.

**This directory is not automatically cleaned up after tests.** Files persist for offline inspection — remove manually when no longer needed:

```bash
rm -rf /tmp/hx_test_e2e   # or whatever HXE2E_TEST_HOME is set to
```

## Running

### Full harness scenarios (trajectory validation)

```bash
pytest tests/e2e/test_harness_e2e.py -v -s
```

Scenarios:
| # | Name | What it tests |
|---|------|---------------|
| 1 | TerminalBench 2.0 | File system tools, Workspace isolation, trajectory data flow |
| 2 | DeepResearch | make_context(), multi-step synthesis |
| 3 | DailyConversation | no-tool chat, FullStateSnapshot immutability |

Each scenario validates trajectory completeness (step count, FullStateSnapshot, StateDelta, training records).

Session and run files are written under `$HXE2E_TEST_HOME/workspaces/{agent_id}/default/` — one directory per test, where `agent_id` matches the test module name (e.g. `harness_e2e`, `real_case_tests`):

```
$HXE2E_TEST_HOME/
└── workspaces/
    ├── harness_e2e/default/sessions/
    ├── real_case_tests/default/sessions/
    └── {test_name}/default/sessions/
```

### Real-world cases (8 tests)

```bash
pytest tests/e2e/test_real_cases.py -v -s
```

Tests:
| Name | Description |
|------|-------------|
| `webpage_creation` | Agent writes HTML, screenshots it with Browser, self-verifies layout |
| `data_analysis` | Synthetic dataset — survival rates + matplotlib chart via Bash |
| `pptx_creation` | 3-slide PPTX via python-pptx, visual self-verification |
| `docx_creation` | DOCX via python-docx |
| `terminal_bench` | Fibonacci via Bash, file I/O verification |
| `deep_research` | WebSearch + WebFetch, structured 3-technique LLM inference report |
| `memory_adapters` | InMemoryMemory / SlidingWindowMemory protocol conformance (no LLM) |
| `api_exports` | All public `harnessx.*` symbols present (no LLM) |

### All e2e via pytest

```bash
pytest tests/e2e/ -v -s
```

The `provider` fixture in `conftest.py` is shared across all pytest-based tests.

## Configuration priority

Provider selection is determined by whichever key group is present in the environment or `tests/e2e/.env`:

| Priority | Keys present | Provider |
|----------|-------------|----------|
| 1 | `ANTHROPIC_API_KEY` or `ANTHROPIC_DEFAULT_MAIN_MODEL` | `AnthropicProvider` |
| 2 | `OPENAI_API_KEY` or `OPENAI_DEFAULT_MAIN_MODEL` | `LiteLLMProvider` |
| 3 | `LITELLM_API_KEY` or `LITELLM_DEFAULT_MAIN_MODEL` | `LiteLLMProvider` |
| 4 | (none set) | `AnthropicProvider("claude-sonnet-4-6")` |

Real env vars take priority over values in `tests/e2e/.env` within each group. You can skip the `.env` file entirely and export vars directly in your shell.

## Cost notes

- `test_harness_e2e.py` cap: `max_cost_usd=1.0` per scenario, `token_budget=50_000`.
- `test_real_cases.py`: each test has a 5-minute timeout enforced by `conftest.py`.
- The two non-model tests (`memory_adapters`, `api_exports`) never call the API.
