# Contributing to HarnessX

Thank you for contributing! Please read the guidelines below before opening a Pull Request.

---

## Branch naming

All changes must go through a feature branch + Pull Request. Direct pushes to `main` are not allowed.

| Scenario | Prefix |
|----------|--------|
| New feature | `feat/<name>` |
| Bug fix | `fix/<name>` |
| Documentation | `docs/<name>` |
| Refactor | `refactor/<name>` |
| Tests | `test/<name>` |

```bash
git checkout -b feat/my-feature
# ... develop, commit ...
# A maintainer reviews and merges into main
```

---

## Development setup

```bash
pip install -e ".[dev]"
```

---

## Tests

### Unit & integration (no model required)

These must all pass before opening a PR:

```bash
python -m pytest tests/unit/ tests/integration/ -q
```

### End-to-end (real model)

E2E tests run the full agent loop against a live LLM. They are optional — run them when your change touches the agent loop, providers, tools, or processors.

**Setup:** copy the example env file and fill in your credentials:

```bash
cp tests/e2e/.env.example tests/e2e/.env
# edit tests/e2e/.env — set ANTHROPIC_API_KEY or OPENAI_API_KEY and a model
```

Minimal `.env` (pick one provider):

```bash
# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_DEFAULT_MAIN_MODEL=claude-haiku-4-5-20251001

# OpenAI-compatible endpoint
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=sk-...
OPENAI_DEFAULT_MAIN_MODEL=openai/your-model
```

Run:

```bash
# Full harness scenarios (trajectory validation)
pytest tests/e2e/test_harness_e2e.py -v -s

# Real-world task suite
pytest tests/e2e/test_real_cases.py -v -s

# All e2e via pytest
pytest tests/e2e/ -v -s
```

See [`tests/e2e/README.md`](tests/e2e/README.md) for the full scenario list and cost notes.

---

## Code conventions

- **No third-party benchmark dependencies in the core library**: all benchmark adapters go in `benchmarks/<name>/` or `recipe/<name>/`; `harnessx/` depends only on its declared dependencies
- **New Processor**: subclass `MultiHookProcessor`, set `_singleton_group` and `_order`, override only the `on_*` hooks you need
- **New Sandbox**: subclass `Sandbox` and `SandboxProvider` from `harnessx/sandbox/base.py`; only `exec()` and `workspace_path` are required — file operations have default implementations
- **New recipe**: goes in `recipe/<name>/` with its own `README.md`; do not add dependencies to `harnessx/`
- **Do not** add comments, type annotations, or docstrings to code you did not change
- **Do not** add abstractions or configuration options for hypothetical future requirements

---

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(sandbox): add OpenSandbox environment adapter
fix(tb2): clear baked-in proxy env vars in exec()
docs: rewrite README.zh.md
refactor(processors): rename PreCompletionChecklistProcessor → SelfVerifyProcessor
```

---

## Pre-PR checklist

- [ ] `pytest tests/unit/ tests/integration/ -q` passes
- [ ] No hardcoded API keys, internal addresses, or tokens
- [ ] New files confirmed to contain no sensitive information
- [ ] `pyproject.toml` updated if new dependencies were added
- [ ] Frontend changes pass `npm run build` (if applicable)
- [ ] New features have corresponding unit tests
- [ ] New recipes under `recipe/` or `benchmarks/` include a `README.md`
- [ ] Commit messages follow the convention above
- [ ] No direct commits to `main`
