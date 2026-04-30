# Model Configuration

HarnessX separates model configuration from harness behavior configuration.

- `ModelConfig`: model registry + role slots + fallback strategy
- `HarnessConfig`: tools / processors / workspace / tracing

They are combined at runtime with `model.agentic(harness_config)`.

---

## Concepts

### Model roles (slots)

`ModelConfig` is a key-to-provider mapping.

| Slot | Purpose |
|---|---|
| `main` | Required primary model |
| `compact` | Optional lighter model for compaction/summarization (falls back to `main`) |
| `judge` | Optional model for evaluation/self-verify (falls back to `main`) |
| custom | Any role key required by processors |

Fallback behavior is controlled by `fallback_key` (default: `main`).

### Provider groups

A role can bind either:

- a single provider
- a provider group with strategy (`primary`, `fallback`, `round_robin`)

This allows per-role redundancy and routing.

---

## Configure in Lab UI

Open **Settings → Model**.

1. Add model definitions (vendor, model id, API key/base URL)
2. Assign models to role slots
3. Save/export YAML as needed

---

## YAML format (v0.1)

`ModelConfig.to_yaml()` writes the current v0.1 format:

```yaml
schema_version: 2
models:
  - id: claude-sonnet-4-6
    provider: anthropic
    _target_: harnessx.providers.anthropic_provider.AnthropicProvider
    model: claude-sonnet-4-6
  - id: gpt-4o-mini
    provider: openai
    _target_: harnessx.providers.litellm_provider.LiteLLMProvider
    model: openai/gpt-4o-mini

roles:
  main:
    default: claude-sonnet-4-6
  compact:
    default: gpt-4o-mini
  judge:
    default: claude-sonnet-4-6
```

`ModelConfig.from_yaml_file()` supports the current v0.1 format and legacy pre-release input.

---

## CLI and runtime behavior

Model config resolution priority for `hx`:

1. `~/.harnessx/model_config.yaml`
2. environment variables

Environment variables commonly used:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `LITELLM_API_KEY`
- `ANTHROPIC_DEFAULT_MAIN_MODEL`
- `OPENAI_DEFAULT_MAIN_MODEL`
- `LITELLM_DEFAULT_MAIN_MODEL`

---

## Capability tags

Capability tags are metadata used by UI and routing logic.

| Tag | Meaning |
|---|---|
| `text` | General text generation |
| `code` | Code generation and analysis |
| `omni` | Multimodal model |
| `vl` | Vision-language |
| `tts` | Text-to-speech |
| `asr` | Speech recognition |
| `embedding` | Embedding model |
| `image_gen` | Image generation |
| `video_gen` | Video generation |
