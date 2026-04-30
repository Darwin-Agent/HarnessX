# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .harness import HarnessConfig, Harness


class ModelConfig:
    """Model configuration — key→provider mapping with fallback strategy.

    The ``main`` key is required; all other keys (``judge``, ``evaluator``,
    ``summarize``, …) are optional and fall back to ``main`` when missing.

    Construction::

        # keyword-arg form (most readable)
        ModelConfig(main=AnthropicProvider("claude-sonnet-4-6"))
        ModelConfig(main=p1, judge=p2)

        # dict form (programmatic)
        ModelConfig({"main": p1, "judge": p2})

        # YAML
        ModelConfig.from_yaml_file("model.yaml")
    """

    def __init__(
        self,
        models: "dict[str, Any] | None" = None,
        *,
        fallback_key: str = "main",
        **kwargs: Any,
    ) -> None:
        if models is not None and kwargs:
            raise TypeError("ModelConfig: pass either a dict as the first argument or keyword arguments — not both.")
        raw: dict[str, Any] = dict(models) if models is not None else dict(kwargs)
        if "main" not in raw:
            raise ValueError(
                "ModelConfig requires a 'main' provider. "
                "Example: ModelConfig(main=AnthropicProvider('claude-sonnet-4-6'))"
            )
        self.models: dict[str, Any] = raw
        self.fallback_key: str = fallback_key

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def main(self) -> Any:
        """Primary model provider (the ``main`` key)."""
        return self.models["main"]

    def get(self, key: str) -> Any:
        """Return the provider for *key*, falling back to ``fallback_key`` if absent."""
        return self.models.get(key) or self.models[self.fallback_key]

    # ── Mutation ──────────────────────────────────────────────────────────────

    def copy(self, **overrides: Any) -> "ModelConfig":
        """Return a new ModelConfig with the given keys overridden.

        Example::

            new_model = model_config.copy(main=LiteLLMProvider("openai/gpt-4o"))
        """
        new_models = {**self.models, **overrides}
        return ModelConfig(new_models, fallback_key=self.fallback_key)

    # ── Composition ───────────────────────────────────────────────────────────

    def agentic(self, harness_config: "HarnessConfig") -> "Harness":
        """Combine this model config with a harness config to produce a runnable agent.

        Args:
            harness_config: A fully built HarnessConfig (tools, processors, tracer, …).
                            Must contain NO model information — the model is provided here.

        Returns:
            A :class:`~harnessx.core.harness.Harness` ready for
            ``await agent.run(task)``.
        """
        from .harness import Harness

        return Harness(model_config=self, config=harness_config)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to the current v0.1 format: ``models`` registry + ``roles`` pointer dict.

        Format::

            schema_version: 2
            models:
              - id: claude-sonnet-4-6
                provider: anthropic
                _target_: harnessx.providers.anthropic_provider.AnthropicProvider
                model: claude-sonnet-4-6
                api_key: sk-ant-...
            roles:
              main:
                default: claude-sonnet-4-6
              compact:
                default: claude-sonnet-4-6
                model_ids: [claude-sonnet-4-6, claude-opus-4-6]
                strategy: fallback

        Load with :meth:`from_yaml_file` (supports current v0.1 and legacy pre-release formats).
        """
        from ..providers.group import ProviderGroup as _PG

        ANTHROPIC_TARGET = "harnessx.providers.anthropic_provider.AnthropicProvider"
        LITELLM_TARGET = "harnessx.providers.litellm_provider.LiteLLMProvider"

        models_list: list[dict] = []
        seen: dict[tuple, str] = {}  # (target, model_str) → id

        def _provider_label(target: str, model: str) -> str:
            if "Anthropic" in target:
                return "anthropic"
            prefix = model.split("/")[0].lower() if "/" in model else ""
            return prefix if prefix in ("openai", "gemini", "deepseek") else "litellm"

        def _ensure_model(target: str, model: str, kwargs: dict) -> str:
            """Add provider spec to models_list (deduped) and return its id."""
            key = (target, model)
            if key in seen:
                return seen[key]
            m_id = model.replace("anthropic/", "").replace("openai/", "") or "model"
            existing = {m["id"] for m in models_list}
            base, suffix = m_id, 2
            while m_id in existing:
                m_id = f"{base}-{suffix}"
                suffix += 1
            seen[key] = m_id
            entry: dict = {
                "id": m_id,
                "provider": _provider_label(target, model),
                "_target_": target,
            }
            if model:
                entry["model"] = model
            for k in ("api_key", "api_base", "base_url"):
                if v := kwargs.get(k):
                    entry[k] = v
            models_list.append(entry)
            return m_id

        roles_dict: dict = {}

        for role_name, provider in self.models.items():
            if isinstance(provider, _PG):
                pg = provider.to_dict()
                role_ids: list[str] = []
                default_id: str | None = None
                for entry in pg.get("entries", []):
                    if entry.get("_bare"):
                        continue
                    ptype = entry.get("type", "anthropic")
                    target = ANTHROPIC_TARGET if ptype == "anthropic" else LITELLM_TARGET
                    ekw = {k: v for k, v in entry.items() if k not in ("type", "models", "_bare")}
                    for mspec in entry.get("models", []):
                        m_id = _ensure_model(target, mspec.get("model", ""), ekw)
                        role_ids.append(m_id)
                        if mspec.get("default") and default_id is None:
                            default_id = m_id
                if not default_id and role_ids:
                    default_id = role_ids[0]
                role_entry: dict = {
                    "default": default_id,
                    "model_ids": role_ids,
                    "strategy": pg.get("strategy") or "fallback",
                }
                roles_dict[role_name] = role_entry
            else:
                target = f"{type(provider).__module__}.{type(provider).__qualname__}"
                model = getattr(provider, "model", "") or ""
                kw: dict = {}
                for attr, dest in [
                    ("_api_key", "api_key"),
                    ("api_key", "api_key"),
                    ("_base_url", "api_base"),
                    ("api_base", "api_base"),
                ]:
                    if v := getattr(provider, attr, None):
                        kw.setdefault(dest, v)
                kw.update(
                    {k: v for k, v in (getattr(provider, "kwargs", {}) or {}).items() if k in ("api_key", "api_base")}
                )
                m_id = _ensure_model(target, model, kw)
                roles_dict[role_name] = {"default": m_id}

        result: dict = {
            "schema_version": 2,
            "models": models_list,
            "roles": roles_dict,
        }
        if self.fallback_key != "main":
            result["fallback_key"] = self.fallback_key
        return result

    def to_yaml(self) -> str:
        """Serialize to a YAML string (current v0.1 format: models registry + roles).

        Reload with :meth:`from_yaml` or :meth:`from_yaml_file`.
        Supports legacy pre-release format on load for backward compatibility.
        """
        from omegaconf import OmegaConf

        header = "# HarnessX Model Config v0.1\n# Python: ModelConfig.from_yaml_file(path)\n\n"
        return header + OmegaConf.to_yaml(OmegaConf.create(self.to_dict()))

    def to_yaml_file(self, path: Any) -> None:
        """Write YAML to *path* (creates parent directories if needed)."""
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Restore a ModelConfig from a plain dict.

        Supports two formats:

        **v0.1** (recommended) — ``models`` registry + ``roles`` pointer dict::

            {"schema_version": 2,
             "models": [{"id": "sonnet", "_target_": "...", "model": "...", ...}],
             "roles":  {"main": {"default": "sonnet"}, ...}}

        **legacy pre-release** — role name as top-level key with ``_target_`` inline::

            {"main": {"_target_": "...", "model": "..."}, ...}

        Both formats are detected automatically.
        """
        from omegaconf import DictConfig, OmegaConf

        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)  # type: ignore[arg-type]
        if "models" in d and "roles" in d:
            return cls._from_dict_v2(d)
        return cls._from_dict_v1(d)

    @classmethod
    def _from_dict_v1(cls, d: dict) -> "ModelConfig":
        """Parse legacy pre-release format: each top-level key is a role with ``_target_`` spec."""
        fallback_key = "main"
        providers: dict[str, Any] = {}
        for key, spec in d.items():
            if key in ("schema_version",):
                continue
            if key == "fallback_key":
                fallback_key = str(spec)
                continue
            if isinstance(spec, dict):
                providers[key] = _instantiate_provider(spec)
            else:
                raise ValueError(f"ModelConfig.from_dict: unexpected value type for key '{key}': {type(spec)}")
        return cls(providers, fallback_key=fallback_key)

    @classmethod
    def _from_dict_v2(cls, d: dict) -> "ModelConfig":
        """Parse current v0.1 format: ``models`` registry + ``roles`` pointer dict."""
        # Build raw spec lookup keyed by model id
        specs: dict[str, dict] = {}
        for entry in d.get("models", []):
            m_id = entry["id"]
            # Strip frontend-only metadata (_-prefixed), but keep _target_ which
            # is required by _instantiate_provider. Also drop registry-only fields.
            spec = {
                k: v
                for k, v in entry.items()
                if (k == "_target_" or not k.startswith("_")) and k not in ("id", "provider")
            }
            specs[m_id] = spec

        providers: dict[str, Any] = {}
        fallback_key = str(d.get("fallback_key", "main"))

        for role_name, rcfg in d.get("roles", {}).items():
            default_id = rcfg["default"]
            model_ids = rcfg.get("model_ids") or [default_id]
            strategy = rcfg.get("strategy", "fallback")

            if len(model_ids) <= 1:
                providers[role_name] = _instantiate_provider(specs[default_id])
            else:
                from ..providers.group import ProviderGroup

                entries: list[dict] = []
                for mid in model_ids:
                    sp = specs.get(mid, {})
                    target = sp.get("_target_", "")
                    ptype = "anthropic" if "Anthropic" in target else "litellm"
                    ekw: dict = {
                        "type": ptype,
                        "models": [{"model": sp.get("model", ""), "default": mid == default_id}],
                    }
                    if sp.get("api_key"):
                        ekw["api_key"] = sp["api_key"]
                    if sp.get("api_base"):
                        ekw["api_base"] = sp["api_base"]
                    entries.append(ekw)
                pg_kwargs: dict = {
                    "entries": entries,
                    "max_retries": int(rcfg.get("max_retries", 5)),
                }
                if strategy == "round_robin":
                    pg_kwargs["strategy"] = "round_robin"
                providers[role_name] = ProviderGroup(**pg_kwargs)

        return cls(providers, fallback_key=fallback_key)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ModelConfig":
        """Parse a YAML string into a ModelConfig (supports current v0.1 and legacy pre-release formats)."""
        from omegaconf import OmegaConf

        d = OmegaConf.to_container(OmegaConf.create(yaml_str), resolve=True) or {}
        return cls.from_dict(d)

    @classmethod
    def from_yaml_file(cls, path: Any) -> "ModelConfig":
        """Load a ModelConfig from a YAML file."""
        from pathlib import Path

        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        keys = list(self.models.keys())
        main_model = getattr(self.main, "model", type(self.main).__name__)
        return f"ModelConfig(main={main_model!r}, keys={keys})"


# ── Internal helpers ──────────────────────────────────────────────────────────


def _instantiate_provider(spec: dict) -> Any:
    """Instantiate a provider from a ``_target_``-format dict."""
    if "_target_" not in spec:
        raise ValueError(f"Provider spec missing '_target_': {spec}")

    # to_dict() serialises base_url as "api_base" (LiteLLM convention).
    # AnthropicProvider expects "base_url", not "api_base" — rename on load.
    if "AnthropicProvider" in spec["_target_"] and "api_base" in spec:
        spec = {**spec, "base_url": spec["api_base"]}
        spec = {k: v for k, v in spec.items() if k != "api_base"}

    import importlib

    target = spec["_target_"]
    mod_path, cls_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_path), cls_name)
    kwargs = {k: v for k, v in spec.items() if k != "_target_"}
    return cls(**kwargs)
