# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ModelDefItem(BaseModel):
    id: str
    display_name: str
    vendor: str
    model_id: str
    api_key: str = ""
    base_url: str = ""
    extra_headers: dict[str, str] = {}
    capabilities: list[str] = []
    extended_thinking: bool | None = None
    thinking_budget_tokens: int | None = None
    reasoning_effort: str | None = None
    reasoning_summary: bool | None = None


class ModelSlotItem(BaseModel):
    slot_name: str
    model_ids: list[str]
    strategy: str = "primary"


class ModelConfigResponse(BaseModel):
    registry: list[ModelDefItem]
    slots: list[ModelSlotItem]


def _to_int_or_none(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except Exception:
            return None
    return None


def _vendor_from_target(target: str) -> str:
    if "Anthropic" in target:
        return "anthropic"
    if "OpenAIProvider" in target:
        return "openai"
    return "litellm"


def _target_from_vendor(vendor: str) -> str:
    if vendor == "anthropic":
        return "harnessx.providers.anthropic_provider.AnthropicProvider"
    if vendor == "openai":
        return "harnessx.providers.openai_provider.OpenAIProvider"
    return "harnessx.providers.litellm_provider.LiteLLMProvider"


def _refine_vendor(vendor: str, model: str, *, explicit_provider: bool = False) -> str:
    v = (vendor or "").strip().lower()
    # When provider is explicitly persisted, keep backend choice stable.
    # This is required for openai vs litellm routing correctness in Lab UI.
    if explicit_provider and v in {"anthropic", "openai", "litellm"}:
        return v
    p = (model or "").split("/", 1)[0].lower()
    if p in {"openai", "gemini", "deepseek"}:
        return p
    return v or "litellm"


def _sanitize_registry(registry: list[ModelDefItem]) -> list[ModelDefItem]:
    out: list[ModelDefItem] = []
    seen: set[str] = set()
    for item in registry:
        mid = item.id.strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        vendor = item.vendor.strip() if item.vendor.strip() else "litellm"
        model = item.model_id.strip()
        display = item.display_name.strip() if item.display_name.strip() else (model or mid)
        caps = [c for c in item.capabilities if isinstance(c, str) and c.strip()]
        out.append(
            ModelDefItem(
                id=mid,
                display_name=display,
                vendor=vendor,
                model_id=model,
                api_key=item.api_key.strip(),
                base_url=item.base_url.strip(),
                extra_headers={str(k): str(v) for k, v in (item.extra_headers or {}).items() if str(k).strip()},
                capabilities=caps,
                extended_thinking=(bool(item.extended_thinking) if item.extended_thinking is not None else None),
                thinking_budget_tokens=_to_int_or_none(item.thinking_budget_tokens),
                reasoning_effort=(
                    item.reasoning_effort if item.reasoning_effort in {"low", "medium", "high"} else None
                ),
                reasoning_summary=(bool(item.reasoning_summary) if item.reasoning_summary is not None else None),
            )
        )
    return out


def _sanitize_slots(
    slots: list[ModelSlotItem],
    model_ids: set[str],
) -> list[ModelSlotItem]:
    out: list[ModelSlotItem] = []
    seen_slot_names: set[str] = set()
    for slot in slots:
        name = slot.slot_name.strip()
        if not name or name in seen_slot_names:
            continue
        seen_slot_names.add(name)
        mids = [mid for mid in slot.model_ids if mid in model_ids]
        if not mids:
            continue
        strategy = slot.strategy if slot.strategy in {"primary", "fallback", "round_robin"} else "primary"
        out.append(ModelSlotItem(slot_name=name, model_ids=mids, strategy=strategy))
    return out


def _build_v2_dict(req: ModelConfigResponse) -> dict:
    registry = _sanitize_registry(req.registry)
    if not registry:
        raise HTTPException(status_code=422, detail="model registry is empty")

    # Map frontend random IDs to human-readable IDs (prefer model_id over
    # the auto-generated frontend ID like "m1q2w3e4").  Deduplicate with
    # suffix when two entries share the same model_id.
    id_remap: dict[str, str] = {}
    used_ids: set[str] = set()
    for m in registry:
        readable = (m.model_id.strip() or m.id).strip()
        base = readable
        suffix = 2
        while readable in used_ids:
            readable = f"{base}-{suffix}"
            suffix += 1
        used_ids.add(readable)
        id_remap[m.id] = readable

    id_set = set(id_remap.values())
    # Remap slot model_ids from frontend IDs to readable IDs
    remapped_slots = [
        ModelSlotItem(
            slot_name=s.slot_name,
            model_ids=[id_remap[mid] for mid in s.model_ids if mid in id_remap],
            strategy=s.strategy,
        )
        for s in req.slots
    ]
    slots = _sanitize_slots(remapped_slots, id_set)

    # Always ensure main slot exists and has at least one model.
    first_id = id_remap.get(registry[0].id, registry[0].id) if registry else ""
    main_slot = next((s for s in slots if s.slot_name == "main"), None)
    if main_slot is None:
        slots.insert(0, ModelSlotItem(slot_name="main", model_ids=[first_id], strategy="primary"))
    elif not main_slot.model_ids:
        main_slot.model_ids = [first_id]

    models: list[dict] = []
    for m in registry:
        entry: dict = {
            "id": id_remap.get(m.id, m.id),
            "provider": m.vendor,
            "_target_": _target_from_vendor(m.vendor),
        }
        if m.model_id:
            entry["model"] = m.model_id
        if m.api_key:
            entry["api_key"] = m.api_key
        if m.base_url:
            # Keep one canonical key in yaml; loader maps api_base->base_url for Anthropic.
            entry["api_base"] = m.base_url
        if m.extra_headers:
            if m.vendor == "anthropic":
                entry["default_headers"] = m.extra_headers
            else:
                entry["extra_headers"] = m.extra_headers
        if m.display_name and m.display_name != m.model_id:
            entry["_display_name"] = m.display_name
        if m.capabilities:
            entry["_capabilities"] = m.capabilities
        if m.vendor == "anthropic":
            if m.extended_thinking:
                entry["extended_thinking"] = True
            if m.thinking_budget_tokens and m.thinking_budget_tokens > 0:
                entry["thinking_budget_tokens"] = int(m.thinking_budget_tokens)
        else:
            if m.reasoning_effort:
                entry["reasoning_effort"] = m.reasoning_effort
            if m.reasoning_summary:
                entry["reasoning_summary"] = True
        models.append(entry)

    roles: dict[str, dict] = {}
    for s in slots:
        if not s.model_ids:
            continue
        role_cfg: dict = {"default": s.model_ids[0]}
        if len(s.model_ids) > 1:
            role_cfg["model_ids"] = s.model_ids
            if s.strategy != "primary":
                role_cfg["strategy"] = s.strategy
        roles[s.slot_name] = role_cfg

    return {
        "schema_version": 2,
        "models": models,
        "roles": roles,
    }


def _parse_v2(data: dict) -> ModelConfigResponse:
    registry: list[ModelDefItem] = []
    id_set: set[str] = set()

    for m in data.get("models", []):
        m_id = m.get("id", "")
        if not m_id:
            continue
        target = m.get("_target_", "")
        raw_provider = m.get("provider")
        vendor = _refine_vendor(
            raw_provider or _vendor_from_target(target),
            m.get("model", ""),
            explicit_provider=bool(raw_provider),
        )
        caps = m.get("_capabilities", ["text", "code"])
        if isinstance(caps, str):
            caps = [c.strip() for c in caps.split(",")]
        registry.append(
            ModelDefItem(
                id=m_id,
                display_name=m.get("_display_name") or m.get("model") or m_id,
                vendor=vendor,
                model_id=m.get("model", ""),
                api_key=m.get("api_key", ""),
                base_url=m.get("api_base") or m.get("base_url") or "",
                extra_headers=(
                    m.get("extra_headers")
                    if isinstance(m.get("extra_headers"), dict)
                    else (m.get("default_headers") if isinstance(m.get("default_headers"), dict) else {})
                ),
                capabilities=list(caps),
                extended_thinking=(bool(m.get("extended_thinking")) if "extended_thinking" in m else None),
                thinking_budget_tokens=_to_int_or_none(m.get("thinking_budget_tokens")),
                reasoning_effort=(
                    str(m.get("reasoning_effort")) if isinstance(m.get("reasoning_effort"), str) else None
                ),
                reasoning_summary=(bool(m.get("reasoning_summary")) if "reasoning_summary" in m else None),
            )
        )
        id_set.add(m_id)

    slots: list[ModelSlotItem] = []
    for slot_name, cfg in data.get("roles", {}).items():
        if not isinstance(cfg, dict):
            continue
        default_id = cfg.get("default", "")
        model_ids = cfg.get("model_ids") or ([default_id] if default_id else [])
        strategy = cfg.get("strategy", "primary")
        slots.append(
            ModelSlotItem(
                slot_name=slot_name,
                model_ids=[mid for mid in model_ids if mid in id_set],
                strategy=strategy,
            )
        )

    return ModelConfigResponse(registry=registry, slots=slots)


def _parse_v1(data: dict) -> ModelConfigResponse:
    """Parse legacy pre-release format: each top-level key is a role with _target_ + model + api_key."""
    registry: list[ModelDefItem] = []
    slots: list[ModelSlotItem] = []
    seen: dict[tuple, str] = {}  # (target, model) → id

    def _slug(model_str: str, role: str) -> str:
        s = model_str.replace("anthropic/", "").replace("openai/", "")
        return s or role

    for role_name, cfg in data.items():
        if role_name.startswith("_") or not isinstance(cfg, dict):
            continue
        target = cfg.get("_target_", "")
        if not target:
            continue
        model = cfg.get("model", "")
        api_key = cfg.get("api_key", "")
        base_url = cfg.get("api_base") or cfg.get("base_url") or ""
        vendor = _vendor_from_target(target)
        vendor = _refine_vendor(vendor, model)

        key = (target, model)
        if key not in seen:
            m_id = _slug(model, role_name)
            # deduplicate
            existing = {r.id for r in registry}
            base = m_id
            suf = 2
            while m_id in existing:
                m_id = f"{base}-{suf}"
                suf += 1
            seen[key] = m_id
            registry.append(
                ModelDefItem(
                    id=m_id,
                    display_name=model or role_name,
                    vendor=vendor,
                    model_id=model,
                    api_key=api_key,
                    base_url=base_url,
                    extra_headers=(
                        cfg.get("extra_headers")
                        if isinstance(cfg.get("extra_headers"), dict)
                        else (cfg.get("default_headers") if isinstance(cfg.get("default_headers"), dict) else {})
                    ),
                    capabilities=["text", "code"],
                    extended_thinking=(bool(cfg.get("extended_thinking")) if "extended_thinking" in cfg else None),
                    thinking_budget_tokens=_to_int_or_none(cfg.get("thinking_budget_tokens")),
                    reasoning_effort=(
                        str(cfg.get("reasoning_effort")) if isinstance(cfg.get("reasoning_effort"), str) else None
                    ),
                    reasoning_summary=(bool(cfg.get("reasoning_summary")) if "reasoning_summary" in cfg else None),
                )
            )

        slots.append(
            ModelSlotItem(
                slot_name=role_name,
                model_ids=[seen[key]],
                strategy="primary",
            )
        )

    return ModelConfigResponse(registry=registry, slots=slots)


@router.get("/model-config", response_model=Optional[ModelConfigResponse])
async def get_model_config():
    """Read ~/.harnessx/model_config.yaml and return as frontend-ready registry+slots.

    Returns null (204) if no config file exists.
    """
    from harnessx.home import agent_home
    import yaml as _yaml

    path: Path = agent_home() / "model_config.yaml"
    if not path.exists():
        return None

    try:
        data = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    if "models" in data and "roles" in data:
        return _parse_v2(data)
    return _parse_v1(data)


@router.put("/model-config", response_model=ModelConfigResponse)
async def put_model_config(req: ModelConfigResponse):
    """Persist frontend model registry/slots to ~/.harnessx/model_config.yaml."""
    from harnessx.home import agent_home
    import yaml as _yaml

    data = _build_v2_dict(req)
    yaml_text = "# HarnessX Model Config v0.1\n# Python: ModelConfig.from_yaml_file(path)\n\n" + _yaml.safe_dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )

    path: Path = agent_home() / "model_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_text, encoding="utf-8")

    # Return normalized shape to keep frontend state consistent with persisted file.
    return _parse_v2(data)
