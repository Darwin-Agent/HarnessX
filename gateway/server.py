from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

if TYPE_CHECKING:
    from .core.dispatch import ChannelDispatcher
    from .core.cron import CronManager

logger = logging.getLogger(__name__)

gateway_router = APIRouter()

_dispatcher: "ChannelDispatcher | None" = None
_config_path: Path | None = None
_cron_manager: "CronManager | None" = None


def set_dispatcher(d: "ChannelDispatcher") -> None:
    global _dispatcher
    _dispatcher = d


def set_config_path(path: Path) -> None:
    global _config_path
    _config_path = path


def get_dispatcher() -> "ChannelDispatcher":
    if _dispatcher is None:
        raise HTTPException(status_code=503, detail="Gateway not initialized — start with `python -m gateway`")
    return _dispatcher


def set_cron_manager(cm: "CronManager") -> None:
    global _cron_manager
    _cron_manager = cm


def get_cron_manager() -> "CronManager":
    if _cron_manager is None:
        raise HTTPException(status_code=503, detail="Cron manager not available")
    return _cron_manager


# ── Config file helpers ───────────────────────────────────────────────────────


def _get_config_path() -> Path:
    return _config_path or (Path.home() / ".harnessx" / "gateway.yaml")


def _load_raw_config() -> dict:
    p = _get_config_path()
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to load config from %s: %s", p, e)
        return {}


def _save_raw_config(config: dict) -> None:
    p = _get_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        os.replace(tmp, p)
    except Exception as e:
        logger.error("Failed to save config to %s: %s", p, e)
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")


# ── Webhook ───────────────────────────────────────────────────────────────────


@gateway_router.post("/gateway/webhook/{channel_name}")
async def webhook(channel_name: str, request: Request) -> Response:
    """Unified webhook ingress. Signature is verified before any processing."""
    dispatcher = get_dispatcher()
    channel = dispatcher.get_channel(channel_name)
    if channel is None:
        raise HTTPException(status_code=404, detail=f"Channel '{channel_name}' not found")

    body = await request.body()
    headers = dict(request.headers)

    try:
        if not channel.verify_webhook(headers, body):
            logger.warning("[%s] webhook signature verification failed", channel_name)
            raise HTTPException(status_code=403, detail="Invalid signature")
    except NotImplementedError:
        pass  # Channel doesn't implement verify_webhook; allow through

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # URL verification challenges (feishu / slack)
    if "challenge" in payload:
        return Response(content=json.dumps({"challenge": payload["challenge"]}), media_type="application/json")
    if payload.get("type") == "url_verification":
        return Response(content=json.dumps({"challenge": payload.get("challenge", "")}), media_type="application/json")

    try:
        webhook_result = await channel._on_webhook(payload)
    except Exception as e:
        logger.error("[%s] webhook handler error: %s", channel_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")

    body = webhook_result if webhook_result is not None else {"ok": True}
    return Response(content=json.dumps(body), media_type="application/json")


# ── Channel Management API ────────────────────────────────────────────────────


class ChannelInfo(BaseModel):
    name: str
    display_name: str
    enabled: bool
    connection_state: str


class ChannelConfigUpdate(BaseModel):
    config: dict


def _display_name_for(channel_name: str) -> str:
    """Best-effort display name: try registry, fall back to capitalize."""
    try:
        from .channels import get_registry

        cls = get_registry().get(channel_name)
        if cls:
            return getattr(cls, "display_name", channel_name.capitalize())
    except Exception:
        pass
    return channel_name.capitalize()


@gateway_router.get("/gateway/channels", response_model=list[ChannelInfo])
async def list_channels() -> list[ChannelInfo]:
    """List all configured+enabled channels merged with live dispatcher state."""
    running: dict[str, str] = {}
    if _dispatcher is not None:
        running = {ch.name: ch.connection_state for ch in _dispatcher.channels}

    config = _load_raw_config()
    configured: dict[str, dict] = config.get("channels", {})

    result: list[ChannelInfo] = []
    seen: set[str] = set()
    for name, cfg in configured.items():
        seen.add(name)
        if not cfg.get("enabled", False):
            continue
        state = running.get(name, "offline")
        result.append(
            ChannelInfo(
                name=name,
                display_name=cfg.get("display_name") or _display_name_for(name),
                enabled=True,
                connection_state=state,
            )
        )

    # Channels in dispatcher but not in config (defensive)
    for name, state in running.items():
        if name not in seen:
            result.append(
                ChannelInfo(
                    name=name,
                    display_name=_display_name_for(name),
                    enabled=True,
                    connection_state=state,
                )
            )

    return result


@gateway_router.get("/gateway/channels/{name}/config")
async def get_channel_config(name: str) -> dict:
    """Return channel config (sensitive fields masked). Falls back to gateway.yaml for offline channels."""
    # Try running channel first
    if _dispatcher is not None:
        ch = _dispatcher.get_channel(name)
        if ch:
            cfg = dict(ch.config)
            for field_name, schema in ch.config_schema.get("properties", {}).items():
                if schema.get("format") == "password" and field_name in cfg:
                    cfg[field_name] = "***"
            return {"name": name, "config": cfg, "schema": ch.config_schema}

    # Fall back to gateway.yaml + registry schema
    raw = _load_raw_config()
    ch_cfg = raw.get("channels", {}).get(name)
    if ch_cfg is None:
        raise HTTPException(status_code=404, detail=f"Channel '{name}' not found")
    schema: dict = {}
    try:
        from .channels import get_registry, _auto_discover

        _auto_discover()
        platform_type = ch_cfg.get("channel_type", name)
        cls = get_registry().get(platform_type)
        if cls is None:
            # Last-resort: scan the module directly (handles channel_type not yet in registry)
            import importlib
            from .core.base_channel import BaseChannel

            dir_name = (
                platform_type + "_"
                if not (Path(__file__).parent / "channels" / platform_type).exists()
                else platform_type
            )
            try:
                mod = importlib.import_module(f"gateway.channels.{dir_name}.channel")
                cls = next(
                    (
                        v
                        for v in vars(mod).values()
                        if isinstance(v, type) and issubclass(v, BaseChannel) and v is not BaseChannel
                    ),
                    None,
                )
            except Exception:
                pass
        if cls:
            schema = getattr(cls, "config_schema", {})
    except Exception:
        pass
    cfg = dict(ch_cfg)
    for field_name, field_schema in schema.get("properties", {}).items():
        if field_schema.get("format") == "password" and field_name in cfg:
            cfg[field_name] = "***"
    return {"name": name, "config": cfg, "schema": schema}


@gateway_router.put("/gateway/channels/{name}/config")
async def update_channel_config(name: str, body: ChannelConfigUpdate) -> dict:
    """Update running channel config AND persist to gateway.yaml."""
    # Strip password mask sentinel — "***" means "leave unchanged"
    password_fields: set[str] = set()
    if _dispatcher is not None:
        ch = _dispatcher.get_channel(name)
        if ch:
            password_fields = {
                k for k, v in ch.config_schema.get("properties", {}).items() if v.get("format") == "password"
            }
    sanitized = {k: v for k, v in body.config.items() if not (k in password_fields and v == "***")}

    # Update running channel if dispatcher is active
    if _dispatcher is not None:
        ch = _dispatcher.get_channel(name)
        if ch:
            ch.config.update(sanitized)

    # Also persist to gateway.yaml
    config = _load_raw_config()
    channels = config.setdefault("channels", {})
    if name in channels:
        channels[name].update(sanitized)
    else:
        channels[name] = {**sanitized, "enabled": True}
    _save_raw_config(config)

    return {"ok": True, "message": "Config saved. Restart the channel to apply changes."}


@gateway_router.get("/gateway/channels/{name}/status")
async def get_channel_status(name: str) -> dict:
    """Return detailed channel status. Returns offline status for configured-but-not-running channels."""
    if _dispatcher is not None:
        ch = _dispatcher.get_channel(name)
        if ch:
            q = _dispatcher._channel_queues.get(name)
            return {
                "name": name,
                "connection_state": ch.connection_state,
                "queue_size": q.qsize() if q else 0,
                "active_sessions": sum(1 for sid in _dispatcher._session_last_active if sid.startswith(f"{name}:")),
            }

    # Channel is configured but not in dispatcher (offline or gateway not started)
    raw = _load_raw_config()
    if name not in raw.get("channels", {}):
        raise HTTPException(status_code=404, detail=f"Channel '{name}' not found")
    return {"name": name, "connection_state": "offline", "queue_size": 0, "active_sessions": 0}


@gateway_router.post("/gateway/channels/{name}/restart")
async def restart_channel(name: str) -> dict:
    """Hot-restart a channel: stop the existing instance and start a fresh one."""
    raw = _load_raw_config()
    ch_cfg = raw.get("channels", {}).get(name)
    if ch_cfg is None:
        raise HTTPException(status_code=404, detail=f"Channel '{name}' not found in config")
    if not ch_cfg.get("enabled", False):
        raise HTTPException(status_code=400, detail=f"Channel '{name}' is not enabled")

    dispatcher = get_dispatcher()

    # Stop existing instance (if any)
    await dispatcher.stop_channel(name)

    # Re-instantiate from saved config
    from .channels import get_channel_class, _auto_discover

    _auto_discover()
    platform_type = ch_cfg.get("channel_type", name)
    cls = get_channel_class(platform_type)
    if cls is None:
        raise HTTPException(
            status_code=400,
            detail=f"Channel type '{platform_type}' not found in registry",
        )

    ch = cls(config=ch_cfg, dispatcher=dispatcher)
    ch.name = name

    # Rebuild per-channel harness so the fresh instance uses the correct workspace/sessions dir
    from .main import _build_harness, _load_model_config

    gw_cfg = raw.get("gateway", {})
    gw_agent_id = gw_cfg.get("agent_id", "gateway")
    default_cfg = raw.get("default", {})
    try:
        harness = _build_harness(ch_cfg, default_cfg, _load_model_config(), agent_id=gw_agent_id, channel_name=name)
        dispatcher._channel_harnesses[name] = harness
    except Exception as e:
        logger.warning("Failed to rebuild harness for '%s': %s", name, e)

    await dispatcher.start_channel(ch)
    logger.info("Restarted channel: %s", name)
    return {"ok": True, "message": f"Channel '{name}' restarted."}


@gateway_router.post("/gateway/channels/{name}/reset_session")
async def reset_channel_session(name: str, request: Request) -> dict:
    """Reset a specific session epoch (admin action)."""
    dispatcher = get_dispatcher()
    ch = dispatcher.get_channel(name)
    if not ch:
        raise HTTPException(status_code=404, detail=f"Channel '{name}' not found")
    body = await request.json()
    session_id = body.get("session_id", "")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    base = session_id.split("#")[0]
    new_epoch = dispatcher._session_epochs.get(base, 0) + 1
    dispatcher._session_epochs[base] = new_epoch
    dispatcher._session_store.save_epoch(base, new_epoch)
    return {"ok": True, "base_session_id": base, "new_epoch": new_epoch}


@gateway_router.post("/gateway/pairing/generate")
async def generate_pairing_code(request: Request) -> dict:
    """Generate a new pairing code for a channel."""
    body = await request.json()
    channel_name = body.get("channel", "")
    dispatcher = get_dispatcher()
    auth = dispatcher._auth.get(channel_name)
    if not auth:
        raise HTTPException(
            status_code=400,
            detail=f"Channel '{channel_name}' does not use pairing auth mode",
        )
    try:
        code = auth.generate_code()
        return {"code": code, "ttl_seconds": 3600}
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))


@gateway_router.get("/gateway/health")
async def health() -> dict:
    """Health check endpoint."""
    if _dispatcher is None:
        return {"ok": False, "channels": {}, "status": "initializing"}
    states = {ch.name: ch.connection_state for ch in _dispatcher.channels}
    all_ok = all(s == "online" for s in states.values()) if states else True
    return {"ok": all_ok, "channels": states}


# ── Gateway config API ────────────────────────────────────────────────────────


@gateway_router.get("/gateway/config")
async def get_gateway_config() -> dict:
    """Return gateway-level parameters (agent_id, host, port, etc.)."""
    config = _load_raw_config()
    return {"gateway": config.get("gateway", {})}


class GatewayConfigUpdate(BaseModel):
    gateway: dict


@gateway_router.put("/gateway/config")
async def update_gateway_config(body: GatewayConfigUpdate) -> dict:
    """Persist gateway-level parameters to gateway.yaml."""
    config = _load_raw_config()
    config["gateway"] = body.gateway
    _save_raw_config(config)
    return {"ok": True}


# ── Channel types (available platforms) ──────────────────────────────────────


@gateway_router.get("/gateway/channel-types")
async def list_channel_types() -> list[dict]:
    """List all available channel platform types with their config schemas."""
    import importlib
    from .core.base_channel import BaseChannel

    channels_dir = Path(__file__).parent / "channels"
    result = []
    for path in sorted(channels_dir.iterdir()):
        if not path.is_dir() or not (path / "channel.py").exists():
            continue
        name = path.name
        try:
            mod = importlib.import_module(f"gateway.channels.{name}.channel")
            cls = next(
                (
                    v
                    for v in vars(mod).values()
                    if isinstance(v, type) and issubclass(v, BaseChannel) and v is not BaseChannel
                ),
                None,
            )
            if cls:
                result.append(
                    {
                        "name": getattr(cls, "name", name),  # platform type name (e.g. "discord")
                        "display_name": getattr(cls, "display_name", name.capitalize()),
                        "schema": getattr(cls, "config_schema", {}),
                        "available": True,
                    }
                )
        except ImportError as e:
            result.append(
                {
                    "name": name,
                    "display_name": name.capitalize(),
                    "schema": {},
                    "available": False,
                    "missing_dep": str(e),
                }
            )
        except Exception as e:
            logger.warning("Failed to inspect channel '%s': %s", name, e)
    return result


# ── Channel CRUD (config-file level) ─────────────────────────────────────────


class ChannelCreateBody(BaseModel):
    name: str
    channel_type: str = ""  # platform type (e.g. "discord"); falls back to name
    config: dict


@gateway_router.post("/gateway/channels/create")
async def create_channel(body: ChannelCreateBody) -> dict:
    """Add or replace a channel entry in gateway.yaml."""
    name = body.name.strip()
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid channel name (alphanumeric, _ and - only)")
    config = _load_raw_config()
    channels = config.setdefault("channels", {})
    entry: dict = {**body.config, "enabled": body.config.get("enabled", True)}
    # Store channel_type only when the instance name differs from the platform type
    platform_type = body.channel_type.strip() or name
    if platform_type != name:
        entry["channel_type"] = platform_type
    channels[name] = entry
    _save_raw_config(config)
    return {"ok": True, "message": "Channel saved. Restart gateway to apply."}


@gateway_router.delete("/gateway/channels/{name}")
async def delete_channel(name: str) -> dict:
    """Remove a channel entry from gateway.yaml."""
    config = _load_raw_config()
    channels = config.get("channels", {})
    if name not in channels:
        raise HTTPException(status_code=404, detail=f"Channel '{name}' not found in config")
    del channels[name]
    _save_raw_config(config)
    return {"ok": True}


# ── Sessions listing (from on-disk workspace) ─────────────────────────────────


def _read_first_query(session_dir: Path, run_ids: list[str]) -> str:
    """Extract the first user message from the session's JSONL traces."""
    for run_id in run_ids:
        jsonl = session_dir / f"{run_id}.jsonl"
        if not jsonl.exists():
            continue
        try:
            with open(jsonl, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    t = rec.get("type")
                    if t == "session_start":
                        task = (rec.get("task") or "").strip()
                        if task:
                            return task[:200]
                    elif t in ("user", "raw_user"):
                        msg = rec.get("message") or {}
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                        text = (content or rec.get("task") or "").strip()
                        if text:
                            return text[:200]
        except Exception:
            pass
    return ""


@gateway_router.get("/gateway/sessions")
async def list_gateway_sessions(channel: str | None = None) -> list[dict]:
    """List IM sessions from the gateway workspace directory."""
    try:
        from harnessx.home import agent_home

        home = agent_home()
    except Exception:
        home = Path.home() / ".harnessx"

    gw_cfg = _load_raw_config().get("gateway", {})
    gw_agent_id = gw_cfg.get("agent_id", "gateway")
    workspace_base = home / "im-workspaces" / gw_agent_id
    if channel:
        channels_to_scan = [channel]
    elif workspace_base.exists():
        channels_to_scan = [d.name for d in workspace_base.iterdir() if d.is_dir()]
    else:
        channels_to_scan = []

    sessions: list[dict] = []
    for ch_name in sorted(channels_to_scan):
        sessions_dir = workspace_base / ch_name / "sessions"
        if not sessions_dir.exists():
            continue
        for idx_file in sorted(
            sessions_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        ):
            try:
                with open(idx_file, encoding="utf-8") as f:
                    idx = json.load(f)
                sid = idx.get("session_id", idx_file.stem)
                run_ids = idx.get("run_ids", [])
                session_dir = sessions_dir / sid
                sessions.append(
                    {
                        "session_id": sid,
                        "channel": ch_name,
                        "agent_id": gw_agent_id,
                        "project": ch_name,
                        "first_query": _read_first_query(session_dir, run_ids),
                        "updated_at": idx.get("updated_at", ""),
                        "run_count": len(run_ids),
                    }
                )
            except Exception:
                pass

    return sessions


# ── Cron jobs ─────────────────────────────────────────────────────────────────


@gateway_router.get("/gateway/cron/jobs")
async def cron_list_jobs() -> list[dict]:
    return get_cron_manager().list_jobs()


@gateway_router.post("/gateway/cron/jobs")
async def cron_create_job(request: Request) -> dict:
    body = await request.json()
    if not body.get("name"):
        raise HTTPException(status_code=400, detail="'name' is required")
    if not body.get("cron") and not body.get("every"):
        raise HTTPException(status_code=400, detail="'cron' or 'every' is required")
    job = get_cron_manager().create_job(body)
    return {"ok": True, "id": job.id}


@gateway_router.get("/gateway/cron/jobs/{job_id}")
async def cron_get_job(job_id: str) -> dict:
    job = get_cron_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@gateway_router.put("/gateway/cron/jobs/{job_id}")
async def cron_update_job(job_id: str, request: Request) -> dict:
    body = await request.json()
    job = get_cron_manager().update_job(job_id, body)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {"ok": True, "id": job.id}


@gateway_router.delete("/gateway/cron/jobs/{job_id}")
async def cron_delete_job(job_id: str) -> dict:
    if not get_cron_manager().delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {"ok": True}


@gateway_router.post("/gateway/cron/jobs/{job_id}/run")
async def cron_run_job_now(job_id: str) -> dict:
    if not await get_cron_manager().run_now(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {"ok": True, "message": "Job triggered"}


@gateway_router.post("/gateway/cron/jobs/{job_id}/pause")
async def cron_pause_job(job_id: str) -> dict:
    if not get_cron_manager().pause_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {"ok": True}


@gateway_router.post("/gateway/cron/jobs/{job_id}/resume")
async def cron_resume_job(job_id: str) -> dict:
    if not get_cron_manager().resume_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {"ok": True}


@gateway_router.get("/gateway/heartbeat")
async def get_heartbeat_state() -> dict:
    state = get_cron_manager().get_heartbeat_state()
    if state is None:
        return {"enabled": False}
    return state


@gateway_router.get("/gateway/heartbeat/config")
async def get_heartbeat_config() -> dict:
    """Read the heartbeat section from gateway.yaml."""
    return _load_raw_config().get("heartbeat") or {"enabled": False}


@gateway_router.put("/gateway/heartbeat/config")
async def update_heartbeat_config(request: Request) -> dict:
    """Save heartbeat section to gateway.yaml and hot-reload the scheduler."""
    body = await request.json()
    config = _load_raw_config()
    config["heartbeat"] = body
    _save_raw_config(config)
    # Hot-reload running heartbeat without gateway restart
    try:
        await get_cron_manager().reload_heartbeat(body)
    except Exception as e:
        logger.warning("Failed to reload heartbeat: %s", e)
    return {"ok": True}


# ── Gateway docs (docs/gateway/) ─────────────────────────────────────────────

_GW_DOCS_BASE = Path(__file__).parents[1] / "docs" / "gateway"
_GW_DOCS_LANGS = {"zh", "en"}


def _gw_docs_root(lang: str) -> Path:
    """Return docs/gateway/{lang}/, falling back to zh if lang dir missing."""
    p = _GW_DOCS_BASE / lang
    return p if p.is_dir() else _GW_DOCS_BASE / "zh"


class GwDocEntry(BaseModel):
    path: str
    title: str


class GwDocSection(BaseModel):
    name: str
    items: list[GwDocEntry]


class GwDocTree(BaseModel):
    sections: list[GwDocSection]


class GwDocContent(BaseModel):
    path: str
    title: str
    content: str


def _gw_first_h1(md_file: Path) -> str:
    try:
        for line in md_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
    except Exception:
        pass
    return md_file.stem.replace("-", " ").replace("_", " ").title()


_GW_DOCS_SECTION_ORDER = ["channels"]
_GW_DOCS_SECTION_LABELS: dict[str, str] = {"channels": "Channels"}


@gateway_router.get("/gateway/docs", response_model=GwDocTree)
async def gateway_doc_tree(lang: str = "zh") -> GwDocTree:
    """Return docs/gateway/{lang}/ document tree grouped by subdirectory."""
    root = _gw_docs_root(lang)
    if not root.is_dir():
        return GwDocTree(sections=[])

    sections_map: dict[str, list[GwDocEntry]] = {}
    root_items: list[GwDocEntry] = []

    for md_file in sorted(root.rglob("*.md")):
        rel = md_file.relative_to(root)
        parts = rel.parts
        path = str(rel.with_suffix(""))
        entry = GwDocEntry(path=path, title=_gw_first_h1(md_file))
        if len(parts) == 1:
            root_items.append(entry)
        else:
            sections_map.setdefault(parts[0], []).append(entry)

    sections: list[GwDocSection] = []
    if root_items:
        sections.append(GwDocSection(name="Gateway", items=root_items))
    for key in _GW_DOCS_SECTION_ORDER:
        if key in sections_map:
            label = _GW_DOCS_SECTION_LABELS.get(key, key.title())
            sections.append(GwDocSection(name=label, items=sections_map[key]))
    for key, items in sections_map.items():
        if key not in _GW_DOCS_SECTION_ORDER:
            sections.append(GwDocSection(name=key.title(), items=items))

    return GwDocTree(sections=sections)


@gateway_router.get("/gateway/docs/{path:path}", response_model=GwDocContent)
async def gateway_doc_content(path: str, lang: str = "zh") -> GwDocContent:
    """Return raw markdown for a docs/gateway/{lang}/ entry."""
    clean = path.strip("/")
    if ".." in clean:
        raise HTTPException(400, "Invalid path")
    root = _gw_docs_root(lang)
    md_file = (root / clean).with_suffix(".md")
    if not md_file.is_file():
        raise HTTPException(404, f"Doc not found: {path}")
    try:
        content = md_file.read_text(encoding="utf-8")
    except Exception:
        raise HTTPException(500, "Failed to read doc")
    return GwDocContent(path=clean, title=_gw_first_h1(md_file), content=content)


# ── Console chat (web_ui channel) ────────────────────────────────────────────


class ConsoleRunRequest(BaseModel):
    message: str
    session_id: str | None = None


@gateway_router.post("/gateway/console/run")
async def console_run(body: ConsoleRunRequest) -> dict:
    """Start a harness run for the console chat (project=web_ui).

    Registers the run in the shared SSE store so GET /api/run/{id}/stream works.
    Returns { run_id, session_id }.
    """
    from harnessx.api.routes.run import _runs, _run_tasks
    from harnessx.api.sse_tracer import SSETracer, _sse
    from harnessx import BaseTask

    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message required")

    gw_cfg = _load_raw_config().get("gateway", {})
    gw_agent_id = gw_cfg.get("agent_id", "gateway")

    run_id = str(uuid.uuid4())
    session_id = body.session_id or f"web_ui:{uuid.uuid4()}"
    queue: asyncio.Queue = asyncio.Queue()
    _runs[run_id] = queue

    async def _run() -> None:
        try:
            from .main import _build_harness, _load_model_config

            base = _build_harness({}, {}, _load_model_config(), agent_id=gw_agent_id, channel_name="web_ui")
            sse_tracer = SSETracer(queue=queue, inner=base.config.tracer, api_run_id=run_id)
            harness_config = base.config.copy(tracer=sse_tracer)
            harness = base.model_config.agentic(harness_config)

            task = BaseTask(description=body.message.strip())

            def stream_cb(delta: object) -> None:
                kind = "token"
                content = ""
                if isinstance(delta, str):
                    content = delta
                elif isinstance(delta, dict):
                    raw_kind = delta.get("type") or delta.get("kind")
                    raw_content = delta.get("content") or delta.get("delta")
                    if isinstance(raw_kind, str) and raw_kind in {"token", "thinking"}:
                        kind = raw_kind
                    if isinstance(raw_content, str):
                        content = raw_content
                if content:
                    sse_tracer.emit_stream_delta(run_id, content, kind=kind)

            result = await harness.run(task, session_id=session_id, stream_callback=stream_cb)
            passed = result.eval_result.passed if result.eval_result else None
            await queue.put(
                _sse(
                    {
                        "type": "done",
                        "exit_reason": result.exit_reason,
                        "steps": result.total_steps,
                        "total_cost": result.total_cost_usd,
                        "total_input_tokens": result.total_input_tokens,
                        "total_output_tokens": result.total_output_tokens,
                        "passed": passed,
                        "error": result.error or "",
                    }
                )
            )
        except asyncio.CancelledError:
            await queue.put(
                _sse(
                    {
                        "type": "done",
                        "exit_reason": "interrupted",
                        "steps": 0,
                        "total_cost": 0.0,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "passed": None,
                        "error": "",
                    }
                )
            )
        except Exception as exc:
            await queue.put(_sse({"type": "error", "message": str(exc)}))
        finally:
            _run_tasks.pop(run_id, None)

    _run_tasks[run_id] = asyncio.create_task(_run())
    return {"run_id": run_id, "session_id": session_id}


# ── Lazy static files ─────────────────────────────────────────────────────────


class _LazyStaticFiles:
    """StaticFiles mount that initializes on first request.

    Allows the /console mount to work even when the gateway process was started
    before the console dist was built (e.g. first-time install or rebuild).
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._inner: StaticFiles | None = None

    def _get_inner(self) -> StaticFiles | None:
        if self._inner is not None:
            return self._inner
        if self._directory.exists():
            self._inner = StaticFiles(directory=str(self._directory), html=True)
            return self._inner
        return None

    async def __call__(self, scope, receive, send) -> None:
        from starlette.responses import Response

        inner = self._get_inner()
        if inner is None:
            await Response(
                "Gateway console not built. Run: bash scripts/build-frontend.sh --gateway",
                status_code=503,
                media_type="text/plain",
            )(scope, receive, send)
            return
        await inner(scope, receive, send)


# ── App factory ───────────────────────────────────────────────────────────────


def _build_app():
    """Build the gateway FastAPI app: Lab API routes + IM gateway routes + console."""
    from harnessx.api.app import create_app as create_lab_app

    app = create_lab_app(serve_static=False)

    # Mount all gateway routes
    app.include_router(gateway_router)

    # Console static files — use lazy mount so a process started before the
    # console dist was built still serves correctly after a rebuild/install.
    console_dist = Path(__file__).parent / "console" / "dist"
    app.mount("/console", _LazyStaticFiles(console_dist), name="console")

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/console/")

    return app


app = _build_app()
