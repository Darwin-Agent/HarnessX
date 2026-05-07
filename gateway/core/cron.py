from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dispatch import ChannelDispatcher

logger = logging.getLogger(__name__)

HEARTBEAT_JOB_ID = "_heartbeat"


# ── Models ─────────────────────────────────────────────────────────────────────


@dataclass
class CronJob:
    id: str
    name: str
    enabled: bool = True
    cron: str = ""  # 5-field cron expression, e.g. "0 9 * * *"
    every: str = ""  # shorthand interval: "30m" / "1h" / "2h30m"
    prompt: str = ""  # text sent to the agent
    target: str = "channel"  # "channel" (use channel+chat_id) | "last" (last active user)
    channel: str = ""  # gateway channel name to reply into (empty = run silently)
    chat_id: str = ""  # platform chat/channel ID to post the reply to
    session_id: str = "cron"
    timeout: int = 120
    timezone: str = "UTC"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CronJob":
        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class CronJobState:
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    last_status: str | None = None  # "success" | "error" | "running" | "skipped"
    last_error: str | None = None
    last_output: str | None = None

    def to_dict(self) -> dict:
        return {
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_status": self.last_status,
            "last_error": self.last_error,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _every_to_cron(every: str) -> str:
    """'30m' → '*/30 * * * *', '1h' → '0 */1 * * *', '2h30m' → '*/30 */2 * * *'."""
    every = every.strip().lower()
    total_seconds = 0
    for val, unit in re.findall(r"(\d+)([smh])", every):
        v = int(val)
        if unit == "s":
            total_seconds += v
        elif unit == "m":
            total_seconds += v * 60
        elif unit == "h":
            total_seconds += v * 3600
    total_minutes = max(1, total_seconds // 60)
    if total_minutes < 60:
        return f"*/{total_minutes} * * * *"
    hours = total_minutes // 60
    mins = total_minutes % 60
    if mins == 0:
        return f"0 */{hours} * * *"
    return f"*/{mins} */{hours} * * *"


def _next_run_time(job: CronJob) -> datetime:
    try:
        from croniter import croniter

        expr = job.cron if job.cron else _every_to_cron(job.every or "1h")
        try:
            import pytz

            tz = pytz.timezone(job.timezone)
        except Exception:
            import pytz

            tz = pytz.UTC
        return croniter(expr, datetime.now(tz=tz)).get_next(datetime)
    except Exception as e:
        logger.warning("[cron] cannot compute next_run for '%s': %s — defaulting to 1h", job.id, e)
        from datetime import timedelta

        return datetime.now().astimezone() + timedelta(hours=1)


def _in_active_hours(active_hours: dict, timezone: str) -> bool:
    try:
        import pytz

        tz = pytz.timezone(timezone) if timezone not in ("UTC", "") else pytz.UTC
        now = datetime.now(tz=tz).time().replace(second=0, microsecond=0)
        start = dtime(*map(int, active_hours["start"].split(":")))
        end = dtime(*map(int, active_hours["end"].split(":")))
        return start <= now <= end
    except Exception:
        return True


# ── CronManager ────────────────────────────────────────────────────────────────


class CronManager:
    """Asyncio-native cron scheduler for the IM Gateway.

    Jobs are persisted in agent_root/cron_jobs.json.
    Heartbeat is a special built-in job driven by gateway.yaml [heartbeat] config.
    """

    def __init__(self, dispatcher: "ChannelDispatcher", agent_root: Path) -> None:
        self._dispatcher = dispatcher
        self._agent_root = agent_root
        self._jobs: dict[str, CronJob] = {}
        self._states: dict[str, CronJobState] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._stopped = False
        self._jobs_file = agent_root / "cron_jobs.json"
        self._heartbeat_active_hours: dict | None = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._jobs_file.exists():
            return
        try:
            data = json.loads(self._jobs_file.read_text(encoding="utf-8"))
            for d in data.get("jobs", []):
                job = CronJob.from_dict(d)
                self._jobs[job.id] = job
                self._states[job.id] = CronJobState()
            logger.debug("[cron] loaded %d jobs from %s", len(self._jobs), self._jobs_file)
        except Exception as e:
            logger.warning("[cron] failed to load cron_jobs.json: %s", e)

    def _save(self) -> None:
        self._agent_root.mkdir(parents=True, exist_ok=True)
        tmp = self._jobs_file.with_suffix(".tmp")
        jobs = [j.to_dict() for j in self._jobs.values() if j.id != HEARTBEAT_JOB_ID]
        tmp.write_text(json.dumps({"version": 1, "jobs": jobs}, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._jobs_file)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, heartbeat_cfg: dict | None = None) -> None:
        self._stopped = False
        self._load()
        for job in list(self._jobs.values()):
            if job.enabled:
                self._schedule(job)
        if heartbeat_cfg and heartbeat_cfg.get("enabled"):
            await self._start_heartbeat(heartbeat_cfg)
        logger.info("[cron] started with %d user jobs", sum(1 for j in self._jobs if j != HEARTBEAT_JOB_ID))

    async def stop(self) -> None:
        self._stopped = True
        for t in self._tasks.values():
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    # ── Scheduling ────────────────────────────────────────────────────────────

    def _schedule(self, job: CronJob) -> None:
        if job.id in self._tasks:
            self._tasks[job.id].cancel()
        t = asyncio.create_task(self._job_loop(job), name=f"cron:{job.id}")
        self._tasks[job.id] = t

    async def _start_heartbeat(self, cfg: dict) -> None:
        prompt = self._read_heartbeat_md()
        if not prompt:
            logger.info("[cron] HEARTBEAT.md is empty — heartbeat disabled")
            return
        job = CronJob(
            id=HEARTBEAT_JOB_ID,
            name="Heartbeat",
            enabled=True,
            cron=cfg.get("cron", ""),
            every=cfg.get("every", "1h"),
            prompt=prompt,
            target=cfg.get("target", "last"),  # default: follow last active user
            channel=cfg.get("channel", ""),
            chat_id=cfg.get("chat_id", ""),
            session_id=cfg.get("session_id", "heartbeat"),
            timeout=cfg.get("timeout", 120),
            timezone=cfg.get("timezone", "UTC"),
        )
        self._heartbeat_active_hours = cfg.get("active_hours")
        self._jobs[HEARTBEAT_JOB_ID] = job
        self._states[HEARTBEAT_JOB_ID] = CronJobState()
        self._schedule(job)
        logger.info("[cron] heartbeat every=%r cron=%r channel=%r", job.every, job.cron, job.channel)

    def _read_heartbeat_md(self) -> str:
        path = self._agent_root / "HEARTBEAT.md"
        if not path.exists():
            return ""
        content = path.read_text(encoding="utf-8").strip()
        # Strip comment-only lines
        lines = [ln for ln in content.splitlines() if not ln.strip().startswith("#")]
        return "\n".join(lines).strip()

    # ── Job loop ──────────────────────────────────────────────────────────────

    async def _job_loop(self, job: CronJob) -> None:
        while not self._stopped:
            next_run = _next_run_time(job)
            state = self._states.get(job.id)
            if state:
                state.next_run_at = next_run

            now = datetime.now(tz=next_run.tzinfo)
            delay = (next_run - now).total_seconds()
            if delay > 0:
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return
            if self._stopped:
                return

            # Active hours guard (heartbeat only for now)
            if job.id == HEARTBEAT_JOB_ID and self._heartbeat_active_hours:
                if not _in_active_hours(self._heartbeat_active_hours, job.timezone):
                    if state:
                        state.last_status = "skipped"
                    logger.debug("[cron] heartbeat skipped — outside active hours")
                    continue

            # Overlap guard: skip if previous run is still in progress
            if state and state.last_status == "running":
                logger.info("[cron] job '%s' still running from previous tick — skipping", job.name)
                continue

            await self._run_job(job)

    async def _run_job(self, job: CronJob) -> None:
        state = self._states.get(job.id)
        if state:
            state.last_run_at = datetime.now()
            state.last_status = "running"

        logger.info("[cron] running job '%s' (%s)", job.name, job.id)

        # Resolve dispatch target
        channel_name: str | None = job.channel or None
        chat_id: str | None = job.chat_id or None
        session_id: str = job.session_id

        if job.target == "last":
            ld = self._dispatcher._last_dispatch  # (channel, chat_id, session_id) | None
            if ld:
                channel_name, chat_id, session_id = ld
                logger.debug("[cron] target=last → %s / %s / %s", channel_name, chat_id, session_id)
            else:
                logger.info("[cron] job '%s' target=last but no interaction yet — running silently", job.name)
                channel_name = None
                chat_id = None

        try:
            output = await asyncio.wait_for(
                self._dispatcher.run_cron(
                    prompt=job.prompt,
                    channel_name=channel_name,
                    chat_id=chat_id,
                    session_id=session_id,
                ),
                timeout=float(job.timeout),
            )
            if state:
                state.last_status = "success"
                state.last_output = (output or "")[:500]
                state.last_error = None
            logger.info("[cron] job '%s' done", job.name)
        except asyncio.TimeoutError:
            if state:
                state.last_status = "error"
                state.last_error = f"timeout after {job.timeout}s"
            logger.warning("[cron] job '%s' timed out after %ds", job.name, job.timeout)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if state:
                state.last_status = "error"
                state.last_error = str(e)
            logger.error("[cron] job '%s' error: %s", job.name, e, exc_info=True)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def list_jobs(self) -> list[dict]:
        result = []
        for jid, job in self._jobs.items():
            if jid == HEARTBEAT_JOB_ID:
                continue
            state = self._states.get(jid, CronJobState())
            result.append({**job.to_dict(), "state": state.to_dict()})
        return result

    def get_job(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if not job or job.id == HEARTBEAT_JOB_ID:
            return None
        state = self._states.get(job_id, CronJobState())
        d = state.to_dict()
        d["last_output"] = state.last_output
        return {**job.to_dict(), "state": d}

    def create_job(self, spec: dict) -> CronJob:
        if not spec.get("id"):
            spec["id"] = str(uuid.uuid4())[:8]
        # Clamp timeout to safe bounds
        if "timeout" in spec:
            spec["timeout"] = max(10, min(600, int(spec["timeout"])))
        job = CronJob.from_dict(spec)
        self._jobs[job.id] = job
        self._states[job.id] = CronJobState()
        if job.enabled:
            self._schedule(job)
        self._save()
        return job

    def update_job(self, job_id: str, spec: dict) -> CronJob | None:
        existing = self._jobs.get(job_id)
        if not existing or job_id == HEARTBEAT_JOB_ID:
            return None
        merged = {**existing.to_dict(), **spec, "id": job_id}
        job = CronJob.from_dict(merged)
        self._jobs[job_id] = job
        if job.enabled:
            self._schedule(job)
        elif job_id in self._tasks:
            self._tasks[job_id].cancel()
            del self._tasks[job_id]
        self._save()
        return job

    def delete_job(self, job_id: str) -> bool:
        if job_id not in self._jobs or job_id == HEARTBEAT_JOB_ID:
            return False
        if job_id in self._tasks:
            self._tasks[job_id].cancel()
            del self._tasks[job_id]
        del self._jobs[job_id]
        del self._states[job_id]
        self._save()
        return True

    async def run_now(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.id == HEARTBEAT_JOB_ID:
            return False
        state = self._states.get(job_id)
        if state and state.last_status == "running":
            return False
        asyncio.create_task(self._run_job(job), name=f"cron:run_now:{job_id}")
        return True

    def pause_job(self, job_id: str) -> bool:
        return self.update_job(job_id, {"enabled": False}) is not None

    def resume_job(self, job_id: str) -> bool:
        return self.update_job(job_id, {"enabled": True}) is not None

    def get_heartbeat_state(self) -> dict | None:
        job = self._jobs.get(HEARTBEAT_JOB_ID)
        if not job:
            return None
        state = self._states.get(HEARTBEAT_JOB_ID, CronJobState())
        return {
            "enabled": job.enabled,
            "every": job.every,
            "cron": job.cron,
            "target": job.target,
            "channel": job.channel,
            "chat_id": job.chat_id,
            "session_id": job.session_id,
            "timezone": job.timezone,
            **state.to_dict(),
        }

    async def reload_heartbeat(self, cfg: dict) -> None:
        """Hot-reload heartbeat from updated config (called after gateway.yaml save)."""
        if HEARTBEAT_JOB_ID in self._tasks:
            self._tasks[HEARTBEAT_JOB_ID].cancel()
            del self._tasks[HEARTBEAT_JOB_ID]
        self._jobs.pop(HEARTBEAT_JOB_ID, None)
        self._states.pop(HEARTBEAT_JOB_ID, None)
        if cfg.get("enabled"):
            await self._start_heartbeat(cfg)
