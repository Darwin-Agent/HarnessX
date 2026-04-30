from __future__ import annotations

import asyncio
import logging
import random
import time
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from harnessx import Harness, BaseTask

from .base_channel import BaseChannel, MessageEvent, MessageType
from .im_stream import im_stream_q_var

if TYPE_CHECKING:
    from .session_store import SessionStore

logger = logging.getLogger(__name__)

# ContextVars: set by dispatcher before harness.run(), read by processors
_im_channel_var: ContextVar[BaseChannel | None] = ContextVar("im_channel", default=None)
_im_event_var: ContextVar[MessageEvent | None] = ContextVar("im_event", default=None)
_im_session_id_var: ContextVar[str | None] = ContextVar("im_session_id", default=None)
_im_confirm_registry_var: ContextVar[dict | None] = ContextVar("im_confirm_registry", default=None)


QUEUE_MAXSIZE = 100
SESSION_IDLE_TIMEOUT = 3600  # 1 hour
SESSION_GC_INTERVAL = 600  # GC every 10 minutes
DEBOUNCE_MS = 50  # 50ms debounce for client-fragmented messages

_EDIT_INTERVALS: dict[str, float] = {
    "feishu": 0.2,
    "telegram": 1.0,
    "slack": 0.8,
    "discord": 0.8,
    "dingtalk": 0.5,
}
_BUFFER_THRESHOLDS: dict[str, int] = {
    "feishu": 10,
    "telegram": 30,
    "slack": 20,
    "discord": 20,
    "dingtalk": 15,
}


def _resolve_provider(model_str: str, model_config: Any) -> Any:
    """Return a provider for *model_str*.

    Priority:
    1. Named role in model_config.models (e.g. "judge", "compact")
    2. Bare model string — wrapped in LiteLLMProvider for portability
    """
    if model_str in model_config.models:
        return model_config.models[model_str]
    from harnessx.providers.litellm_provider import LiteLLMProvider

    return LiteLLMProvider(model_str)


class CommandRegistry:
    """
    Short-circuit built-in commands before entering Harness.
    Built-ins: /reset /help /cancel
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable] = {}

    def register(self, command: str, fn: Callable) -> None:
        self._handlers[command.lstrip("/")] = fn

    async def handle(self, event: MessageEvent, channel: BaseChannel, dispatcher: "ChannelDispatcher") -> bool:
        if not event.is_command():
            return False
        cmd = event.get_command()

        if cmd == "reset":
            await dispatcher.reset_session(channel, event)
            target = channel.make_reply_target(event)
            await channel.send(target, "✅ Conversation history cleared. New session started.")
            return True

        if cmd == "help":
            target = channel.make_reply_target(event)
            await channel.send(target, channel.help_text())
            return True

        if cmd == "cancel":
            session_id = dispatcher._make_session_id(channel, event)
            run_task = dispatcher._active_runs.get(session_id)
            if run_task and not run_task.done():
                target = channel.make_reply_target(event)
                await channel.send(target, "⏹ Cancelling the current task...")
                dispatcher._cancel_flags[session_id] = True
            else:
                target = channel.make_reply_target(event)
                await channel.send(target, "No running task.")
            return True

        if cmd == "pair":
            args = event.get_command_args()
            if not args:
                target = channel.make_reply_target(event)
                await channel.send(target, "Usage: /pair <pairing_code>")
                return True
            auth = dispatcher._auth.get(channel.name)
            if auth:
                success, msg = await auth.verify(event.sender_id, args[0])
                target = channel.make_reply_target(event)
                await channel.send(target, msg)
            return True

        if cmd == "status":
            session_id = dispatcher._make_session_id(channel, event)
            q = dispatcher._channel_queues.get(channel.name)
            qsize = q.qsize() if q else 0
            run_task = dispatcher._active_runs.get(session_id)
            is_running = run_task is not None and not run_task.done()
            last_ts = dispatcher._session_last_active.get(session_id)
            if last_ts:
                elapsed = int(time.time() - last_ts)
                if elapsed < 60:
                    last_str = f"{elapsed}s ago"
                elif elapsed < 3600:
                    last_str = f"{elapsed // 60}m ago"
                else:
                    last_str = f"{elapsed // 3600}h ago"
            else:
                last_str = "never"
            harness = dispatcher._get_harness(channel)
            try:
                model_name = harness.model_config.main.model
            except Exception:
                model_name = "unknown"
            budget = channel.config.get("token_budget")
            budget_str = f"{int(budget):,}" if budget else "unlimited"
            session_mode = channel.config.get("session_mode", "shared")
            status_icon = "🔄 running" if is_running else "🟢 idle"
            lines = [
                f"*Session:* `{session_id}`",
                f"*Status:* {status_icon}",
                f"*Last active:* {last_str}",
                f"*Queue depth:* {qsize}",
                f"*Model:* {model_name}",
                f"*Token budget:* {budget_str}",
                f"*Session mode:* {session_mode}",
            ]
            target = channel.make_reply_target(event)
            await channel.send(target, "\n".join(lines))
            return True

        if cmd == "usage":
            session_id = dispatcher._make_session_id(channel, event)
            usage = dispatcher._session_usage.get(session_id)
            target = channel.make_reply_target(event)
            if not usage:
                await channel.send(target, "No usage data for this session yet.")
                return True
            last_t = usage.get("last_run_tokens", 0)
            last_c = usage.get("last_run_cost_usd", 0.0)
            total_t = usage.get("total_tokens", 0)
            total_c = usage.get("total_cost_usd", 0.0)
            lines = [
                f"*Last run:* {last_t:,} tokens / ${last_c:.4f}",
                f"*Session total:* {total_t:,} tokens / ${total_c:.4f}",
            ]
            await channel.send(target, "\n".join(lines))
            return True

        if cmd == "model":
            session_id = dispatcher._make_session_id(channel, event)
            target = channel.make_reply_target(event)
            harness = dispatcher._get_harness(channel)
            args = event.get_command_args()

            try:
                default_model = harness.model_config.main.model
            except Exception:
                default_model = "unknown"
            override = dispatcher._session_model_overrides.get(session_id)
            current_model = override or default_model

            if not args or args[0] in ("show",):
                lines = [f"*Current model:* `{current_model}`"]
                if override:
                    lines.append(f"_(session override — default: `{default_model}`)_")
                await channel.send(target, "\n".join(lines))
                return True

            if args[0] == "list":
                models_dict = harness.model_config.models
                lines = ["*Available models:*"]
                for role, provider in models_dict.items():
                    model_name = getattr(provider, "model", None) or role
                    marker = " ✓" if (override == role or (not override and role == "main")) else ""
                    lines.append(f"• `{role}` → `{model_name}`{marker}")
                if len(models_dict) <= 1:
                    lines.append("_(Add more roles in model\\_config.yaml to enable switching)_")
                await channel.send(target, "\n".join(lines))
                return True

            if args[0] == "reset":
                dispatcher._session_model_overrides.pop(session_id, None)
                await channel.send(target, f"✅ Model reset to default: `{default_model}`")
                return True

            new_model_str = args[0]
            try:
                provider = _resolve_provider(new_model_str, harness.model_config)
                dispatcher._session_model_overrides[session_id] = new_model_str
                actual = getattr(provider, "model", None) or new_model_str
                await channel.send(target, f"✅ Model switched to `{actual}` for this session.")
            except Exception as e:
                await channel.send(target, f"❌ Failed to resolve model `{new_model_str}`: {e}")
            return True

        if cmd == "logs":
            args = event.get_command_args()
            try:
                n = int(args[0]) if args else 20
                n = max(1, min(n, 200))
            except (ValueError, IndexError):
                n = 20
            log_file = Path("/tmp/hx-gateway/gateway.log")
            target = channel.make_reply_target(event)
            if not log_file.exists():
                await channel.send(target, "Log file not found.")
                return True
            lines = log_file.read_text(errors="replace").splitlines()
            tail = lines[-n:] if len(lines) > n else lines
            await channel.send(target, "```\n" + "\n".join(tail) + "\n```")
            return True

        if cmd == "version":
            try:
                from harnessx import __version__ as hx_ver
            except Exception:
                hx_ver = "unknown"
            import sys

            working_dir = Path.cwd()
            log_file = Path("/tmp/hx-gateway/gateway.log")
            lines = [
                f"*HarnessX gateway* v{hx_ver}",
                f"*Python:* {sys.version.split()[0]}",
                f"*Working dir:* `{working_dir}`",
                f"*Log:* `{log_file}`",
            ]
            target = channel.make_reply_target(event)
            await channel.send(target, "\n".join(lines))
            return True

        if cmd == "compact":
            session_id = dispatcher._make_session_id(channel, event)
            harness = dispatcher._get_harness(channel)
            target = channel.make_reply_target(event)
            try:
                result = await harness.run(
                    BaseTask(description="", max_steps=1, force_compact=True),
                    session_id=session_id,
                )
                await channel.send(target, result.final_output or "✅ 压缩完成")
            except Exception as e:
                logger.error("[%s] /compact failed: %s", channel.name, e, exc_info=True)
                await channel.send(target, f"❌ Compact failed: {e}")
            return True

        if cmd == "skills":
            harness = dispatcher._get_harness(channel)
            target = channel.make_reply_target(event)
            try:
                from harnessx.workspace.skill_index import SkillIndex, collect_plugin_skill_dirs

                workspace = getattr(harness._rt, "workspace", None)
                home = None
                skills_dir = None
                if workspace is not None:
                    home = getattr(workspace, "home", None)
                    if home is not None:
                        candidate = Path(home) / "skills"
                        if candidate.is_dir():
                            skills_dir = candidate
                    if skills_dir is None and hasattr(workspace, "root"):
                        skills_dir = Path(workspace.root) / "skills"
                plugin_dirs = collect_plugin_skill_dirs(home)
                if (skills_dir is None or not skills_dir.exists()) and not plugin_dirs:
                    await channel.send(target, "No skills directory found.")
                    return True
                index = SkillIndex(skills_dir, extra_dirs=plugin_dirs)
                skills = index.list_skills()
                if not skills:
                    await channel.send(target, "No skills installed.")
                    return True
                lines = [f"*Installed skills ({len(skills)}):*"]
                for s in skills:
                    desc = s.description[:80] if s.description else ""
                    lines.append(f"• `{s.name}` — {desc}" if desc else f"• `{s.name}`")
                await channel.send(target, "\n".join(lines))
            except Exception as e:
                await channel.send(target, f"Could not list skills: {e}")
            return True

        if cmd in ("restart", "reload-config"):
            target = channel.make_reply_target(event)

            if cmd == "restart":
                await channel.send(target, "⏳ Restarting channel...")
                try:
                    ch_name = channel.name
                    ch_class = type(channel)
                    ch_config = dict(channel.config)
                    await dispatcher.stop_channel(ch_name)
                    new_ch = ch_class(config=ch_config, dispatcher=dispatcher)
                    new_ch.name = ch_name
                    await dispatcher.start_channel(new_ch)
                    # Send confirmation via new channel instance
                    from .base_channel import ReplyTarget

                    new_target = ReplyTarget(chat_id=target.chat_id, message_id=target.message_id)
                    await new_ch.send(new_target, "✅ Channel restarted.")
                except Exception as e:
                    logger.error("[%s] /restart failed: %s", channel.name, e, exc_info=True)
                    try:
                        await channel.send(target, f"❌ Restart failed: {e}")
                    except Exception:
                        pass
                return True

            # reload-config
            cfg_path = dispatcher._gateway_config_path or (Path.home() / ".harnessx" / "gateway.yaml")
            if not cfg_path.exists():
                await channel.send(target, f"❌ Config file not found: `{cfg_path}`")
                return True
            try:
                import yaml

                with open(cfg_path, encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                channels_cfg = raw.get("channels", {})
                updated = []
                for ch_name, ch in dispatcher._channels.items():
                    if ch_name in channels_cfg:
                        ch.config = channels_cfg[ch_name]
                        updated.append(ch_name)
                await channel.send(target, f"✅ Config reloaded. Updated channels: {', '.join(updated) or 'none'}")
            except Exception as e:
                logger.error("[dispatch] /reload-config failed: %s", e, exc_info=True)
                await channel.send(target, f"❌ Reload failed: {e}")
            return True

        if cmd in self._handlers:
            await self._handlers[cmd](event, channel, dispatcher)
            return True

        return False


class OutboundRateLimiter:
    """
    Per-channel token bucket for outbound API rate limiting.
    Reference rates (requests/second):
      feishu: 5, telegram: 1, slack: 1, discord: 5, dingtalk: 0.33 (20/min)
    """

    _DEFAULT_RPS: dict[str, float] = {
        "feishu": 5.0,
        "telegram": 1.0,
        "slack": 1.0,
        "discord": 5.0,
        "dingtalk": 0.33,
    }

    def __init__(self) -> None:
        self._tokens: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._rps: dict[str, float] = {}

    def configure(self, channel_name: str, rps: float) -> None:
        self._rps[channel_name] = rps
        # Bucket capacity is max(rps, 1.0) so low-rps channels (e.g. DingTalk=0.33)
        # can still send one message immediately on startup.
        self._tokens[channel_name] = max(rps, 1.0)
        self._last[channel_name] = time.time()

    def _refill(self, name: str) -> None:
        rps = self._rps.get(name, self._DEFAULT_RPS.get(name, 1.0))
        capacity = max(rps, 1.0)  # bucket capacity must be >= 1 token
        now = time.time()
        elapsed = now - self._last.get(name, now)
        self._tokens[name] = min(capacity, self._tokens.get(name, capacity) + elapsed * rps)
        self._last[name] = now

    async def acquire(self, channel_name: str) -> None:
        backoff = 0.1
        while True:
            self._refill(channel_name)
            if self._tokens.get(channel_name, 1.0) >= 1.0:
                self._tokens[channel_name] -= 1.0
                return
            jitter = random.uniform(0, backoff * 0.1)
            await asyncio.sleep(backoff + jitter)
            backoff = min(backoff * 2, 60.0)


class ChannelDispatcher:
    """
    Core dispatcher:
    - per-channel asyncio.Queue (maxsize=100, drop-oldest overflow)
    - per-session asyncio.Lock (serial execution, journal consistency)
    - concurrent stream sender task during harness.run()
    - session GC for idle cleanup
    """

    def __init__(
        self,
        default_harness: Harness,
        session_store: "SessionStore",
        channel_harnesses: dict[str, Harness] | None = None,
    ) -> None:
        self._default_harness = default_harness
        self._session_store = session_store
        self._channel_harnesses: dict[str, Harness] = channel_harnesses or {}

        self._channels: dict[str, BaseChannel] = {}
        self._channel_queues: dict[str, asyncio.PriorityQueue] = {}
        self._seq_counter: int = 0  # monotonic sequence to break priority ties
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_epochs: dict[str, int] = {}
        self._session_last_active: dict[str, float] = {}
        self._active_runs: dict[str, asyncio.Task] = {}
        self._cancel_flags: dict[str, bool] = {}
        self._session_usage: dict[str, dict] = {}
        self._confirm_registry: dict[str, asyncio.Future] = {}
        self._session_model_overrides: dict[str, str] = {}
        self._gateway_config_path: Path | None = None
        self._rate_limiter = OutboundRateLimiter()
        self._commands = CommandRegistry()
        self._auth: dict[str, object] = {}
        self._tasks: list[asyncio.Task] = []
        self._channel_tasks: dict[str, list[asyncio.Task]] = {}
        self._last_dispatch: tuple[str, str, str] | None = None  # (channel, chat_id, session_id)
        self._last_dispatch_lock = asyncio.Lock()

        # Restore epochs from store
        self._session_epochs.update(session_store.load_epochs())

    def register(self, channel: BaseChannel) -> None:
        self._channels[channel.name] = channel
        self._channel_queues[channel.name] = asyncio.PriorityQueue(maxsize=QUEUE_MAXSIZE)
        rl_cfg = channel.config.get("rate_limit", {})
        rps = rl_cfg.get("requests_per_second")
        if rps:
            self._rate_limiter.configure(channel.name, float(rps))

        # Set up auth if pairing mode
        if channel.config.get("auth_mode") == "pairing":
            from .auth import PairingAuth

            auth = PairingAuth(channel.name, self._session_store)
            self._auth[channel.name] = auth

    def set_config_path(self, path: Path) -> None:
        self._gateway_config_path = path

    @property
    def channels(self) -> list[BaseChannel]:
        return list(self._channels.values())

    def get_channel(self, name: str) -> BaseChannel | None:
        return self._channels.get(name)

    async def start_all(self) -> None:
        for channel in self._channels.values():
            t = asyncio.create_task(channel.start(), name=f"ch:{channel.name}")
            ct = asyncio.create_task(self._consume(channel), name=f"consume:{channel.name}")
            self._tasks.extend([t, ct])
            self._channel_tasks[channel.name] = [t, ct]
        gc = asyncio.create_task(self._gc_sessions(), name="session_gc")
        self._tasks.append(gc)

    async def start_channel(self, channel: BaseChannel) -> None:
        """Register and start a single channel (hot-add / hot-restart)."""
        self.register(channel)
        t = asyncio.create_task(channel.start(), name=f"ch:{channel.name}")
        ct = asyncio.create_task(self._consume(channel), name=f"consume:{channel.name}")
        self._tasks.extend([t, ct])
        self._channel_tasks[channel.name] = [t, ct]

    async def stop_channel(self, name: str, timeout: float = 10.0) -> None:
        """Stop and deregister a single channel."""
        ch = self._channels.pop(name, None)
        if ch:
            try:
                await asyncio.wait_for(ch.stop(timeout=timeout), timeout=timeout + 2)
            except Exception:
                pass
        for t in self._channel_tasks.pop(name, []):
            t.cancel()
        self._channel_queues.pop(name, None)
        self._auth.pop(name, None)

    async def stop_all(self, timeout: float = 30.0) -> None:
        await asyncio.gather(
            *[ch.stop(timeout=timeout) for ch in self._channels.values()],
            return_exceptions=True,
        )
        await asyncio.gather(
            *[q.join() for q in self._channel_queues.values()],
            return_exceptions=True,
        )
        for t in self._tasks:
            t.cancel()

    async def enqueue(self, channel: BaseChannel, event: MessageEvent) -> None:
        q = self._channel_queues.get(channel.name)
        if q is None:
            logger.error(
                "[dispatch] enqueue: no queue for channel=%r (registered: %s)",
                channel.name,
                list(self._channel_queues.keys()),
            )
            return
        # Commands (priority=0) jump ahead of normal messages (priority=1)
        priority = 0 if event.is_command() else 1
        self._seq_counter += 1
        item = (priority, self._seq_counter, channel, event)
        logger.info(
            "[dispatch] enqueue channel=%r priority=%d qsize=%d text=%r",
            channel.name,
            priority,
            q.qsize(),
            event.text[:40],
        )
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            # Drop oldest normal-priority item to make room
            try:
                q.get_nowait()
                q.task_done()
            except asyncio.QueueEmpty:
                pass
            q.put_nowait(item)
            target = channel.make_reply_target(event)
            try:
                await channel.send(target, "⚠️ Message queue is busy. The oldest pending message was dropped.")
            except Exception:
                pass

    async def _consume(self, channel: BaseChannel) -> None:
        q = self._channel_queues[channel.name]
        logger.info("[dispatch] _consume started for channel=%r", channel.name)
        while True:
            _priority, _seq, ch, event = await q.get()
            logger.info(
                "[dispatch] _consume dequeued channel=%r priority=%d text=%r", channel.name, _priority, event.text[:40]
            )
            try:
                await self._handle_event(ch, event)
            except Exception as e:
                logger.error("[%s] error handling event: %s", channel.name, e, exc_info=True)
            finally:
                q.task_done()

    async def _handle_event(self, channel: BaseChannel, event: MessageEvent) -> None:
        logger.info(
            "[dispatch] _handle_event channel=%r sender=%r text=%r mentioned=%s",
            channel.name,
            event.sender_id,
            event.text[:40],
            event.conversation.mentioned,
        )
        # Auth check first
        if channel.config.get("auth_mode") == "pairing":
            auth = self._auth.get(channel.name)
            if auth and not auth.is_authorized(event.sender_id):
                # Allow /pair command through
                if event.is_command() and event.get_command() == "pair":
                    await self._commands.handle(event, channel, self)
                else:
                    target = channel.make_reply_target(event)
                    await channel.send(
                        target, "You are not authorized yet. Send /pair <pairing_code> to complete verification."
                    )
                return

        if not channel.should_handle(event):
            logger.info(
                "[%s] should_handle=False sender=%s conv_type=%s mentioned=%s require_mention=%s",
                channel.name,
                event.sender_id,
                event.conversation.type,
                event.conversation.mentioned,
                channel.config.get("require_mention", True),
            )
            return

        if event.message_type == MessageType.SYSTEM:
            return

        if await self._commands.handle(event, channel, self):
            return

        session_id = self._make_session_id(channel, event)

        # Resolve a pending tool-confirmation Future BEFORE acquiring the session lock.
        # The harness run holds the lock while awaiting this Future, so we must not
        # try to acquire the lock here — just resolve and return.
        pending_fut = self._confirm_registry.get(session_id)
        if pending_fut and not pending_fut.done():
            text = event.text.strip().lower()
            approved = text in ("确认", "确认执行", "yes", "y", "confirm", "ok")
            self._confirm_registry.pop(session_id, None)
            pending_fut.set_result(approved)
            target = channel.make_reply_target(event)
            ack = "✅ 确认，继续执行。" if approved else "❌ 已取消。"
            try:
                await channel.send(target, ack)
            except Exception:
                pass
            return

        lock = self._session_locks.setdefault(session_id, asyncio.Lock())

        async with lock:
            is_new = session_id not in self._session_last_active
            if is_new:
                is_first_ever = session_id not in self._session_epochs
                self._session_epochs.setdefault(session_id, 0)
                if is_first_ever:
                    await channel.on_session_start(session_id, event)

            self._session_last_active[session_id] = time.time()
            target = channel.make_reply_target(event)

            # Immediate typing indicator
            try:
                await channel.send_typing(target)
            except Exception:
                pass

            harness = self._get_session_harness(session_id, channel)

            # Capability-aware task description (images/voice/video routing)
            from .model_caps import get_input_modalities as _get_mods

            try:
                _model_name = harness.model_config.main.model
            except Exception:
                _model_name = ""
            description = self._build_description(event, _get_mods(_model_name))

            # Stream queue + concurrent sender task
            stream_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2000)

            def sync_cb(delta: object) -> None:
                if isinstance(delta, dict):
                    kind = delta.get("type")
                    if kind == "token":
                        content = delta.get("content", "")
                        if content:
                            try:
                                stream_q.put_nowait(content)
                            except asyncio.QueueFull:
                                pass
                    elif kind == "tool_start":
                        # Forward tool_start as segment boundary for send_stream
                        try:
                            stream_q.put_nowait(delta)
                        except asyncio.QueueFull:
                            pass
                else:
                    content = str(delta)
                    if content:
                        try:
                            stream_q.put_nowait(content)
                        except asyncio.QueueFull:
                            pass

            edit_interval = _EDIT_INTERVALS.get(channel.name, 0.8)
            buf_threshold = _BUFFER_THRESHOLDS.get(channel.name, 20)

            sender_task = asyncio.create_task(
                channel.send_stream(
                    target,
                    stream_q,
                    edit_interval=edit_interval,
                    buffer_threshold=buf_threshold,
                )
            )

            # Processing reaction: add ⏳ while the agent runs
            _reaction_id: str | None = None
            if channel.supports_reactions:
                try:
                    _reaction_id = await channel.add_reaction(event.message_id, "⏳")
                except Exception:
                    pass

            # Set context vars for processors
            tok1 = _im_channel_var.set(channel)
            tok2 = _im_event_var.set(event)
            tok3 = im_stream_q_var.set(stream_q)
            tok4 = _im_session_id_var.set(session_id)
            tok5 = _im_confirm_registry_var.set(self._confirm_registry)
            _run_success = False
            try:
                _raw_budget = channel.config.get("token_budget")
                harness_task_obj = BaseTask(
                    description=description,
                    token_budget=int(_raw_budget) if _raw_budget is not None else None,
                )
                run_coro = harness.run(
                    harness_task_obj,
                    session_id=session_id,
                    stream_callback=sync_cb,
                )
                run_asyncio_task = asyncio.create_task(run_coro)
                self._active_runs[session_id] = run_asyncio_task

                async def _cancel_watcher():
                    while not run_asyncio_task.done():
                        if self._cancel_flags.get(session_id):
                            run_asyncio_task.cancel()
                            return
                        await asyncio.sleep(0.2)

                watcher_task = asyncio.create_task(_cancel_watcher())
                try:
                    result = await run_asyncio_task
                    _run_success = result.exit_reason in ("done", "budget_exceeded")
                except asyncio.CancelledError:
                    _run_success = False
                    result = None
                finally:
                    watcher_task.cancel()
            finally:
                _im_channel_var.reset(tok1)
                _im_event_var.reset(tok2)
                im_stream_q_var.reset(tok3)
                _im_session_id_var.reset(tok4)
                _im_confirm_registry_var.reset(tok5)

            # On success: remove ⏳ (the reply is the success signal).
            # On failure: remove ⏳ then add ❌.
            if channel.supports_reactions:
                try:
                    if _reaction_id:
                        await channel.remove_reaction(event.message_id, _reaction_id)
                    if not _run_success:
                        await channel.add_reaction(event.message_id, "❌")
                except Exception:
                    pass

            # Signal stream sender to finish
            await stream_q.put(None)
            try:
                await asyncio.wait_for(sender_task, timeout=15.0)
            except asyncio.TimeoutError:
                sender_task.cancel()

            # Fallback for channels that use reply_text() (e.g. DingTalk Stream):
            # if the channel's send_stream() never resolved the reply_future
            # (e.g. webhook failed / no tokens streamed), resolve it now with
            # the harness final output so process() can call reply_text().
            final_text = (result.final_output if result else None) or ""
            channel.resolve_stream_reply(target.chat_id, final_text)

            # Track per-session token usage
            if result is not None:
                try:
                    run_tokens = result.total_tokens or 0
                    run_cost = result.total_cost_usd or 0.0
                    prev = self._session_usage.get(session_id, {"total_tokens": 0, "total_cost_usd": 0.0})
                    self._session_usage[session_id] = {
                        "total_tokens": prev["total_tokens"] + run_tokens,
                        "total_cost_usd": prev["total_cost_usd"] + run_cost,
                        "last_run_tokens": run_tokens,
                        "last_run_cost_usd": run_cost,
                    }
                except Exception:
                    pass

            self._active_runs.pop(session_id, None)
            self._cancel_flags.pop(session_id, None)

            # Record last user interaction for heartbeat target="last"
            async with self._last_dispatch_lock:
                self._last_dispatch = (channel.name, event.conversation.chat_id, session_id)

    def _build_description(self, event: MessageEvent, modalities: frozenset[str] = frozenset()) -> str | list:
        import base64
        import mimetypes
        import os

        match event.message_type:
            case MessageType.IMAGE if event.media_paths:
                # Multimodal: content blocks
                blocks: list[dict] = []
                for path in event.media_paths:
                    try:
                        with open(path, "rb") as f:
                            data = base64.standard_b64encode(f.read()).decode()
                        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
                        blocks.append(
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": mime, "data": data},
                            }
                        )
                    except Exception:
                        pass
                if event.text and event.text not in ("[image]", ""):
                    blocks.append({"type": "text", "text": event.text})
                elif not blocks:
                    return event.text or "[image]"
                return blocks if blocks else event.text

            case MessageType.VOICE if event.media_paths:
                path = event.media_paths[0]
                caption = event.text if event.text and event.text not in ("[voice]", "") else ""
                if "audio" in modalities:
                    from .model_caps import AUDIO_EMBED_MAX_BYTES

                    try:
                        if os.path.getsize(path) <= AUDIO_EMBED_MAX_BYTES:
                            with open(path, "rb") as f:
                                data = base64.standard_b64encode(f.read()).decode()
                            mime = mimetypes.guess_type(path)[0] or "audio/ogg"
                            fmt = mime.split("/")[-1]
                            blocks = []
                            if caption:
                                blocks.append({"type": "text", "text": caption})
                            blocks.append({"type": "input_audio", "input_audio": {"data": data, "format": fmt}})
                            return blocks
                    except Exception:
                        pass
                # Fallback: text with file path so agent can use tools
                parts = [f"[voice file: {path}]"]
                if caption:
                    parts.append(caption)
                return "\n".join(parts)

            case MessageType.VOICE:
                # No media file — just transcription text
                caption = event.text if event.text and event.text != "[voice]" else ""
                return f"[voice message]{': ' + caption if caption else ''}"

            case MessageType.VIDEO if event.media_paths:
                path = event.media_paths[0]
                caption = event.text if event.text and event.text not in ("[video]", "") else ""
                if "video" in modalities:
                    from .model_caps import VIDEO_EMBED_MAX_BYTES

                    try:
                        if os.path.getsize(path) <= VIDEO_EMBED_MAX_BYTES:
                            with open(path, "rb") as f:
                                data = base64.standard_b64encode(f.read()).decode()
                            mime = mimetypes.guess_type(path)[0] or "video/mp4"
                            blocks = []
                            if caption:
                                blocks.append({"type": "text", "text": caption})
                            blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
                            return blocks
                    except Exception:
                        pass
                # Fallback: text with file path so agent can use tools
                parts = [f"[video file: {path}]"]
                if caption:
                    parts.append(caption)
                return "\n".join(parts)

            case MessageType.FILE:
                return f"[file: {event.text}, path: {', '.join(event.media_paths)}]"
            case _:
                return event.text

    def _make_session_id(self, channel: BaseChannel, event: MessageEvent) -> str:
        base = channel.resolve_session_id(event)
        epoch = self._session_epochs.get(base, 0)
        return f"{base}#{epoch}" if epoch > 0 else base

    def _get_harness(self, channel: BaseChannel) -> Harness:
        return self._channel_harnesses.get(channel.name, self._default_harness)

    def _get_session_harness(self, session_id: str, channel: BaseChannel) -> Harness:
        """Return a harness with per-session model override applied (if any)."""
        harness = self._get_harness(channel)
        override = self._session_model_overrides.get(session_id)
        if not override:
            return harness
        try:
            provider = _resolve_provider(override, harness.model_config)
            new_mc = harness.model_config.copy(main=provider)
            new_harness = new_mc.agentic(harness.config)
            new_harness.child_harness_config = getattr(harness, "child_harness_config", None)
            return new_harness
        except Exception as e:
            logger.warning("[%s] ignoring model override %r: %s", channel.name, override, e)
            return harness

    async def reset_session(self, channel: BaseChannel, event: MessageEvent) -> None:
        base = channel.resolve_session_id(event)
        old_epoch = self._session_epochs.get(base, 0)
        old_session_id = f"{base}#{old_epoch}" if old_epoch > 0 else base
        fut = self._confirm_registry.pop(old_session_id, None)
        if fut and not fut.done():
            fut.set_result(False)
        self._session_model_overrides.pop(old_session_id, None)
        new_epoch = old_epoch + 1
        self._session_epochs[base] = new_epoch
        self._session_store.save_epoch(base, new_epoch)
        new_session_id = f"{base}#{new_epoch}"
        self._session_usage.pop(base, None)
        self._session_usage.pop(new_session_id, None)
        await channel.on_session_reset(base)
        logger.info("[%s] session %s reset → epoch %d", channel.name, base, new_epoch)

    async def get_last_dispatch(self) -> tuple[str, str, str] | None:
        """Return the (channel_name, chat_id, session_id) of the most recent dispatch."""
        async with self._last_dispatch_lock:
            return self._last_dispatch

    async def run_cron(
        self,
        prompt: str,
        channel_name: str | None = None,
        chat_id: str | None = None,
        session_id: str = "cron",
    ) -> str:
        """Run a cron / heartbeat task against the harness.

        If channel_name + chat_id are provided, the final output is sent to that channel.
        Otherwise the agent runs silently (useful for maintenance tasks).
        """
        from harnessx import BaseTask

        harness = (
            self._channel_harnesses.get(channel_name, self._default_harness) if channel_name else self._default_harness
        )
        full_session_id = f"{channel_name}:{session_id}" if channel_name else session_id

        task = BaseTask(description=prompt)
        result = await harness.run(task, session_id=full_session_id)

        output = result.final_output or ""

        if channel_name and chat_id and output:
            channel = self._channels.get(channel_name)
            if channel:
                from .base_channel import ReplyTarget

                target = ReplyTarget(chat_id=chat_id)
                try:
                    await channel.send(target, output)
                except Exception as e:
                    logger.warning("[cron] failed to send reply to %s/%s: %s", channel_name, chat_id, e)

        return output

    async def _gc_sessions(self) -> None:
        while True:
            await asyncio.sleep(SESSION_GC_INTERVAL)
            now = time.time()
            expired = [sid for sid, ts in self._session_last_active.items() if now - ts > SESSION_IDLE_TIMEOUT]
            for sid in expired:
                self._session_locks.pop(sid, None)
                self._session_last_active.pop(sid, None)
                self._session_model_overrides.pop(sid, None)
            if expired:
                logger.debug("GC: removed %d idle sessions", len(expired))
