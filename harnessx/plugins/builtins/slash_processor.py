# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator

from ...core.processor import MultiHookProcessor, PRE

if TYPE_CHECKING:
    from ...core.events import TaskStartEvent


class SlashCommandProcessor(MultiHookProcessor):
    """Processor that intercepts slash commands at task_start.

    Registered at PRE phase so it runs before system-prompt assembly or any
    other processor.  When the task description starts with "/" the processor
    handles the command and yields a TaskEndEvent so the run loop exits
    immediately — no model call, no tool calls, no steps.

    The processor also removes the slash command message from state message
    tracks so it does not pollute persisted conversation history.

    Args:
        model_config: Optional ModelConfig instance for /model display commands.
    """

    _singleton_group = "slash_command"
    _order = PRE

    def __init__(self, model_config: "Any | None" = None) -> None:
        super().__init__()
        self._model_config = model_config

    async def on_task_start(self, event: "TaskStartEvent") -> AsyncIterator:  # type: ignore[override]
        from ...core.events import TaskEndEvent

        desc = event.task_description.strip()
        if not desc.startswith("/"):
            yield event
            return

        parts = desc.split()
        cmd = parts[0].lower()
        args = parts[1:]

        # Remove the slash command from conversation history so it does not
        # appear in future turns.  event.state is a mutable reference to the
        # live State object; popping the last user message is safe here.
        _pop_last_user_msg(event)

        def _end(exit_reason: str, final_output: str = "") -> "TaskEndEvent":
            return TaskEndEvent(
                run_id=event.run_id,
                step_id=0,
                final_output=final_output,
                exit_reason=exit_reason,
            )

        # ── /quit /exit /q ────────────────────────────────────────────────────
        if cmd in ("/quit", "/exit", "/q"):
            yield _end("slash:quit")
            return

        # ── /new ──────────────────────────────────────────────────────────────
        if cmd == "/new":
            new_sid = str(uuid.uuid4())
            print(f"New session: {new_sid}", file=sys.stderr)
            yield _end("slash:new", new_sid)
            return

        # ── /session ──────────────────────────────────────────────────────────
        if cmd == "/session":
            sid = event.session_id or event.run_id
            print(f"Session: {sid}", file=sys.stderr)
            yield _end("slash:info")
            return

        # ── /help ─────────────────────────────────────────────────────────────
        if cmd == "/help":
            from ...plugins.registry import plugin_registry

            print(plugin_registry.help_text(), file=sys.stderr)
            yield _end("slash:info")
            return

        # ── /compact ──────────────────────────────────────────────────────────
        if cmd == "/compact":
            sid = event.session_id or event.run_id
            ws = event.workspace
            if ws is not None and sid:
                await _compact_session(sid, ws)
            else:
                print(
                    "  /compact requires a workspace-backed session.",
                    file=sys.stderr,
                )
            yield _end("slash:compact_done")
            return

        # ── /home ─────────────────────────────────────────────────────────────
        if cmd == "/home":
            from harnessx.home import agent_home

            home = agent_home()
            ws = event.workspace
            print(f"AGENT_HOME : {home}", file=sys.stderr)
            print(f"Agent      : {ws.agent_id if ws else '(none)'}", file=sys.stderr)
            print(f"Project    : {ws.project if ws else '(none)'}", file=sys.stderr)
            print(f"Workspace  : {ws.root if ws else '(none)'}", file=sys.stderr)
            yield _end("slash:info")
            return

        # ── /agent [name] ─────────────────────────────────────────────────────
        if cmd == "/agent":
            ws = event.workspace
            if not args:
                print(f"Agent: {ws.agent_id if ws else '(unknown)'}", file=sys.stderr)
                yield _end("slash:info")
            else:
                new_agent = args[0]
                project = ws.project if ws is not None else "default"
                new_sid = str(uuid.uuid4())
                print(
                    f"Switching to agent='{new_agent}' project='{project}' → new session {new_sid}",
                    file=sys.stderr,
                )
                yield _end(
                    "slash:switch",
                    json.dumps(
                        {
                            "session_id": new_sid,
                            "agent_id": new_agent,
                            "project": project,
                        }
                    ),
                )
            return

        # ── /project [name] ───────────────────────────────────────────────────
        if cmd == "/project":
            ws = event.workspace
            if not args:
                print(f"Project: {ws.project if ws else '(unknown)'}", file=sys.stderr)
                yield _end("slash:info")
            else:
                new_project = args[0]
                agent = ws.agent_id if ws is not None else "default"
                new_sid = str(uuid.uuid4())
                print(
                    f"Switching to agent='{agent}' project='{new_project}' → new session {new_sid}",
                    file=sys.stderr,
                )
                yield _end(
                    "slash:switch",
                    json.dumps(
                        {
                            "session_id": new_sid,
                            "agent_id": agent,
                            "project": new_project,
                        }
                    ),
                )
            return

        # ── /model [list | use [role] <id> | default [role] <id>] ───────────
        if cmd == "/model":
            if not args:
                # Show current model config
                _print_model_config(self._model_config)
                yield _end("slash:info")
            elif args[0] == "list":
                _print_model_list()
                yield _end("slash:info")
            elif args[0] == "use":
                # /model use <id>  OR  /model use <role> <id>
                rest = args[1:]
                if len(rest) == 0:
                    print("  Usage: /model use [role] <id-or-model-name>", file=sys.stderr)
                    yield _end("slash:info")
                elif len(rest) == 1:
                    yield _end(
                        "slash:model_use",
                        json.dumps({"role": "main", "model_ref": rest[0]}),
                    )
                else:
                    yield _end(
                        "slash:model_use",
                        json.dumps({"role": rest[0], "model_ref": rest[1]}),
                    )
            elif args[0] == "default":
                # /model default <id>  OR  /model default <role> <id>
                rest = args[1:]
                if len(rest) == 0:
                    print(
                        "  Usage: /model default [role] <id-or-model-name>",
                        file=sys.stderr,
                    )
                    yield _end("slash:info")
                elif len(rest) == 1:
                    msg = _set_model_default("main", rest[0], self._model_config)
                    print(f"  {msg}", file=sys.stderr)
                    yield _end("slash:info")
                else:
                    msg = _set_model_default(rest[0], rest[1], self._model_config)
                    print(f"  {msg}", file=sys.stderr)
                    yield _end("slash:info")
            else:
                print(
                    "  /model usage:\n"
                    "    /model                         — show current model config\n"
                    "    /model list                    — list all models in registry\n"
                    "    /model use [role] <id>         — session-only switch\n"
                    "    /model default [role] <id>     — persist as default (writes yaml)\n",
                    file=sys.stderr,
                )
                yield _end("slash:info")
            return

        # ── Unknown command ───────────────────────────────────────────────────
        print(f"Unknown command: {cmd}  (try /help)", file=sys.stderr)
        yield _end("slash:unknown")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pop_last_user_msg(event: "TaskStartEvent") -> None:
    """Remove the trailing user message from state so slash commands don't
    pollute the persistent conversation history."""
    state = getattr(event, "state", None)
    if state is None:
        return
    if state.messages and state.messages[-1].role == "user":
        state.messages.pop()
    if state.raw_messages and state.raw_messages[-1].role == "user":
        state.raw_messages.pop()


def _print_model_config(model_config: "Any | None") -> None:
    """Print the current model config summary to stderr."""
    from pathlib import Path

    yaml_path = Path.home() / ".harnessx" / "model_config.yaml"

    if model_config is not None:
        DIM = "\033[2m"
        BLD = "\033[1m"
        NC = "\033[0m"
        CYN = "\033[36m"
        print(f"\n{BLD}  Current model config:{NC}", file=sys.stderr)
        for role, provider in model_config.models.items():
            pname = type(provider).__name__.replace("Provider", "")
            mname = getattr(provider, "model", None) or getattr(provider, "_model", "?")
            print(f"    {CYN}{role:<12}{NC}  {pname}/{mname}", file=sys.stderr)
        if yaml_path.exists():
            print(f"\n{DIM}  Config file: {yaml_path}{NC}", file=sys.stderr)
        print(
            f"\n{DIM}  /model use [role] <id>      — session-only switch"
            f"\n  /model default [role] <id>  — persist across sessions{NC}\n",
            file=sys.stderr,
        )
    else:
        print("  No model config available.", file=sys.stderr)


def _print_model_list() -> None:
    """Print all models from ~/.harnessx/model_config.yaml."""
    from pathlib import Path

    yaml_path = Path.home() / ".harnessx" / "model_config.yaml"
    if not yaml_path.exists():
        print(
            "  No model registry found at ~/.harnessx/model_config.yaml\n"
            "  Configure models in harnessx lab or set ANTHROPIC_API_KEY.",
            file=sys.stderr,
        )
        return

    try:
        import yaml as _yaml

        data = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  Could not read model registry: {exc}", file=sys.stderr)
        return

    DIM = "\033[2m"
    BLD = "\033[1m"
    NC = "\033[0m"
    CYN = "\033[36m"

    if data.get("schema_version") == 2:
        models = data.get("models", [])
        roles = data.get("roles", {})
        defaults = {rcfg.get("default") for rcfg in roles.values()}

        print(f"\n{BLD}  Models registry:{NC}", file=sys.stderr)
        for m in models:
            mid = m.get("id", "?")
            mstr = m.get("model", mid)
            prov = m.get("provider", "?")
            star = " *" if mid in defaults else ""
            print(f"    {CYN}{mid:<24}{NC}  {prov}/{mstr}{DIM}{star}{NC}", file=sys.stderr)

        if roles:
            print(f"\n{BLD}  Active roles:{NC}", file=sys.stderr)
            for role, rcfg in roles.items():
                default = rcfg.get("default", "?")
                ids = rcfg.get("model_ids", [])
                extra = f"  {DIM}({', '.join(ids)}){NC}" if len(ids) > 1 else ""
                print(f"    {CYN}{role:<12}{NC}  → {default}{extra}", file=sys.stderr)
        print("", file=sys.stderr)
    else:
        # Legacy format or unknown
        print(f"\n{BLD}  Model config (legacy format):{NC}", file=sys.stderr)
        for key, val in (data or {}).items():
            if key in ("schema_version", "fallback_key"):
                continue
            mname = val.get("model", "?") if isinstance(val, dict) else str(val)
            print(f"    {CYN}{key:<12}{NC}  {mname}", file=sys.stderr)
        print("", file=sys.stderr)


def _set_model_default(role: str, model_ref: str, model_config: "Any | None") -> str:
    """Update roles.<role>.default in ~/.harnessx/model_config.yaml.

    Creates the file from the current session's model_config if it doesn't
    exist yet.  Returns a human-readable status message.
    """
    from pathlib import Path

    path = Path.home() / ".harnessx" / "model_config.yaml"

    # Ensure file exists (create from current session config if needed)
    if not path.exists():
        if model_config is None:
            return (
                "No model config file found at ~/.harnessx/model_config.yaml\n  Configure models in harnessx lab first."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        model_config.to_yaml_file(path)

    text = path.read_text(encoding="utf-8")

    # Upgrade legacy format → current format if needed
    if "schema_version: 2" not in text:
        if model_config is None:
            return "Config file is legacy format; start a session with harnessx lab to upgrade."
        model_config.to_yaml_file(path)
        text = path.read_text(encoding="utf-8")

    # Try to update existing roles.<role>.default line
    # Match: two-space indent, role name, colon, then on its own or next lines "    default: <val>"
    # Simpler regex: find "    default: <val>" that appears within the role block.
    # We use a stateful approach: find "  <role>:" header, then update the "    default:" line after it.
    lines = text.splitlines(keepends=True)
    in_role = False
    role_found = False
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        # Detect entering the target role block (indent == 2)
        if indent == 2 and stripped == f"{role}:":
            in_role = True
            role_found = True
            new_lines.append(line)
            i += 1
            continue

        # If we're in the role block and hit another top-level or role-level key, exit
        if in_role and indent <= 2 and stripped and not stripped.startswith("#"):
            in_role = False

        # Update the default line inside the role block
        if in_role and indent == 4 and stripped.startswith("default:"):
            new_lines.append(f"    default: {model_ref}\n")
            i += 1
            continue

        new_lines.append(line)
        i += 1

    if role_found:
        new_text = "".join(new_lines)
    else:
        # Role doesn't exist — append it to the roles section
        # Find "roles:" and append after last role entry
        new_text = text.rstrip() + f"\n  {role}:\n    default: {model_ref}\n"

    path.write_text(new_text, encoding="utf-8")
    return (
        f"Default '{role}' model set to '{model_ref}'.\n"
        f"  Saved to: {path}\n"
        f"  Restart CLI or use '/model use {role} {model_ref}' to apply this session."
    )


async def _compact_session(session_id: str, workspace: object) -> None:
    """Run context compaction on the on-disk session state in-place."""
    from harnessx.tracing.journal import HarnessJournal
    from harnessx.processors.control.compaction import CompactionProcessor
    from harnessx.core.events import make_run_id, rough_token_count

    ws_root = getattr(workspace, "root", None)
    if ws_root is None:
        print("  /compact: workspace has no root directory.", file=sys.stderr)
        return

    try:
        state = HarnessJournal.wake(session_id, str(ws_root))
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"  /compact: cannot load session state: {e}", file=sys.stderr)
        return

    before = len(state.messages)
    before_tokens = rough_token_count(list(state.messages))

    compactor = CompactionProcessor()
    new_messages = await compactor._compact(state.messages, run_id=make_run_id())
    after = len(new_messages)
    after_tokens = rough_token_count(list(new_messages))

    if after == before:
        print(
            f"  Nothing to compact ({before} messages, {before_tokens} tokens).",
            file=sys.stderr,
        )
        return

    # Persist compacted state back to disk.
    import os as _os
    from pathlib import Path as _Path

    sessions_dir = _Path(ws_root) / "sessions"
    index_path = sessions_dir / f"{session_id}.json"
    if not index_path.exists():
        print("  /compact: could not locate session index.", file=sys.stderr)
        return

    import json as _json

    with open(index_path) as f:
        idx = _json.load(f)
    state_rel = idx.get("latest_state_path", "")
    if not state_rel:
        print("  /compact: could not locate state file.", file=sys.stderr)
        return

    state_path = _Path(ws_root) / state_rel
    if not state_path.exists():
        print("  /compact: state file not found.", file=sys.stderr)
        return

    with open(state_path) as f:
        snap = _json.load(f)
    from harnessx.core.events import message_to_dict

    snap["messages"] = [message_to_dict(m) for m in new_messages]
    if "raw_messages" in snap:
        snap["raw_messages"] = [message_to_dict(m) for m in new_messages]
    snap["segment_end_reason"] = "manual_compaction"
    tmp = str(state_path) + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(snap, f, ensure_ascii=False, indent=2)
    _os.replace(tmp, state_path)
    print(
        f"  Compacted: {before} → {after} messages  ({before_tokens} → {after_tokens} tokens)",
        file=sys.stderr,
    )
