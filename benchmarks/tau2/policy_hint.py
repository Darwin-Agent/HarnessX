"""PolicyHintProcessor — Scans conversation history at each step start and injects
policy compliance reminders into system_prompt when unresolved issues are found.

Supported rules (all for mobile_data_issue tasks):

1. data_exceeded + refuel_data not yet called
2. data_balance_not_checked: calling line seen with data_used_gb > 0, but get_data_usage
   not yet called; prompts agent to verify quota (triggers data_exceeded detection)
3. roaming_disabled_abroad + enable_roaming not yet called
   Gate: only fires when (data_used_gb == 0 for this line) OR user_abroad keywords detected.
   Rationale: Line.roaming_enabled defaults to False for ALL lines; using data_used_gb==0 as
   a discriminator prevents false-positive roaming hints on data_usage_exceeded tasks.
   Phase 2: after enable_roaming called, remind agent to guide user to toggle device roaming
4. check_correct_line: agent queried a Line (via get_details_by_id) whose phone_number ≠ reported_phone;
   redirect agent to find and check the line matching reported_phone
5. get_data_usage_unverified: agent called get_data_usage on a line that is NOT exceeded, but
   calling_line_seen=False (correct line identity not confirmed yet); tell agent to call
   get_details_by_id to verify line identity before concluding no data issue
6. airplane_mode_on + toggle_airplane_mode not yet called  [NOTE: fires only if tool results
   contain airplane_mode=True, which requires agent-side tools; rarely triggers]
7. data_disabled + toggle_data not yet called  [NOTE: same as above, rarely triggers]
8. vpn_connected + disconnect_vpn not yet called  [NOTE: same as above, rarely triggers]

Design principle: Only generate positive/affirmative directives ("call X to fix Y").
Do NOT generate prohibitive hints ("do not do Z") — Haiku 4.5 ignores them.

All hint text is in English for better model compliance with Haiku 4.5.
Hints are prepended to system_prompt (not appended) so they receive stronger attention.

Key implementation notes
------------------------
- The 'location' field NEVER appears in agent tool results (it's a user-side field);
  do not use it for location detection.
- Use _detect_user_abroad() for keyword-based abroad detection (backup for combos).
- Use data_used_gb == 0 as the primary gate for roaming hints (roaming tasks have 0 usage).
- calling_line_seen is only set True for LINE results (those with roaming_enabled ≠ None),
  NOT for Customer results — Customer.phone_number matches reported_phone but is NOT a Line.
- get_details_by_id uses "id" param (not "line_id"); _detect_actions_taken uses id as fallback.

Implementation
--------------
- hook: step_start (order=2), runs before TokenBudgetProcessor (default order=10)
- scans raw_messages tool results (JSON parsed) for conditions and context (line_id, customer_id)
- scans raw_messages assistant tool_calls for actions already taken on correct lines
- prepends pending reminders to event.system_prompt
- passes through unchanged if no pending reminders
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator

from harnessx.core.events import StepStartEvent
from harnessx.core.processor import MultiHookProcessor

if TYPE_CHECKING:
    from harnessx.core.events import Message

logger = logging.getLogger(__name__)


# ─── Condition records ───────────────────────────────────────────────────────


@dataclass
class ConditionRecord:
    """A single detected condition with the context needed for remediation."""

    condition: str  # condition type, e.g. "data_exceeded"
    line_id: str = ""  # affected line ID
    customer_id: str = ""  # affected customer ID
    extra: dict = field(default_factory=dict)  # extra diagnostic info


def _parse_tool_result(content: str) -> dict:
    """Try to parse tool result content as a dict; return {} on failure."""
    if not content:
        return {}
    try:
        result = json.loads(content)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


_PHONE_RE = re.compile(r"\b(\d{3}[-.\s]\d{3}[-.\s]\d{4}|\+?1?\d{10,11})\b")

# Keywords indicating user is abroad/traveling (for roaming hint gating)
_ABROAD_KEYWORDS = (
    "abroad",
    "roaming",
    "international",
    "overseas",
    "traveling",
    "travelling",
    "foreign",
    "outside the country",
    "outside my country",
    "europe",
    "asia",
    "another country",
    "different country",
    "vacation",
    "trip",
    "travel",
    "holiday",
)


def _extract_reported_phone(messages: tuple["Message", ...]) -> str:
    """Extract the phone number reported by the user from user/system messages (first match)."""
    for msg in messages:
        if msg.role not in ("user", "system"):
            continue
        m = _PHONE_RE.search(str(msg.content or ""))
        if m:
            return m.group(1)
    return ""


def _detect_user_abroad(messages: tuple["Message", ...]) -> bool:
    """Return True only if user messages contain positive evidence of being abroad.

    The Line data model has no 'location' field, so we cannot rely on tool results
    for location detection. We must scan user messages for abroad/travel keywords.
    This prevents false-positive roaming hints for non-roaming tasks (e.g. data_usage_exceeded)
    where roaming_enabled=False is simply the default Line value.
    """
    for msg in messages:
        if msg.role != "user":
            continue
        content = str(msg.content or "").lower()
        if any(kw in content for kw in _ABROAD_KEYWORDS):
            return True
    return False


def _detect_conditions(messages: tuple["Message", ...]) -> list[ConditionRecord]:
    """Scan tool results and return the list of detected conditions (with context).

    Detection strategy
    ------------------
    - Extract reported_phone from user messages
    - Cache customer_id from Customer tool results (Customer results contain customer_id + line_ids)
    - roaming_disabled_abroad only fires for lines whose phone_number matches reported_phone,
      avoiding false positives on unrelated lines that happen to have roaming_enabled=False by default
    """
    records: list[ConditionRecord] = []
    seen: set[tuple] = set()  # dedup key: (condition, line_id)

    # ── Pre-scan: extract phone number, customer_id, calling_line_seen ──────
    reported_phone = _extract_reported_phone(messages)
    # Gate roaming hints on positive evidence of user being abroad.
    # Line.roaming_enabled defaults to False for ALL lines, so we must NOT fire the
    # roaming hint unless we have evidence the user is actually abroad — otherwise
    # data_usage_exceeded and other non-roaming tasks get false-positive roaming hints.
    user_abroad: bool = _detect_user_abroad(messages)
    cached_customer_id: str = ""
    calling_line_seen: bool = False  # True if a line result matching reported_phone has been seen
    plan_data_limits: dict[str, float] = {}  # plan_id → data_limit_gb (from Plan results)
    data_usage_checked_lines: set[str] = set()  # line_ids where get_data_usage result seen
    for msg in messages:
        if msg.role != "tool":
            continue
        data = _parse_tool_result(msg.content)
        if not data:
            continue
        # Customer tool result signature: contains customer_id + line_ids list
        if data.get("customer_id") and isinstance(data.get("line_ids"), list):
            cached_customer_id = str(data["customer_id"])
        # Detect whether the target line has been retrieved (phone_number matches reported_phone)
        # IMPORTANT: Only count LINE results (those with roaming_enabled), NOT Customer results.
        # Customer objects also have phone_number (the customer's primary phone) which can match
        # reported_phone — but seeing the customer's phone in a Customer result does NOT mean
        # we've actually retrieved the correct Line. We only count get_details_by_id-style Line
        # results which carry roaming_enabled as a discriminating field.
        line_phone_pre = str(data.get("phone_number") or "")
        is_line_result_prescan = data.get("roaming_enabled") is not None
        if reported_phone and line_phone_pre and line_phone_pre == reported_phone and is_line_result_prescan:
            calling_line_seen = True
        # Cache Plan data limits (Plan results contain plan_id + data_limit_gb)
        if data.get("plan_id") and data.get("data_limit_gb") is not None:
            plan_data_limits[str(data["plan_id"])] = _to_float(data["data_limit_gb"])
        # Record lines where get_data_usage result has been seen (results contain data_used_gb + data_limit_gb + line_id)
        if data.get("line_id") and data.get("data_used_gb") is not None and data.get("data_limit_gb") is not None:
            data_usage_checked_lines.add(str(data["line_id"]))

    # ── Main scan: detect each condition ────────────────────────────────────
    for msg in messages:
        if msg.role != "tool":
            continue
        data = _parse_tool_result(msg.content)
        if not data:
            continue

        line_id = str(data.get("line_id") or data.get("id") or "")
        customer_id = str(data.get("customer_id") or cached_customer_id or "")

        # phone number in the current tool result (Line objects contain phone_number)
        line_phone = str(data.get("phone_number") or "")
        # only enforce matching when both reported_phone and line_phone are available;
        # if either is empty, do not restrict (conservative trigger)
        is_calling_line = not reported_phone or not line_phone or line_phone == reported_phone

        def _add(condition: str, **extra):
            key = (condition, line_id)
            if key not in seen:
                seen.add(key)
                records.append(
                    ConditionRecord(
                        condition=condition,
                        line_id=line_id,
                        customer_id=customer_id,
                        extra=extra,
                    )
                )

        # ── Data exceeded ─────────────────────────────────────────────────────
        # Three formats supported:
        # 1. data_remaining (direct field)
        # 2. data_used_gb + data_limit_gb (get_data_usage response format)
        # 3. data_used_gb + plan_id cross-reference (get_details_by_id Line result + cached Plan limit)
        data_remaining = data.get("data_remaining")
        if data_remaining is None:
            data_used_gb = data.get("data_used_gb")
            data_limit_gb = data.get("data_limit_gb")
            # cross-reference: Line result has plan_id but no data_limit_gb — look up from cached Plan
            if data_used_gb is not None and data_limit_gb is None:
                plan_id_ref = str(data.get("plan_id") or "")
                if plan_id_ref and plan_id_ref in plan_data_limits:
                    data_limit_gb = plan_data_limits[plan_id_ref]
            if data_used_gb is not None and data_limit_gb is not None:
                total_gb = _to_float(data_limit_gb) + _to_float(data.get("data_refueling_gb") or 0)
                data_remaining = total_gb - _to_float(data_used_gb)
        data_status = str(data.get("data_status", "")).lower()
        if (data_remaining is not None and _to_float(data_remaining) <= 0) or any(
            kw in data_status for kw in ("exceeded", "exhausted", "depleted")
        ):
            _add("data_exceeded", data_remaining=data_remaining)

        # ── Data balance not checked ──────────────────────────────────────────
        # If the calling line has been retrieved but get_data_usage not yet called, remind agent to verify quota
        # Only fires when is_calling_line=True and the line has not yet been checked by get_data_usage
        # Require data_used_gb > 0 to avoid triggering for lines with default 0.0 usage
        # (e.g. roaming tasks where the line has no significant data usage)
        raw_data_used = data.get("data_used_gb")
        data_used_for_line = _to_float(raw_data_used) if raw_data_used is not None else 0.0
        if raw_data_used is not None and data_used_for_line > 0 and is_calling_line:
            if line_id and line_id not in data_usage_checked_lines:
                _add("data_balance_not_checked")

        # ── get_data_usage not exceeded but line identity unconfirmed ────────
        # When agent calls get_data_usage(C, LX) and LX is NOT exceeded, but
        # calling_line_seen=False (no Line result with phone_number==reported_phone seen yet),
        # the agent may have queried the WRONG line (e.g. L1001 with 3.2/5.0 GB, while the
        # actual problem line L1002 has 15.1 GB exceeded). We can't tell from the get_data_usage
        # result alone because it has NO phone_number field — only line_id + data_used_gb +
        # data_limit_gb. Redirect the agent to verify line identity via get_details_by_id.
        # Discriminator: get_data_usage results have data_used_gb + data_limit_gb + line_id
        # but NO phone_number and NO roaming_enabled (unlike get_details_by_id Line results).
        if (
            raw_data_used is not None
            and data.get("data_limit_gb") is not None
            and data.get("phone_number") is None  # not a get_details_by_id result
            and data.get("roaming_enabled") is None  # not a get_details_by_id result
            and not calling_line_seen
            and reported_phone
            and line_id
        ):
            limit_val = _to_float(data.get("data_limit_gb"))
            refuel_val = _to_float(data.get("data_refueling_gb") or 0)
            if data_used_for_line < (limit_val + refuel_val):  # NOT exceeded
                _add(
                    "get_data_usage_unverified",
                    queried_line=line_id,
                    reported_phone=reported_phone,
                )

        # ── Wrong line: redirect agent to the line matching reported_phone ──
        # If the agent queries a line whose phone_number ≠ reported_phone and
        # the correct calling line hasn't been seen yet, redirect the agent.
        # This is a GENERAL check with NO roaming gate — it must fire even when
        # the wrong line has data_used_gb > 0 (e.g. data_usage_exceeded tasks
        # where L1001 has 3.2 GB used but L1002 at 15.1 GB is the target).
        # The original check_correct_line was gated behind the roaming condition,
        # which caused it to silently miss data_usage_exceeded scenarios.
        if (
            not is_calling_line
            and not calling_line_seen
            and reported_phone
            and line_phone
            and line_phone != reported_phone
            and data.get("roaming_enabled") is not None
        ):  # proxy: only Line results have roaming_enabled
            _add(
                "check_correct_line",
                wrong_line_id=line_id,
                wrong_phone=line_phone,
                reported_phone=reported_phone,
            )

        # ── Roaming disabled (user abroad) ───────────────────────────────────
        # IMPORTANT: Line.roaming_enabled defaults to False for ALL lines.
        # Gate: fire roaming hint only when data_used_gb == 0 (roaming task) OR user_abroad detected.
        # This prevents false-positives on data_usage_exceeded tasks (data_used_gb > 0).
        # Roaming tasks always have data_used_gb=0.0 (default) since data usage is not the issue.
        # Combo tasks (roaming + data exceeded) are covered by user_abroad keyword detection.
        # The 'location' field never appears in agent tool results, so it cannot be used.
        roaming = data.get("roaming_enabled")
        if roaming is not None and (data_used_for_line == 0 or user_abroad):
            roaming_off = roaming is False or str(roaming).lower() in (
                "false",
                "0",
                "disabled",
                "off",
            )
            if roaming_off:
                if is_calling_line:
                    _add("roaming_disabled_abroad")
                # Note: check_correct_line for non-calling roaming lines is now handled
                # by the general check above (no separate check needed here)

        # ── Airplane mode on ─────────────────────────────────────────────────
        airplane = data.get("airplane_mode")
        if airplane is not None:
            if airplane is True or str(airplane).lower() in (
                "true",
                "1",
                "enabled",
                "on",
            ):
                _add("airplane_mode_on")

        # ── Mobile data disabled ─────────────────────────────────────────────
        data_enabled = data.get("data_enabled")
        if data_enabled is not None:
            if data_enabled is False or str(data_enabled).lower() in (
                "false",
                "0",
                "disabled",
                "off",
            ):
                _add("data_disabled")

        # ── VPN connected (may interfere with data) ──────────────────────────
        vpn_connected = data.get("vpn_connected")
        if vpn_connected is not None:
            if vpn_connected is True or str(vpn_connected).lower() in (
                "true",
                "1",
                "connected",
            ):
                _add("vpn_connected")

        # ── Abnormal network mode ────────────────────────────────────────────
        network_mode = str(data.get("network_mode_preference", "")).lower()
        if network_mode and network_mode in ("2g", "2g_only", "gsm_only"):
            _add("bad_network_mode", current=network_mode)

    return records


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _detect_actions_taken(messages: tuple["Message", ...]) -> dict[str, set[str]]:
    """Scan assistant messages and return {tool_name: {line_id, ...}} of already-taken actions."""
    actions: dict[str, set[str]] = {}
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if not tc.name:
                    continue
                # get_details_by_id uses "id" param; get_data_usage/refuel_data use "line_id"
                line_id = str(tc.input.get("line_id") or tc.input.get("id") or "") if tc.input else ""
                actions.setdefault(tc.name, set()).add(line_id)
    return actions


# ─── Hint generation ─────────────────────────────────────────────────────────

# (condition, remediation_tools, hint_template)
# hint_template placeholders: {line_id}, {customer_id}, and extras from ConditionRecord.extra
_HINT_RULES: list[tuple[str, set[str], str]] = [
    (
        "data_exceeded",
        {"refuel_data"},
        "⚠️ [POLICY ALERT] Line {line_id} has exhausted its data quota (data_remaining=0). "
        "Per policy, you MUST call refuel_data(customer_id={customer_id}, line_id={line_id}) "
        "to add a data package. Without this, the customer's data connection cannot be restored "
        "even if other issues are fixed.",
    ),
    (
        "roaming_disabled_abroad",
        {"enable_roaming"},  # switches to step-2 hint after enable_roaming; see _build_hints
        "⚠️ [POLICY ALERT] Line {line_id} has roaming disabled and the customer is abroad. "
        "Per policy, TWO steps are required: "
        "Step 1 — call enable_roaming(customer_id={customer_id}, line_id={line_id}); "
        "Step 2 — immediately tell the customer: "
        "'Please open your phone Settings, go to Mobile Data / Cellular, and turn ON the Roaming toggle.' "
        "Complete BOTH steps before doing any other diagnostics.",
    ),
    (
        "check_correct_line",
        set(),  # no direct remediation tool — this is a guidance hint
        "⚠️ [POLICY ALERT] You retrieved line {wrong_line_id} (phone {wrong_phone}), "
        "but the customer reported their number as {reported_phone}. "
        "You MUST find and investigate the line matching {reported_phone}. "
        "Check the customer's other lines by calling get_details_by_id for each remaining line_id "
        "until you find the one whose phone_number matches {reported_phone}. "
        "Do NOT make any changes or conclusions until you have checked the correct line.",
    ),
    (
        "airplane_mode_on",
        {"toggle_airplane_mode"},
        "⚠️ [POLICY ALERT] Line {line_id} has airplane mode ON, which blocks all wireless connections. "
        "Per policy, you MUST call toggle_airplane_mode(customer_id={customer_id}, line_id={line_id}) "
        "to disable airplane mode.",
    ),
    (
        "data_disabled",
        {"toggle_data"},
        "⚠️ [POLICY ALERT] Line {line_id} has mobile data turned OFF. "
        "Per policy, you MUST call toggle_data(customer_id={customer_id}, line_id={line_id}) "
        "to re-enable data.",
    ),
    (
        "vpn_connected",
        {"disconnect_vpn"},
        "⚠️ [POLICY ALERT] Line {line_id} has an active VPN connection that may interfere with data routing. "
        "Per policy, you MUST call disconnect_vpn(customer_id={customer_id}, line_id={line_id}) "
        "to disconnect the VPN.",
    ),
    (
        "bad_network_mode",
        {"set_network_mode_preference"},
        "⚠️ [POLICY ALERT] Line {line_id} has a suboptimal network mode setting (current: {current}), "
        "which reduces connection speed. "
        "Per policy, call set_network_mode_preference(customer_id={customer_id}, line_id={line_id}) to fix it.",
    ),
    (
        "data_balance_not_checked",
        {"get_data_usage"},
        "⚠️ [POLICY ALERT] Data balance for line {line_id} has not been verified. "
        "Call get_data_usage(customer_id={customer_id}, line_id={line_id}) to check if the data quota is exceeded.",
    ),
    (
        "get_data_usage_unverified",
        set(),  # no single remediation tool — need to verify identity first
        "⚠️ [POLICY ALERT] Line {queried_line} data usage is within limits (not exceeded). "
        "However, you have NOT yet confirmed which line corresponds to the customer's reported "
        "phone {reported_phone}. Before concluding there is no data issue, verify line identity: "
        "call get_details_by_id(id={queried_line}) to check its phone number. "
        "The actual problem line may be a different line in the customer's account.",
    ),
]

_RULE_MAP = {cond: (tools, tmpl) for cond, tools, tmpl in _HINT_RULES}

# roaming_disabled_abroad phase 2: system roaming enabled; remind agent to guide user to toggle device roaming
_ROAMING_STEP2_TMPL = (
    "⚠️ [POLICY ALERT — ACTION REQUIRED NOW] enable_roaming for line {line_id} is COMPLETE. "
    "Your very next message to the customer MUST say: "
    "'Please open your phone Settings, find Mobile Data or Cellular settings, "
    "and turn ON the Roaming toggle.' "
    "Wait for the customer to confirm they have done this before proceeding. "
    "Do NOT suggest speed tests, LTE settings, or other diagnostics first."
)


def _build_hints(
    conditions: list[ConditionRecord],
    actions_taken: dict[str, set[str]],
) -> str:
    """Build reminder text for unresolved conditions.

    roaming_disabled_abroad uses two-phase logic:
    - Phase 1 (enable_roaming not yet called): full two-step fix instructions
    - Phase 2 (enable_roaming already called): urgent reminder to guide user to toggle device roaming

    check_correct_line is suppressed if roaming_disabled_abroad is already detected
    (meaning the correct line has been identified) or enable_roaming has been called.
    """
    pending: list[str] = []

    # Determine if the correct calling line has already been identified
    has_roaming_disabled_abroad = any(r.condition == "roaming_disabled_abroad" for r in conditions)
    enable_roaming_called = bool(actions_taken.get("enable_roaming"))
    has_data_exceeded = any(r.condition == "data_exceeded" for r in conditions)

    for rec in conditions:
        rule = _RULE_MAP.get(rec.condition)
        if not rule:
            continue
        remediation_tools, tmpl = rule

        if rec.condition == "roaming_disabled_abroad":
            enable_called = rec.line_id in actions_taken.get("enable_roaming", set())
            if enable_called:
                # Phase 2: system roaming fixed; remind agent to guide user to toggle device roaming
                hint = _ROAMING_STEP2_TMPL.format(
                    line_id=rec.line_id or "?",
                    customer_id=rec.customer_id or "?",
                )
                pending.append(hint)
            else:
                # Phase 1: enable_roaming not yet called; give full two-step instructions
                hint = tmpl.format(
                    line_id=rec.line_id or "?",
                    customer_id=rec.customer_id or "?",
                    **rec.extra,
                )
                pending.append(hint)
            continue

        if rec.condition == "check_correct_line":
            # Suppress if the correct line is already identified or enable_roaming is done
            if has_roaming_disabled_abroad or enable_roaming_called:
                continue
            hint = tmpl.format(
                line_id=rec.line_id or "?",
                customer_id=rec.customer_id or "?",
                **rec.extra,
            )
            pending.append(hint)
            continue

        if rec.condition == "data_balance_not_checked":
            # Suppress if data_exceeded is already detected (covered by that hint)
            # or if refuel_data has already been called (data already fixed)
            if has_data_exceeded:
                continue
            if rec.line_id in actions_taken.get("refuel_data", set()):
                continue
            # Standard suppression: get_data_usage already called for this line
            if rec.line_id in actions_taken.get("get_data_usage", set()):
                continue
            hint = tmpl.format(
                line_id=rec.line_id or "?",
                customer_id=rec.customer_id or "?",
                **rec.extra,
            )
            pending.append(hint)
            continue

        if rec.condition == "get_data_usage_unverified":
            # Suppress if data issue was identified elsewhere (data_exceeded fires)
            if has_data_exceeded:
                continue
            # Suppress if refuel_data already called (data issue fixed)
            if actions_taken.get("refuel_data"):
                continue
            # Suppress if get_details_by_id was already called for the queried line
            # (line identity verified — the hint's purpose has been fulfilled)
            queried_line = rec.extra.get("queried_line", rec.line_id or "")
            if queried_line in actions_taken.get("get_details_by_id", set()):
                continue
            hint = tmpl.format(
                line_id=rec.line_id or "?",
                customer_id=rec.customer_id or "?",
                **rec.extra,
            )
            pending.append(hint)
            continue

        # Standard check: the remediation tool has already been called for this line
        already_done = any(rec.line_id in actions_taken.get(tool, set()) for tool in remediation_tools)
        if already_done:
            continue
        hint = tmpl.format(
            line_id=rec.line_id or "?",
            customer_id=rec.customer_id or "?",
            **rec.extra,
        )
        pending.append(hint)
    return "\n".join(pending)


# ─── Processor ───────────────────────────────────────────────────────────────


class PolicyHintProcessor(MultiHookProcessor):
    """Injects policy compliance reminders before each step to help the model follow domain rules.

    Runs at step_start (order=2), before TokenBudgetProcessor (default order=10).
    Hints include the specific line_id / customer_id so the model acts on the correct line.

    Args:
        enabled: whether to enable the processor (default True); set False to disable in tests.
    """

    _singleton_group = "policy.hint"
    _order = 2

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    async def on_step_start(self, event: StepStartEvent) -> AsyncIterator[StepStartEvent]:
        if not self.enabled or not event.raw_messages:
            yield event
            return

        conditions = _detect_conditions(event.raw_messages)
        if not conditions:
            yield event
            return

        actions_taken = _detect_actions_taken(event.raw_messages)
        hints = _build_hints(conditions, actions_taken)

        if not hints:
            yield event
            return

        logger.info(
            "PolicyHintProcessor: %d conditions, %d pending hints — %s",
            len(conditions),
            hints.count("⚠️"),
            "; ".join(c.condition for c in conditions),
        )
        logger.debug("PolicyHintProcessor hints:\n%s", hints)
        # Prepend hints so they receive strong attention (not buried at the end)
        enhanced_system = hints + "\n\n" + event.system_prompt if event.system_prompt else hints
        yield dataclasses.replace(event, system_prompt=enhanced_system)
