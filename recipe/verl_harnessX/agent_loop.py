"""
HarnessX Agent Loop for veRL GRPO training.

All invocations use <tool_call>...</tool_call> format for tool execution
(WebSearch, WebFetch, Browser, Bash, CodeInterpreter, Read).
"""

import asyncio
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from verl_harnessX.tools import Tool, register_tools
from verl_harnessX.processor import EarlyCommitProcessor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_VERL_HARNESSX_ROOT = Path(__file__).resolve().parent

_TOOL_CONCURRENCY = int(os.getenv("VERL_TOOL_CONCURRENCY_PER_WORKER", "6"))
_tool_semaphore: asyncio.Semaphore | None = None


def _get_tool_semaphore() -> asyncio.Semaphore:
    global _tool_semaphore
    if _tool_semaphore is None:
        _tool_semaphore = asyncio.Semaphore(_TOOL_CONCURRENCY)
    return _tool_semaphore


def _harnessx_schema_to_openai(tool: Tool) -> dict:
    schema = tool.to_schema()
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.input_schema,
        },
    }


_FORCE_ANSWER_MSG = (
    "You have reached the maximum number of interaction turns. "
    "You must provide your answer in <answer>...</answer> tags now. "
    "Do NOT call any tools."
)


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the HarnessX agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
    ):
        self.messages = messages
        self.metrics = metrics
        self.request_id = request_id

        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.tool_calls: list[FunctionCall] = []
        self.total_tool_calls: int = 0

        self.routed_experts = None
        self.extra_fields: dict[str, Any] = {}

        self.force_answer_issued = False
        self._track_logprobs = False


@register("harnessx_agent")
class HarnessXAgentLoop(AgentLoopBase):
    """veRL agent loop with unified <tool_call> pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        self.force_answer_remaining_tokens = getattr(
            self.rollout_config.multi_turn, "force_answer_remaining_tokens", 1000
        )

        self._stop_token_ids = self.tokenizer.encode("</tool_call>", add_special_tokens=False) + [
            self.tokenizer.eos_token_id
        ]
        self._gen_prompt_ids = self.tokenizer.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False)

        self._init_tools()

    def _encode_turn_messages(self, messages: list[dict[str, Any]]) -> list[int]:
        """Encode tool/user messages into token IDs for Qwen3.5 format.

        Bypasses apply_chat_template which is broken for standalone tool messages
        due to veRL's Qwen3.5 workaround returning BatchEncoding.

        Tool responses and user messages (e.g. nudge) are merged into a single
        <|im_start|>user block to avoid consecutive user turns.
        """
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        user_msgs = [m for m in messages if m.get("role") == "user"]

        inner_parts = []
        if tool_msgs:
            inner_parts.append(
                "".join(f"<tool_response>\n{m['content']}\n</tool_response>\n" for m in tool_msgs).rstrip()
            )
        for m in user_msgs:
            inner_parts.append(m["content"])

        if inner_parts:
            inner = "\n\n".join(inner_parts)
            text = f"<|im_start|>user\n{inner}<|im_end|>\n"
        else:
            text = ""

        turn_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return turn_ids + list(self._gen_prompt_ids)

    def _truncate_by_tokens(self, text: str, max_tokens: int, side: str = "left") -> str:
        """Truncate text to fit within max_tokens using the tokenizer."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        if side == "left":
            truncated_ids = token_ids[:max_tokens]
            return self.tokenizer.decode(truncated_ids, skip_special_tokens=True) + "...(truncated)"
        elif side == "right":
            truncated_ids = token_ids[-max_tokens:]
            return "(truncated)..." + self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        else:
            half = max_tokens // 2
            head = self.tokenizer.decode(token_ids[:half], skip_special_tokens=True)
            tail = self.tokenizer.decode(token_ids[-half:], skip_special_tokens=True)
            return head + "...(truncated)..." + tail

    def _init_tools(self):
        tool_names = getattr(self.rollout_config.multi_turn, "tool_names", None)
        self.harnessx_registry = register_tools(tool_names)

        self._tool_map: dict[str, Tool] = {}
        for name in self.harnessx_registry.list_names():
            self._tool_map[name] = self.harnessx_registry._tools[name]

        self.tool_schemas = [_harnessx_schema_to_openai(t) for t in self._tool_map.values()]

        self._tool_names = list(self._tool_map.keys())

        self._early_commit = EarlyCommitProcessor(max_turns=self.max_assistant_turns)

        logger.info("HarnessX tools initialized: %s", self._tool_names)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        metrics = {}
        request_id = uuid4().hex

        agent_data = AgentData(
            messages=messages,
            metrics=metrics,
            request_id=request_id,
        )

        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, sampling_params)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data={},
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length] if agent_data.response_logprobs else None
            ),
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "reqs_ids": request_id,
            }
        )

        total_resp = len(agent_data.response_mask[: self.response_length])
        mask_1 = sum(agent_data.response_mask[: self.response_length])
        mask_0 = total_resp - mask_1
        truncated = len(agent_data.response_mask) > self.response_length
        logger.info(
            "Seq done | req=%s turns=%d tool_calls=%d resp_tokens=%d (model=%d env=%d) truncated=%s max=%d",
            request_id[:8],
            agent_data.assistant_turns,
            agent_data.total_tool_calls,
            total_resp,
            mask_1,
            mask_0,
            truncated,
            self.response_length,
        )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=None,
            videos=None,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        gen_params = dict(sampling_params)
        gen_params["stop_token_ids"] = self._stop_token_ids
        max_per_turn = self.rollout_config.multi_turn.max_new_tokens_per_turn
        if max_per_turn:
            gen_params["max_new_tokens"] = max_per_turn

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=gen_params,
                image_data=None,
                video_data=None,
            )

        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        n_gen = len(agent_data.response_ids)
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * n_gen
        if output.log_probs:
            agent_data._track_logprobs = True
            agent_data.response_logprobs += output.log_probs
        elif agent_data._track_logprobs:
            agent_data.response_logprobs += [0.0] * n_gen

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED

        if agent_data.force_answer_issued:
            return AgentState.TERMINATED

        needs_force = False
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            needs_force = True
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            needs_force = True
        remaining = self.response_length - len(agent_data.response_mask)
        if remaining < self.force_answer_remaining_tokens:
            needs_force = True

        if needs_force:
            return await self._force_answer_and_terminate(agent_data, sampling_params)

        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
            agent_data.response_ids,
        )
        agent_data.total_tool_calls += len(agent_data.tool_calls)

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        add_messages: list[dict[str, Any]] = []

        tasks = [self._call_tool(tool_call) for tool_call in agent_data.tool_calls[: self.max_parallel_calls]]

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks) if tasks else []

        for tool_response, tool_reward in responses:
            message = {"role": "tool", "content": tool_response.text or ""}
            add_messages.append(message)
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        nudge = self._early_commit.check(agent_data.assistant_turns)
        if nudge:
            add_messages.append({"role": "user", "content": nudge})

        agent_data.messages.extend(add_messages)

        response_ids = self._encode_turn_messages(add_messages)

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data._track_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _force_answer_and_terminate(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        force_msg = {"role": "user", "content": _FORCE_ANSWER_MSG}
        agent_data.messages.append(force_msg)

        force_ids = self._encode_turn_messages([force_msg])

        if len(agent_data.response_mask) + len(force_ids) >= self.response_length:
            return AgentState.TERMINATED

        agent_data.prompt_ids += force_ids
        agent_data.response_mask += [0] * len(force_ids)
        if agent_data._track_logprobs:
            agent_data.response_logprobs += [0.0] * len(force_ids)

        agent_data.force_answer_issued = True
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _call_tool(self, tool_call: FunctionCall) -> tuple[ToolResponse, Optional[float]]:
        tool_name = tool_call.name
        if tool_name not in self._tool_map:
            msg = f"Unknown tool '{tool_name}'. Available tools: {self._tool_names}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0

        try:
            tool_args = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            msg = f"Invalid JSON in arguments for '{tool_name}': {e}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0

        import time as _time

        _TOOL_CALL_TIMEOUT = int(os.getenv("VERL_TOOL_CALL_TIMEOUT", "80"))
        # Browser already serialized by _op_lock — skip semaphore to avoid holding slots while waiting for lock
        _skip_sem = tool_name.lower() == "browser"

        async def _execute_with_timeout():
            t_start = _time.monotonic()
            try:
                result = await asyncio.wait_for(
                    self.harnessx_registry.execute(tool_name, tool_args),
                    timeout=_TOOL_CALL_TIMEOUT,
                )
                elapsed = _time.monotonic() - t_start
                if elapsed > 10.0:
                    logger.warning(
                        "[TOOL_DEBUG] exec=%.1fs tool=%s (slow exec)",
                        elapsed,
                        tool_name,
                    )
                return result
            except asyncio.TimeoutError:
                elapsed = _time.monotonic() - t_start
                logger.warning("[TOOL_DEBUG] HARD_TIMEOUT tool=%s exec=%.1fs", tool_name, elapsed)
                raise

        try:
            if _skip_sem:
                result = await _execute_with_timeout()
            else:
                sem = _get_tool_semaphore()
                t_wait_start = _time.monotonic()
                async with sem:
                    sem_wait = _time.monotonic() - t_wait_start
                    if sem_wait > 2.0:
                        logger.warning(
                            "[TOOL_DEBUG] sem_wait=%.1fs tool=%s (slow acquire)",
                            sem_wait,
                            tool_name,
                        )
                    result = await _execute_with_timeout()
            text = result.output if result.output else ""
            if result.error:
                text = f"Error: {result.error}"
        except asyncio.TimeoutError:
            return ToolResponse(text=f"Error: tool '{tool_name}' timed out after {_TOOL_CALL_TIMEOUT}s"), 0.0
        except Exception as e:
            logger.warning(f"Error executing tool '{tool_name}': {e}")
            return ToolResponse(text=f"Error executing tool '{tool_name}': {e}"), 0.0

        text = self._truncate_by_tokens(text, self.max_tool_response_length, self.tool_response_truncate_side)

        return ToolResponse(text=text), None
