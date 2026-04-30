"""
ToolRewardManager for veRL GRPO training.

Reward components (8:1:1):
  - Accuracy reward (0.8): correctness of the final answer (with 5% numeric tolerance)
  - Format reward (0.1): well-formed tool calls, <think> blocks, etc.
  - Tool call reward (0.1): any valid tool call in the response
"""

import logging
import re
from collections import defaultdict

import torch

from verl import DataProto

logger = logging.getLogger(__name__)

ACCURACY_REWARD_WEIGHT = 0.8
FORMAT_REWARD_WEIGHT = 0.1
TOOL_CALL_REWARD_WEIGHT = 0.1


# ============================================================
#  LLM Judge for accuracy reward (commented out — enable when
#  judge model is deployed)
# ============================================================
#
# _JUDGE_PROMPT = (
#     "You are a strict factual-accuracy judge. Your task is to determine whether the "
#     "model's answer is semantically equivalent to the ground-truth answer.\n\n"
#     "Question:\n{question}\n\n"
#     "Ground-truth answer:\n{ground_truth}\n\n"
#     "Model's answer:\n{model_answer}\n\n"
#     "Judging criteria:\n"
#     "1. SEMANTIC EQUIVALENCE: The model's answer must convey the same meaning as the "
#     "ground-truth. Minor wording, formatting, or ordering differences are acceptable "
#     "as long as the core facts agree.\n"
#     "2. NUMERIC TOLERANCE: For numeric answers, a relative error within 5% is acceptable. "
#     "For example, if the ground-truth is '100', answers between 95 and 105 are CORRECT.\n"
#     "3. GROUNDING/BBOX TOLERANCE: For spatial grounding or bounding box answers, "
#     "an overlap (IoU) greater than 70% with the ground-truth region is acceptable.\n"
#     "4. COMPLETENESS: The answer must contain the key information from the ground-truth. "
#     "Extra correct context is fine; missing key facts is not.\n\n"
#     "Reply with EXACTLY one word on the first line: CORRECT or INCORRECT\n"
#     "Then a short reason on the next line."
# )
#
#
# def _call_llm_judge(
#     question: str,
#     model_answer: str,
#     ground_truth: str,
#     api_base: str = None,
#     api_key: str = None,
#     model: str = "gpt-4o-mini",
#     timeout: float = 30.0,
# ) -> float:
#     """Call LLM judge to evaluate answer accuracy. Returns 1.0 or 0.0."""
#     import httpx
#
#     api_base = api_base or os.environ.get("OPENAI_API_BASE", "http://your-api-endpoint/v1")
#     api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
#
#     prompt = _JUDGE_PROMPT.format(
#         question=question[:500],
#         ground_truth=ground_truth[:300],
#         model_answer=model_answer[:500],
#     )
#
#     try:
#         resp = httpx.post(
#             f"{api_base.rstrip('/')}/chat/completions",
#             headers={"Authorization": f"Bearer {api_key}"},
#             json={
#                 "model": model,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.0,
#                 "max_tokens": 512,
#             },
#             timeout=timeout,
#         )
#         resp.raise_for_status()
#         text = resp.json()["choices"][0]["message"]["content"].strip()
#         first_line = text.split("\n")[0].strip().upper()
#         if "CORRECT" in first_line and "INCORRECT" not in first_line:
#             return 1.0
#         return 0.0
#     except Exception as e:
#         logger.warning("LLM judge call failed: %s", e)
#         return 0.0
#
#
# def compute_accuracy_score_llm_judge(
#     data_source: str,
#     solution_str: str,
#     ground_truth: str,
#     extra_info: dict = None,
# ) -> float:
#     """Accuracy reward using LLM judge — requires deployed judge model."""
#     answer_text, _ = _extract_answer(solution_str)
#     if not answer_text:
#         return 0.0
#
#     question = ""
#     if extra_info and isinstance(extra_info, dict):
#         question = extra_info.get("question", "")
#
#     return _call_llm_judge(
#         question=question,
#         model_answer=answer_text,
#         ground_truth=ground_truth,
#     )
# ============================================================


def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip()).lower()


def _extract_answer(solution_str: str) -> tuple[str, bool]:
    cleaned = _SPECIAL_TOKEN_RE.sub("", solution_str)

    # 1. Content from <answer>...</answer> tags
    last_open = cleaned.rfind("<answer>")
    last_close = cleaned.rfind("</answer>")
    if last_open != -1 and last_close > last_open:
        return cleaned[last_open + len("<answer>") : last_close].strip(), True

    # 2. Content after last </think>
    if "</think>" in cleaned:
        after_think = cleaned.split("</think>")[-1].strip()
        after_think = re.sub(r"<tool_call>.*?</tool_call>", "", after_think, flags=re.DOTALL)
        after_think = re.sub(r"<tool_response>.*?</tool_response>", "", after_think, flags=re.DOTALL)
        after_think = re.sub(r"\bassistant\b", "", after_think).strip()
        if after_think:
            return after_think, False

    # 3. Last complete assistant turn
    last_assist = solution_str.rfind("<|im_start|>assistant")
    if last_assist != -1:
        turn_text = solution_str[last_assist:]
        eos_pos = turn_text.find("<|im_end|>")
        if eos_pos != -1:
            turn_text = turn_text[:eos_pos]
        turn_text = _SPECIAL_TOKEN_RE.sub("", turn_text)
        turn_text = re.sub(r"<think>.*?</think>", "", turn_text, flags=re.DOTALL)
        turn_text = re.sub(r"<tool_call>.*?</tool_call>", "", turn_text, flags=re.DOTALL)
        turn_text = re.sub(r"<tool_response>.*?</tool_response>", "", turn_text, flags=re.DOTALL)
        turn_text = re.sub(r"\bassistant\b", "", turn_text).strip()
        if turn_text:
            return turn_text, False

    return cleaned.strip(), False


def _try_numeric_match(answer: str, gt: str, tolerance: float = 0.05) -> bool:
    """Check if answer and ground truth are numerically close within tolerance."""
    num_re = re.compile(r"[-+]?\d[\d,]*\.?\d*")
    a_nums = num_re.findall(answer.replace(",", ""))
    g_nums = num_re.findall(gt.replace(",", ""))
    if not a_nums or not g_nums:
        return False
    try:
        a_val = float(a_nums[-1])
        g_val = float(g_nums[-1])
        if g_val == 0:
            return a_val == 0
        return abs(a_val - g_val) / abs(g_val) <= tolerance
    except (ValueError, ZeroDivisionError):
        return False


_MAX_ANSWER_WORDS = 100


def compute_accuracy_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> float:
    answer_text, _ = _extract_answer(solution_str)
    if not answer_text:
        return 0.0

    if len(answer_text.split()) > _MAX_ANSWER_WORDS:
        return 0.0

    norm_answer = _normalize(answer_text)
    norm_gt = _normalize(ground_truth)

    if norm_gt and norm_gt in norm_answer:
        return 1.0
    elif norm_answer and norm_answer in norm_gt and len(norm_answer) > 0:
        return 0.5
    elif _try_numeric_match(norm_answer, norm_gt, tolerance=0.05):
        return 1.0
    else:
        return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    """Combined reward: accuracy (0.8) + format (0.1) + tool_call (0.1)."""
    acc = compute_accuracy_score(data_source, solution_str, ground_truth, extra_info)
    fmt = check_qwen3_coder_format(solution_str)
    tool = check_tool_call_reward(solution_str)
    score = ACCURACY_REWARD_WEIGHT * acc + FORMAT_REWARD_WEIGHT * fmt + TOOL_CALL_REWARD_WEIGHT * tool
    return {"score": score, "acc": acc, "format_reward": fmt, "tool_call_reward": tool}


_SPECIAL_TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_QWEN_FUNCTION_RE = re.compile(r"<function=\w+>.*?</function>", re.DOTALL)


def check_qwen3_coder_format(response_str: str) -> float:
    response_str = _SPECIAL_TOKEN_RE.sub("", response_str)

    last_answer_end = response_str.rfind("</answer>")
    if last_answer_end != -1:
        response_str = response_str[: last_answer_end + len("</answer>")]

    depth = 1  # prompt's <think> not in response
    for m in re.finditer(r"</?think>", response_str):
        if m.group() == "<think>":
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            return 0.0
    if depth != 0:
        return 0.0

    if not re.search(r"<answer>.*?</answer>", response_str, re.DOTALL):
        return 0.0

    for match in _TOOL_CALL_BLOCK_RE.finditer(response_str):
        block = match.group(1).strip()
        if not block:
            return 0.0
        # --- Hermes JSON format (commented out — enable if using hermes tool parser) ---
        # if block.startswith("{"):
        #     try:
        #         info = json.loads(block)
        #         if not isinstance(info.get("name"), str):
        #             return 0.0
        #         if not isinstance(info.get("arguments"), dict):
        #             return 0.0
        #     except json.JSONDecodeError:
        #         return 0.0
        # elif block.startswith("<function="):
        #     if not _QWEN_FUNCTION_RE.search(block):
        #         return 0.0
        # else:
        #     return 0.0
        # --- qwen3_coder XML format only ---
        if not _QWEN_FUNCTION_RE.search(block):
            return 0.0

    return 1.0


def check_tool_call_reward(response_str: str) -> float:
    """Reward for correct tool usage: any valid tool call."""
    tool_blocks = _TOOL_CALL_BLOCK_RE.findall(response_str)
    if not tool_blocks:
        return 0.0

    for block in tool_blocks:
        block = block.strip()
        if not block:
            continue
        if _QWEN_FUNCTION_RE.search(block):
            return 1.0

    return 0.0


class ToolRewardManager:
    """Reward manager for multi-turn agent training."""

    def __init__(
        self,
        tokenizer,
        num_examine: int = 2,
        compute_score=None,
        reward_fn_key: str = "data_source",
        accuracy_reward_weight: float = ACCURACY_REWARD_WEIGHT,
        format_reward_weight: float = FORMAT_REWARD_WEIGHT,
        tool_call_reward_weight: float = TOOL_CALL_REWARD_WEIGHT,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or compute_accuracy_score
        self.reward_fn_key = reward_fn_key
        self.accuracy_reward_weight = accuracy_reward_weight
        self.format_reward_weight = format_reward_weight
        self.tool_call_reward_weight = tool_call_reward_weight

    def __call__(self, data: DataProto, return_dict=False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        (
            prompt_strs,
            response_strs,
            ground_truths,
            data_sources,
            extra_infos,
            valid_resp_lens,
        ) = self._decode_batch(data)
        printed = {}

        for i in range(len(data)):
            accuracy = self.compute_score(
                data_source=data_sources[i],
                solution_str=response_strs[i],
                ground_truth=ground_truths[i],
                extra_info=extra_infos[i],
            )
            if isinstance(accuracy, dict):
                accuracy = accuracy.get("score", 0.0)

            format_reward = check_qwen3_coder_format(response_strs[i])
            tool_call_reward = check_tool_call_reward(response_strs[i])

            weighted_accuracy = accuracy * self.accuracy_reward_weight
            weighted_format = format_reward * self.format_reward_weight
            weighted_tool_call = tool_call_reward * self.tool_call_reward_weight
            combined = weighted_accuracy + weighted_format + weighted_tool_call

            reward_extra_info["accuracy_reward"].append(weighted_accuracy)
            reward_extra_info["tool_format_reward"].append(weighted_format)
            reward_extra_info["tool_call_reward"].append(weighted_tool_call)
            reward_extra_info["combined_reward"].append(combined)

            reward_tensor[i, valid_resp_lens[i] - 1] = combined

            ds = data_sources[i]
            if ds not in printed:
                printed[ds] = 0
            if printed[ds] < self.num_examine:
                printed[ds] += 1
                print(f"[response] {response_strs[i][:200]}")
                print(f"[ground_truth] {ground_truths[i]}")
                print(
                    f"[accuracy] {weighted_accuracy} [format] {weighted_format} [tool_call] {weighted_tool_call} [combined] {combined}"
                )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor

    def _decode_batch(self, data: DataProto):
        prompt_strs = []
        response_strs = []
        ground_truths = []
        data_sources = []
        extra_infos = []
        valid_response_lengths = []

        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_len = item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = item.batch["responses"]
            valid_response_len = item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_len]

            prompt_strs.append(self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True))
            response_strs.append(self.tokenizer.decode(valid_response_ids, skip_special_tokens=True))
            ground_truths.append(item.non_tensor_batch.get("reward_model", {}).get("ground_truth", ""))
            data_sources.append(item.non_tensor_batch.get(self.reward_fn_key, ""))
            extra_infos.append(item.non_tensor_batch.get("extra_info", None))
            valid_response_lengths.append(valid_response_len)

        return (
            prompt_strs,
            response_strs,
            ground_truths,
            data_sources,
            extra_infos,
            valid_response_lengths,
        )
