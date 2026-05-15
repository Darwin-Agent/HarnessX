"""GAIAEvaluator — exact-match + normalized comparison for GAIA answers.

GAIA answers are short, unambiguous strings. The official metric is
exact match after normalization. We also support numeric tolerance,
minor formatting differences, and LLM-as-judge fallback for cases
without ground-truth answers.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Iterable

from harnessx.core.events import EvalResult

if TYPE_CHECKING:
    from harnessx.core.events import Message, TaskEndEvent
    from harnessx.core.harness import BaseTask
    from harnessx.core.state import State

logger = logging.getLogger(__name__)


class GAIAEvaluator:
    """Evaluates GAIA task answers using exact match after normalization.

    Extraction priority:
    1. Lines starting with "FINAL ANSWER:" (case-insensitive)
    2. Last non-empty line of the last assistant message

    When no ground-truth answer is available, delegates to LLMJudgeEvaluator
    for semantic evaluation (requires a model_provider to be passed at init).
    """

    def __init__(self, judge_provider=None):
        """
        Args:
            judge_provider: Optional BaseModelProvider for LLM judge fallback.
                If None, cases without ground-truth answers score 0.
        """
        self._judge_provider = judge_provider

    async def evaluate(self, task: "BaseTask", state: "State") -> EvalResult:
        from .task import GAIATask

        assert isinstance(task, GAIATask)

        agent_answer = self._extract_answer(state)
        if not agent_answer:
            return EvalResult(
                passed=False,
                score=0.0,
                reason="no answer found in agent output",
                reward=0.0,
            )

        # If no ground-truth, try LLM judge fallback
        if not task.final_answer:
            return await self._judge_fallback(task, state, agent_answer)

        correct = task.final_answer.strip()
        passed = _answers_match(agent_answer, correct)
        score = 1.0 if passed else 0.0

        return EvalResult(
            passed=passed,
            score=score,
            reason=f"{'match' if passed else 'no_match'}: extracted='{agent_answer[:80]}'",
            reward=score,
        )

    async def _judge_fallback(self, task, state, agent_answer: str) -> EvalResult:
        """Use LLM judge when no ground-truth answer is available."""
        if self._judge_provider is None:
            # No judge available — mark as needs-review rather than failed
            return EvalResult(
                passed=False,
                score=0.0,
                reason=f"no ground-truth; agent answered: '{agent_answer[:100]}' (no judge configured)",
                reward=0.0,
            )
        try:
            from harnessx.core.events import Message

            prompt = (
                f"You are a factual accuracy judge. The agent was asked:\n{task.description[:500]}\n\n"
                f"The agent's final answer is:\n{agent_answer[:300]}\n\n"
                "Is this answer factually correct and complete? "
                "Respond with exactly one word: PASS or FAIL, followed by a brief reason.\n"
                "Format: PASS/FAIL: <reason>"
            )
            response = await self._judge_provider.complete(
                messages=[Message(role="user", content=prompt)],
                tools=[],
            )
            text = (response.content or "").strip()
            passed = text.upper().startswith("PASS")
            return EvalResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason=f"[LLM judge] {text[:200]} | agent_answer='{agent_answer[:80]}'",
                reward=1.0 if passed else 0.0,
            )
        except Exception as e:
            logger.warning("LLM judge fallback failed: %s", e)
            return EvalResult(
                passed=False,
                score=0.0,
                reason=f"judge fallback error: {e}; agent answered: '{agent_answer[:80]}'",
                reward=0.0,
            )

    def _extract_answer(self, state: "State") -> str:
        return _extract_answer_from_messages(state.messages)


def _extract_answer_from_messages(messages: Iterable["Message"]) -> str:
    """Extract final answer from a sequence of Messages.

    Searches all assistant messages (newest first) for 'FINAL ANSWER:' pattern.
    Falls back to last non-empty line of the last assistant message.
    """
    msg_list = list(messages)

    for msg in reversed(msg_list):
        if msg.role != "assistant" or not msg.content:
            continue
        match = re.search(
            r"(?:final\s+answer)\s*[:：]\s*(.+)",
            msg.content,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            answer_text = match.group(1).strip().split("\n")[0].strip()
            return _strip_markdown(answer_text)

    for msg in reversed(msg_list):
        if msg.role != "assistant" or not msg.content:
            continue
        match = re.search(
            r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:the\s+)?answer\s*(?:is)?\s*[:：]\s*(.+)",
            msg.content,
            re.IGNORECASE,
        )
        if match:
            answer_text = match.group(1).strip().split("\n")[0].strip()
            return _strip_markdown(answer_text)

    for msg in reversed(msg_list):
        if msg.role != "assistant" or not msg.content:
            continue
        lines = [line.strip() for line in msg.content.strip().splitlines() if line.strip()]
        if lines:
            return _strip_markdown(lines[-1])

    return ""


class GAIAPipelineEvaluator:
    """GAIA exact-match + LLM-judge evaluator with two call paths.

    Preferred (external path) — ``evaluate_answer(final_output, ground_truth)``:
        Call after ``harness.run()`` returns. Keeps evaluator outputs off
        the per-task trajectory ``.md`` files so the meta-agent reasons
        from pure behavioural signals.

    Legacy (in-processor path) — ``evaluate(event: TaskEndEvent)`` paired
    with ``set_ground_truth(answer)`` before each run:
        Used by callers that still wire ``EvaluationProcessor`` into the
        harness. Retained for non-GAIA recipes and backward compatibility.

    Both paths share the underlying exact-match / normalization / judge
    fallback logic — identical verdicts.
    """

    def __init__(self, judge_provider=None):
        self._ground_truth: str = ""
        self._judge_provider = judge_provider

    def set_ground_truth(self, answer: str) -> None:
        self._ground_truth = answer or ""

    async def evaluate(self, event: "TaskEndEvent") -> EvalResult:
        agent_answer = _extract_answer_from_messages(event.final_messages)
        if not agent_answer:
            return EvalResult(
                passed=False,
                score=0.0,
                reason="no answer found in agent output",
                reward=0.0,
            )

        if not self._ground_truth:
            return await self._judge_fallback(event, agent_answer)

        correct = self._ground_truth.strip()
        passed = _answers_match(agent_answer, correct)
        score = 1.0 if passed else 0.0
        return EvalResult(
            passed=passed,
            score=score,
            reason=f"{'match' if passed else 'no_match'}: extracted='{agent_answer[:80]}'",
            reward=score,
        )

    async def _judge_fallback(self, event: "TaskEndEvent", agent_answer: str) -> EvalResult:
        if self._judge_provider is None:
            return EvalResult(
                passed=False,
                score=0.0,
                reason=f"no ground-truth; agent answered: '{agent_answer[:100]}' (no judge configured)",
                reward=0.0,
            )
        try:
            from harnessx.core.events import Message

            task_desc = getattr(event, "task_description", "") or event.final_output[:200]
            prompt = (
                f"You are a factual accuracy judge. The agent was asked:\n{task_desc[:500]}\n\n"
                f"The agent's final answer is:\n{agent_answer[:300]}\n\n"
                "Is this answer factually correct and complete? "
                "Respond with exactly one word: PASS or FAIL, followed by a brief reason.\n"
                "Format: PASS/FAIL: <reason>"
            )
            response = await self._judge_provider.complete(
                messages=[Message(role="user", content=prompt)],
                tools=[],
            )
            text = (response.content or "").strip()
            passed = text.upper().startswith("PASS")
            return EvalResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason=f"[LLM judge] {text[:200]} | agent_answer='{agent_answer[:80]}'",
                reward=1.0 if passed else 0.0,
            )
        except Exception as e:
            logger.warning("LLM judge fallback failed: %s", e)
            return EvalResult(
                passed=False,
                score=0.0,
                reason=f"judge fallback error: {e}; agent answered: '{agent_answer[:80]}'",
                reward=0.0,
            )

    async def evaluate_answer(
        self,
        final_output: str,
        ground_truth: str,
    ) -> EvalResult:
        """Standalone answer evaluation — no TaskEndEvent dependency.

        For use by external runners that call the evaluator after
        ``harness.run()`` returns. Mirrors ``evaluate()``'s logic
        (extract → normalized match → judge fallback) but takes
        raw strings instead of an event.

        Kept for backward-compat + emergency fallback when no judge
        provider is configured. Prefer ``evaluate_with_trace_judge``
        when the full trajectory is accessible — it handles the case
        where ``final_output`` is empty but the trajectory contains a
        valid FINAL ANSWER on an earlier turn (e.g. post-commit-nudge).
        """
        from harnessx.core.events import Message

        # Wrap final_output in a minimal Message so we can reuse
        # the shared extractor.
        pseudo_messages = [Message(role="assistant", content=final_output or "")]
        agent_answer = _extract_answer_from_messages(pseudo_messages)

        if not agent_answer:
            return EvalResult(
                passed=False,
                score=0.0,
                reason="no answer found in agent output",
                reward=0.0,
            )

        gt = (ground_truth or "").strip()
        if not gt:
            return await self._judge_fallback_raw(agent_answer)

        passed = _answers_match(agent_answer, gt)
        score = 1.0 if passed else 0.0
        return EvalResult(
            passed=passed,
            score=score,
            reason=f"{'match' if passed else 'no_match'}: extracted='{agent_answer[:80]}'",
            reward=score,
        )

    async def evaluate_with_trace_judge(
        self,
        task_description: str,
        ground_truth: str,
        final_output: str,
        trajectory_messages: "Iterable[Message]",
        *,
        max_recent_assistant: int = 5,
        max_chars_per_msg: int = 1500,
    ) -> EvalResult:
        """LLM-judge-primary evaluation over the full trajectory.

        String-match over ``final_output`` was the source of a known class
        of false negatives: when the agent's last assistant message is a
        tool-call with empty content (or the FINAL ANSWER sentinel was
        emitted one or two turns earlier), ``final_output`` captures an
        empty string, the regex extractor gives up, and a task that was
        answered correctly gets graded ``passed=False``.

        This method instead sends the most recent assistant turns plus
        the task description and ground-truth answer to an LLM judge and
        asks whether the agent correctly arrived at the answer — with
        semantic equivalence (``$12,000`` ≡ ``12000``, "Quincy, MA" ≡
        "Quincy") and FINAL-ANSWER-on-earlier-turn both permissible.

        Falls back to ``evaluate_answer`` only when no judge provider is
        configured (e.g. unit tests without model access).
        """
        if self._judge_provider is None:
            return await self.evaluate_answer(final_output, ground_truth)

        from harnessx.core.events import Message

        msgs = list(trajectory_messages or [])
        # Pick the most recent assistant turns with non-empty content.
        assistant_turns = [
            m for m in msgs
            if getattr(m, "role", None) == "assistant" and getattr(m, "content", None)
        ]
        recent = assistant_turns[-max_recent_assistant:]
        # Latest first so the judge reads the commit-point first.
        recent_rendered = "\n\n---\n\n".join(
            (m.content or "")[:max_chars_per_msg]
            for m in reversed(recent)
        )
        if not recent_rendered and final_output:
            # Fallback: the trajectory didn't surface assistant content but
            # ``final_output`` has something — use it as the sole excerpt.
            recent_rendered = final_output[:max_chars_per_msg]

        gt = (ground_truth or "").strip()
        if not gt:
            # No ground truth — degrade to the legacy judge-fallback path.
            from harnessx.core.events import Message as _M
            return await self._judge_fallback_raw(
                recent_rendered or final_output or "",
            )

        if not recent_rendered:
            return EvalResult(
                passed=False,
                score=0.0,
                reason="no assistant content in trajectory (empty run?)",
                reward=0.0,
            )

        prompt = (
            "You are grading a tool-using agent's answer against a known "
            "ground-truth answer for a benchmark question.\n\n"
            f"QUESTION:\n{(task_description or '')[:1500]}\n\n"
            f"GROUND TRUTH:\n{gt[:500]}\n\n"
            "AGENT'S MOST RECENT ASSISTANT MESSAGES (most recent first, "
            "separated by ---):\n"
            f"{recent_rendered}\n\n"
            "Did the agent correctly commit to the ground-truth answer?\n"
            "Accept:\n"
            "- Semantic equivalence ('$12,000' ≡ '12000', 'Quincy, MA' ≡ 'Quincy',\n"
            "  date formats, singular/plural, trivial paraphrase).\n"
            "- Answer anywhere in the recent messages, not only on an explicit\n"
            "  `FINAL ANSWER:` line (the agent may have emitted it a turn or\n"
            "  two before the trajectory ended).\n"
            "Reject:\n"
            "- Partial correctness (e.g. only one of two required items).\n"
            "- A guess that happens to string-match without justification.\n"
            "- Silence / refusal / 'unable to determine' answers.\n\n"
            "Respond with exactly one token PASS or FAIL on the first line, "
            "followed by one short sentence of reasoning on the second line.\n"
            "Format:\n"
            "PASS\n"
            "<reason>\n"
            "OR:\n"
            "FAIL\n"
            "<reason>"
        )

        try:
            response = await self._judge_provider.complete(
                messages=[Message(role="user", content=prompt)],
                tools=[],
            )
            text = (response.content or "").strip()
            first_tok = text.split(None, 1)[0].upper() if text else ""
            passed = first_tok.startswith("PASS")
            return EvalResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason=f"[trace-judge] {text[:220]}",
                reward=1.0 if passed else 0.0,
            )
        except Exception as exc:
            logger.warning(
                "trace-judge LLM call failed (%s) — falling back to string match",
                exc,
            )
            return await self.evaluate_answer(final_output, ground_truth)

    async def _judge_fallback_raw(self, agent_answer: str) -> EvalResult:
        if self._judge_provider is None:
            return EvalResult(
                passed=False,
                score=0.0,
                reason=f"no ground-truth; agent answered: '{agent_answer[:100]}' (no judge configured)",
                reward=0.0,
            )
        try:
            from harnessx.core.events import Message

            prompt = (
                f"You are a factual accuracy judge. The agent's final answer is:\n"
                f"{agent_answer[:300]}\n\n"
                "Is this answer factually correct and complete? "
                "Respond with exactly one word: PASS or FAIL, followed by a brief reason.\n"
                "Format: PASS/FAIL: <reason>"
            )
            response = await self._judge_provider.complete(
                messages=[Message(role="user", content=prompt)],
                tools=[],
            )
            text = (response.content or "").strip()
            passed = text.upper().startswith("PASS")
            return EvalResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason=f"[LLM judge] {text[:200]} | agent_answer='{agent_answer[:80]}'",
                reward=1.0 if passed else 0.0,
            )
        except Exception as e:
            logger.warning("LLM judge fallback failed: %s", e)
            return EvalResult(
                passed=False,
                score=0.0,
                reason=f"judge fallback error: {e}; agent answered: '{agent_answer[:80]}'",
                reward=0.0,
            )


def _strip_markdown(text: str) -> str:
    """Remove common markdown formatting from text."""
    # Remove bold/italic markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    # Remove inline code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Remove links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
    "billion": "1000000000",
}


def _number_words_to_digits(text: str) -> str:
    """Convert written-out numbers to digits: 'five hundred' → '500'."""
    words = text.split()
    result: list[str] = []
    i = 0
    while i < len(words):
        w = words[i].lower().rstrip(".,;:")
        if w in _NUMBER_WORDS:
            num = int(_NUMBER_WORDS[w])
            total = 0
            current = num
            i += 1
            while i < len(words):
                nw = words[i].lower().rstrip(".,;:")
                if nw not in _NUMBER_WORDS:
                    break
                nval = int(_NUMBER_WORDS[nw])
                if nval >= 100:
                    current *= nval
                elif nval >= 10 and current < 10:
                    current += nval
                else:
                    if current >= 1000:
                        total += current
                        current = nval
                    else:
                        current += nval
                i += 1
            total += current
            result.append(str(total))
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)


def _normalize(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.strip().lower()
    # Strip markdown first
    text = _strip_markdown(text)
    # Remove common prefixes/suffixes
    text = re.sub(r"^(the answer is|answer:|the final answer is)\s*", "", text, flags=re.IGNORECASE)
    # Remove leading/trailing quotes (straight + smart)
    text = text.strip("\"'''`")
    # Remove trailing punctuation
    text = text.rstrip(".!;:")
    # Remove leading articles
    text = re.sub(r"^(the|a|an)\s+", "", text)
    # Treat hyphens and en-dashes as spaces (e.g., "Human-Oriented" == "Human Oriented")
    text = re.sub(r"[-–—]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove dollar signs, commas in numbers, percent signs
    text = text.replace("$", "").replace(",", "").replace("%", "")
    # Convert number words to digits ("five hundred" → "500")
    text = _number_words_to_digits(text)
    return text.strip()


def _answers_match(agent: str, correct: str) -> bool:
    """Check if agent answer matches correct answer.

    Matching strategies (in order):
    1. Exact match (after strip)
    2. Normalized match (lower, strip markdown/punct/whitespace/articles/%)
    3. Numeric tolerance (1% for integers, 3% for decimals; 1e-6 absolute floor)
    4. Fuzzy string matching (SequenceMatcher ratio >= 0.9)
    5. Containment check (all parts of correct answer found in agent answer)
    """
    # Exact match
    if agent.strip() == correct.strip():
        return True

    # Normalized match
    a_norm = _normalize(agent)
    c_norm = _normalize(correct)
    if a_norm == c_norm:
        return True

    # Numeric tolerance: 1% for integers, 3% for decimals
    try:
        a_val = float(re.sub(r"[^\d.\-]", "", agent))
        c_val = float(re.sub(r"[^\d.\-]", "", correct))
        both_int = a_val == int(a_val) and c_val == int(c_val)
        tol = 0.01 if both_int else 0.03
        if abs(a_val - c_val) < max(1e-6, abs(c_val) * tol):
            return True
    except (ValueError, TypeError):
        pass

    # Fuzzy string matching for non-trivial strings
    if len(c_norm) >= 5:
        import difflib

        ratio = difflib.SequenceMatcher(None, a_norm, c_norm).ratio()
        if ratio >= 0.9:
            return True

    # Multi-part containment: all comma-separated parts of correct answer in agent
    correct_parts = [p.strip() for p in correct.split(",") if p.strip()]
    if len(correct_parts) > 1:
        all_found = all(_normalize(part) in a_norm for part in correct_parts)
        if all_found:
            return True

    # For purely numeric answers, check if the number appears as a standalone word
    if re.fullmatch(r"[\d.,\s]+", c_norm):
        pattern = r"\b" + re.escape(c_norm) + r"\b"
        if re.search(pattern, a_norm):
            return True
        return False

    # Word-boundary containment (for short non-numeric answers, max 30 chars)
    if len(correct.strip()) <= 30:
        pattern = r"\b" + re.escape(c_norm) + r"\b"
        if re.search(pattern, a_norm):
            return True

    return False
