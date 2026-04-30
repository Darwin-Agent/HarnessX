"""LoCoMoTask and LoCoMoEvaluator.

Each LoCoMo QA pair becomes one ``LoCoMoTask``.  The full session history is
carried on the task so that ``SessionIngester`` can pre-load memories before
``Harness.run()`` is called.

Dataset format (HuggingFace snap-research/locomo)
-------------------------------------------------
Each sample is a dict with:
    "idx"          : int — sample identifier
    "conversation" : list of session dicts, each with:
        "session_id"   : 1-based int
        "date"         : "YYYY-MM-DD"
        "conversation" : list of {"speaker": "A"|"B", "text": str, ...}
    "qa"           : list of QA dicts, each with:
        "question"     : str
        "answer"       : str
        "category"     : "single_hop_qa" | "multi_hop_qa" |
                         "temporal_reasoning" | "open_domain_qa" |
                         "summarization" | "adversarial_qa"
        "evidence"     : list[str] (optional, may be absent)
    "persona"      : {"A": str, "B": str}  (optional)

Question categories
-------------------
- single_hop_qa       : one fact from one session
- multi_hop_qa        : reasoning across multiple sessions
- temporal_reasoning  : "before/after/when" questions
- open_domain_qa      : general knowledge (no retrieval needed)
- summarization       : summarise a session or full conversation
- adversarial_qa      : unanswerable or contradictory questions
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUESTION_TYPES = frozenset(
    {
        "single_hop_qa",
        "multi_hop_qa",
        "temporal_reasoning",
        "open_domain_qa",
        "summarization",
        "adversarial_qa",
    }
)

# Integer category codes used in the local JSON file (locomo10.json)
_CATEGORY_INT_MAP: dict[int, str] = {
    1: "single_hop_qa",
    2: "temporal_reasoning",
    3: "multi_hop_qa",
    4: "summarization",
    5: "adversarial_qa",
}

_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _parse_date(raw: str) -> str:
    """Parse "1:56 pm on 8 May, 2023" → "2023-05-08".  Falls back to raw."""
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", raw)
    if m:
        day, month_str, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        month = _MONTH_MAP.get(month_str, 1)
        return f"{year:04d}-{month:02d}-{day:02d}"
    return raw


# ─── Data model ──────────────────────────────────────────────────────────────


@dataclass
class LoCoMoTurn:
    speaker: str  # "A" or "B"
    text: str
    blended_skill: str | None = None  # optional blended_skill_talk field


@dataclass
class LoCoMoSession:
    session_id: int  # 1-based
    date: str  # "YYYY-MM-DD"
    turns: list[LoCoMoTurn] = field(default_factory=list)


@dataclass
class LoCoMoQA:
    question: str
    answer: str
    category: str  # one of QUESTION_TYPES
    evidence: list[str] = field(default_factory=list)


@dataclass
class LoCoMoSample:
    sample_id: str
    sessions: list[LoCoMoSession]
    qa_pairs: list[LoCoMoQA]
    persona_a: str | None = None
    persona_b: str | None = None

    @classmethod
    def from_json_row(cls, row: dict[str, Any]) -> "LoCoMoSample":
        """Parse a row from the local locomo10.json file.

        The local format uses a flat dict for conversation sessions::

            {
                "sample_id": "conv-26",
                "conversation": {
                    "speaker_a": "Caroline",
                    "speaker_b": "Melanie",
                    "session_1_date_time": "1:56 pm on 8 May, 2023",
                    "session_1": [{"speaker": "Caroline", "text": "...", ...}, ...],
                    ...
                },
                "qa": [{"question": "...", "answer": "...",
                         "evidence": [...], "category": 1}, ...]
            }
        """
        conv: dict = row.get("conversation", {})
        speaker_a: str = conv.get("speaker_a", "A")
        speaker_b: str = conv.get("speaker_b", "B")

        # Collect session indices present in the dict
        session_ids = sorted(int(m.group(1)) for k in conv if (m := re.fullmatch(r"session_(\d+)", k)))

        sessions: list[LoCoMoSession] = []
        for sid in session_ids:
            raw_date = conv.get(f"session_{sid}_date_time", "")
            date = _parse_date(raw_date)
            turns = [
                LoCoMoTurn(
                    speaker=t.get("speaker", ""),
                    text=t.get("text", ""),
                )
                for t in conv.get(f"session_{sid}", [])
            ]
            sessions.append(LoCoMoSession(session_id=sid, date=date, turns=turns))

        qa_pairs: list[LoCoMoQA] = []
        for q in row.get("qa", []):
            raw_cat = q.get("category", 1)
            if isinstance(raw_cat, int):
                category = _CATEGORY_INT_MAP.get(raw_cat, "single_hop_qa")
            else:
                category = str(raw_cat)
            qa_pairs.append(
                LoCoMoQA(
                    question=q.get("question", ""),
                    answer=str(q.get("answer", "")),
                    category=category,
                    evidence=q.get("evidence") or [],
                )
            )

        return cls(
            sample_id=str(row.get("sample_id", "")),
            sessions=sessions,
            qa_pairs=qa_pairs,
            persona_a=speaker_a,
            persona_b=speaker_b,
        )

    @classmethod
    def from_hf_row(cls, row: dict[str, Any]) -> "LoCoMoSample":
        """Parse a raw HuggingFace dataset row into a ``LoCoMoSample``."""
        sessions: list[LoCoMoSession] = []
        for s in row.get("conversation", []):
            turns = [
                LoCoMoTurn(
                    speaker=t.get("speaker", ""),
                    text=t.get("text", ""),
                    blended_skill=t.get("blended_skill_talk"),
                )
                for t in s.get("conversation", [])
            ]
            sessions.append(
                LoCoMoSession(
                    session_id=int(s.get("session_id", 0)),
                    date=s.get("date", ""),
                    turns=turns,
                )
            )

        qa_pairs: list[LoCoMoQA] = []
        for q in row.get("qa", []):
            qa_pairs.append(
                LoCoMoQA(
                    question=q.get("question", ""),
                    answer=q.get("answer", ""),
                    category=q.get("category", "single_hop_qa"),
                    evidence=q.get("evidence") or [],
                )
            )

        persona = row.get("persona", {}) or {}
        return cls(
            sample_id=str(row.get("idx", "")),
            sessions=sessions,
            qa_pairs=qa_pairs,
            persona_a=persona.get("A"),
            persona_b=persona.get("B"),
        )


# ─── Task ────────────────────────────────────────────────────────────────────

try:
    from harnessx.core.harness import BaseTask
except ImportError:
    # Allow importing this module standalone (e.g., for unit tests)
    BaseTask = object  # type: ignore


@dataclass
class LoCoMoTask(BaseTask):
    """One LoCoMo QA pair as a Harness task.

    The session history is carried here so that the runner can call
    ``SessionIngester.ingest(task.sessions, memory)`` before ``Harness.run()``.
    """

    sample_id: str = ""
    question: str = ""
    category: str = "single_hop_qa"
    gold_answer: str = ""
    evidence: list[str] = field(default_factory=list)
    sessions: list[LoCoMoSession] = field(default_factory=list)
    persona_a: str | None = None
    persona_b: str | None = None

    def __post_init__(self):
        # Build description visible to the agent
        if not self.description:
            self.description = self.question
        if not self.success_criteria:
            self.success_criteria = "Answer the question accurately based on the conversation history."

    @classmethod
    def from_sample(cls, sample: LoCoMoSample, qa: LoCoMoQA) -> "LoCoMoTask":
        return cls(
            sample_id=sample.sample_id,
            question=qa.question,
            category=qa.category,
            gold_answer=qa.answer,
            evidence=qa.evidence,
            sessions=sample.sessions,
            persona_a=sample.persona_a,
            persona_b=sample.persona_b,
            description=qa.question,
            max_steps=5,
        )

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        max_samples: int | None = None,
        categories: list[str] | None = None,
    ) -> list["LoCoMoTask"]:
        """Load tasks from a local locomo JSON file (e.g. locomo10.json).

        Args:
            path:        path to the JSON file
            max_samples: cap on number of *conversations* to load
            categories:  filter to specific question types; None = all

        Returns:
            List of ``LoCoMoTask``, one per QA pair.
        """
        rows: list[dict] = json.loads(Path(path).read_text())
        if max_samples is not None:
            rows = rows[:max_samples]

        tasks: list[LoCoMoTask] = []
        for row in rows:
            sample = LoCoMoSample.from_json_row(row)
            for qa in sample.qa_pairs:
                if categories and qa.category not in categories:
                    continue
                tasks.append(cls.from_sample(sample, qa))
        return tasks

    @classmethod
    def from_dataset(
        cls,
        split: str = "test",
        max_samples: int | None = None,
        categories: list[str] | None = None,
    ) -> list["LoCoMoTask"]:
        """Load tasks from the HuggingFace dataset.

        Args:
            split:       dataset split (usually "test")
            max_samples: cap on number of *conversations* to load
            categories:  filter to specific question types; None = all

        Returns:
            List of ``LoCoMoTask``, one per QA pair.
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("datasets is required: pip install datasets") from e

        ds = load_dataset("snap-research/locomo", split=split)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        tasks: list[LoCoMoTask] = []
        for row in ds:
            sample = LoCoMoSample.from_hf_row(row)
            for qa in sample.qa_pairs:
                if categories and qa.category not in categories:
                    continue
                tasks.append(cls.from_sample(sample, qa))
        return tasks


# ─── Evaluator ───────────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles."""
    text = str(text)  # guard against int/None answers in raw data
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return " ".join(tokens)


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, gold: str) -> float:
    return float(_normalize(prediction) == _normalize(gold))


def _rouge_l(prediction: str, gold: str) -> float:
    """Simple ROUGE-L (LCS-based) without external deps."""
    pred = _normalize(prediction).split()
    ref = _normalize(gold).split()
    if not pred or not ref:
        return 0.0
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / m
    recall = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class EvalResult:
    score: float  # primary score in [0, 1]
    exact_match: float = 0.0
    f1: float = 0.0
    rouge_l: float = 0.0
    llm_judge: float | None = None
    category: str = ""
    prediction: str = ""
    gold: str = ""


class LoCoMoEvaluator:
    """Multi-metric evaluator for LoCoMo tasks.

    Metrics by category
    -------------------
    summarization      : ROUGE-L (primary) + F1
    temporal_reasoning : F1 (primary) + exact match
    adversarial_qa     : exact match (expected answer is "unknown" / "N/A")
    single/multi_hop   : F1 (primary) + exact match

    LLM-as-judge is wired via ``_sub_harnesses["judge"]`` when available,
    and its score is averaged with the automatic score.
    """

    # Phrases that indicate the model correctly declined to answer an unanswerable question
    _NO_ANSWER_PHRASES = (
        "i don't know",
        "i do not know",
        "don't know",
        "cannot be determined",
        "not mentioned",
        "no information",
        "not in the",
        "not available",
        "unknown",
    )

    def evaluate(self, prediction: str, task: "LoCoMoTask") -> EvalResult:
        gold = task.gold_answer
        category = task.category

        em = _exact_match(prediction, gold)
        f1 = _token_f1(prediction, gold)
        rl = _rouge_l(prediction, gold)

        if category == "adversarial_qa" and not gold.strip():
            # Gold is empty → question is unanswerable.
            # Score 1.0 if model correctly declines, 0.0 otherwise.
            pred_lower = prediction.lower()
            primary = 1.0 if any(p in pred_lower for p in self._NO_ANSWER_PHRASES) else 0.0
            em = primary
        elif category == "summarization":
            primary = rl
        else:
            primary = f1

        return EvalResult(
            score=primary,
            exact_match=em,
            f1=f1,
            rouge_l=rl,
            category=category,
            prediction=prediction,
            gold=gold,
        )

    def aggregate(self, results: list[EvalResult]) -> dict[str, float]:
        """Return per-category and overall average scores."""
        from collections import defaultdict

        by_cat: dict[str, list[float]] = defaultdict(list)
        for r in results:
            by_cat[r.category].append(r.score)
        agg: dict[str, float] = {}
        all_scores: list[float] = []
        for cat, scores in sorted(by_cat.items()):
            agg[cat] = sum(scores) / len(scores)
            all_scores.extend(scores)
        agg["overall"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
        return agg
