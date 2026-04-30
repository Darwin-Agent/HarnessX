"""GAIATask — wraps a GAIA benchmark instance as an HarnessX task.

GAIA (General AI Assistants) tests multi-step reasoning + web retrieval.
Each question has a single unambiguous final answer.

Dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA
Paper:   https://arxiv.org/abs/2311.12983

Supports two loaders:
- ``load_gaia_tasks``: downloads/reads the HF dataset
  (requires ``pip install datasets huggingface_hub``).
- ``load_gaia_tasks_from_json``: reads a local JSON file
  (e.g. ``recipe/gaia_evolver/data/webthinker_gaia_dev.json``) with the
  webthinker schema (``Question`` / ``answer`` / ``Level`` /
  ``Annotator_Metadata``). No HF download.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from harnessx.core.harness import BaseTask

from .defaults import GPT5_STEPS_PER_LEVEL

logger = logging.getLogger(__name__)

# Enhanced success criteria that strongly guides the agent toward
# the exact-match answer format GAIA expects.
_GAIA_SUCCESS_CRITERIA = (
    "MANDATORY STEPS before answering:\n"
    "1. ALWAYS use web search to verify facts — do NOT answer from memory alone.\n"
    "2. Search at least twice with different queries to cross-check.\n"
    "3. Read the question carefully: pay attention to units, date ranges, and qualifiers.\n"
    "4. Use code_interpreter for ANY computation, counting, or data processing.\n"
    "\n"
    "Answer format — put your answer on a NEW line:\n"
    "   FINAL ANSWER: <your concise answer>\n"
    "\n"
    "Answer rules:\n"
    "- Be maximally concise — only the essential information.\n"
    "- Numbers: plain digits (e.g., '828' not 'eight hundred twenty-eight').\n"
    "- If the question says 'how many thousand', answer in thousands (e.g., '17' not '17000').\n"
    "- Names: most common form (e.g., 'Sam Altman' not 'Samuel H. Altman').\n"
    "- Do NOT include units unless specifically asked.\n"
    "- Do NOT include explanations in the FINAL ANSWER line.\n"
    "- Lists: separate with commas, no 'and' (e.g., 'Alice, Bob, Charlie').\n"
    "\n"
    "Search failures:\n"
    "- If search fails, try a different query. If all searches fail, answer from knowledge.\n"
    "- Always provide a FINAL ANSWER even if uncertain — a best guess beats no answer.\n"
)


@dataclass
class GAIATask(BaseTask):
    """Single GAIA benchmark question.

    Fields mirror the HuggingFace dataset columns:
    - task_id: unique identifier
    - question: the question text (also stored in self.description)
    - level: 1 / 2 / 3 (difficulty)
    - final_answer: ground-truth answer string
    - file_name: optional attachment filename
    - file_path: path to the attachment on disk (resolved after download)
    - annotator_metadata: original annotator notes (dict)
    """

    task_id: str = ""
    question: str = ""
    level: int = 1
    final_answer: str = ""
    file_name: str = ""
    file_path: str = ""
    annotator_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Allow constructing with just question= (description defaults to "")
        if self.question and not self.description:
            self.description = self.question
        elif self.description and not self.question:
            self.question = self.description
        if not self.success_criteria:
            self.success_criteria = _GAIA_SUCCESS_CRITERIA

    @classmethod
    def from_hf_row(cls, row: dict, data_dir: str = "") -> "GAIATask":
        """Create a GAIATask from a HuggingFace dataset row."""
        file_path = ""
        if row.get("file_path") and data_dir:
            file_path = os.path.join(data_dir, row["file_path"])

        question = row.get("Question", "")

        # Build description — multimodal for images, text-only otherwise
        description: str | list = question
        if file_path and os.path.exists(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                description = _build_vision_description(question, file_path, ext)
            else:
                description = question + f"\n\n[Attached file: {row.get('file_name', 'unknown')} at {file_path}]"

        lvl = int(row.get("Level", 1))
        from benchmarks.gaia.defaults import GPT5_STEPS_PER_LEVEL

        return cls(
            task_id=row.get("task_id", ""),
            question=question,
            level=lvl,
            final_answer=row.get("Final answer", ""),
            file_name=row.get("file_name", ""),
            file_path=file_path,
            annotator_metadata=row.get("Annotator Metadata", {}),
            description=description,
            max_steps=GPT5_STEPS_PER_LEVEL.get(lvl, 30),
        )


_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _build_vision_description(question: str, file_path: str, ext: str) -> list:
    """Build a multimodal content list with the image embedded as base64.

    This sends the image directly to GPT-5's vision capability, which is
    far more accurate than OCR for chess positions, colored numbers, music
    notation, geometric shapes, etc.
    """
    media_type = _IMAGE_MEDIA_TYPES.get(ext, "image/png")
    try:
        with open(file_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("ascii")
        return [
            {"type": "text", "text": question + f"\n\n[The image is shown below. Also available at: {file_path}]"},
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}},
        ]
    except Exception as e:
        logger.warning("Failed to read image %s: %s — falling back to path hint", file_path, e)
        return question + f"\n\n[Attached file: {os.path.basename(file_path)} at {file_path}]"


def _find_gaia_cache() -> str | None:
    """Find GAIA dataset in local HuggingFace cache.

    Prefers caches that contain actual data files (validation directory),
    not just metadata.
    """
    candidates = [
        "/ls/data/lushuo/.cache_symlinks/huggingface/hub/datasets--gaia-benchmark--GAIA/snapshots",
        os.path.expanduser("~/.cache/huggingface/hub/datasets--gaia-benchmark--GAIA/snapshots"),
    ]
    best = None
    for base in candidates:
        if not os.path.isdir(base):
            continue
        snaps = sorted(os.listdir(base))
        if not snaps:
            continue
        snap_dir = os.path.join(base, snaps[-1])
        if os.path.isdir(os.path.join(snap_dir, "2023", "validation")):
            return snap_dir
        if best is None:
            best = snap_dir
    return best


def load_gaia_tasks(
    level: int | None = None,
    split: str = "validation",
    year: str = "2023",
    max_tasks: int | None = None,
) -> list[GAIATask]:
    """Load GAIA tasks from HuggingFace (or local cache).

    Args:
        level: 1, 2, or 3. None = all levels.
        split: "validation" (has answers) or "test" (answers hidden).
        year: "2023" (default).
        max_tasks: cap the number of tasks returned.

    Returns:
        List of GAIATask instances with ground-truth answers (validation split).
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("GAIA requires: pip install datasets huggingface_hub") from e

    # Try snapshot_download first; fall back to local cache if gated/auth fails
    data_dir = None
    try:
        data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")
    except Exception as exc:
        logger.warning("snapshot_download failed (%s), trying local cache...", exc)
        data_dir = _find_gaia_cache()
        if data_dir:
            logger.info("Using cached GAIA data at %s", data_dir)
        else:
            raise RuntimeError(
                "Cannot access GAIA dataset: HuggingFace auth required and no local cache found. "
                "Run 'huggingface-cli login' first."
            ) from exc

    tasks: list[GAIATask] = []
    levels = [level] if level else [1, 2, 3]

    for lvl in levels:
        config_name = f"{year}_level{lvl}"
        try:
            ds = load_dataset(data_dir, config_name, split=split)
        except Exception as exc:
            logger.warning("Failed to load GAIA %s/%s: %s", config_name, split, exc)
            continue

        for row in ds:
            tasks.append(GAIATask.from_hf_row(dict(row), data_dir=data_dir))

    if max_tasks:
        tasks = tasks[:max_tasks]

    logger.info("Loaded %d GAIA tasks (level=%s, split=%s)", len(tasks), level, split)
    return tasks


def load_webthinker_gaia_tasks(
    path: str,
    level: int | None = None,
    max_tasks: int | None = None,
) -> list[GAIATask]:
    """Load GAIA tasks from WebThinker's dev.json format (text-only, 103 questions).

    WebThinker's subset differs from the HuggingFace format:
    - Uses "Question", "answer" (lowercase), "Level" (int), "task_id"
    - Text-only: no file attachments

    Args:
        path:      Path to dev.json (e.g., webthinker_gaia_dev.json)
        level:     Filter to specific level (1/2/3). None = all.
        max_tasks: Cap the number of tasks.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tasks: list[GAIATask] = []
    for row in data:
        lvl = int(row.get("Level", 1))
        if level is not None and lvl != level:
            continue
        question = row.get("Question", "")
        tasks.append(
            GAIATask(
                task_id=row.get("task_id", str(row.get("id", ""))),
                question=question,
                level=lvl,
                final_answer=row.get("answer", ""),
                description=question,
                max_steps=GPT5_STEPS_PER_LEVEL.get(lvl, 30),
            )
        )

    if max_tasks:
        tasks = tasks[:max_tasks]

    logger.info(
        "Loaded %d WebThinker GAIA tasks from %s (level=%s)",
        len(tasks),
        path,
        level,
    )
    return tasks


def load_gaia_tasks_from_json(
    path: str,
    level: int | None = None,
    max_tasks: int | None = None,
    attachments_dir: str | None = None,
) -> list[GAIATask]:
    """Load GAIA tasks from a local JSON file (no HuggingFace download).

    Expected schema (webthinker-style):
        [
          {
            "task_id": "...",
            "Question": "...",
            "answer": "...",
            "Level": 1,
            "Annotator_Metadata": {...}
          }
        ]

    Args:
        path: absolute or relative path to the JSON file.
        level: filter by difficulty (1/2/3). None = all.
        max_tasks: cap the number of tasks returned.
        attachments_dir: if set, look for ``<task_id>.*`` files inside this
            directory and attach them. JSON has no file metadata, so this
            is optional; most webthinker dumps are text-only.

    Returns:
        List of GAIATask instances.
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    tasks: list[GAIATask] = []
    for raw in rows:
        lvl = int(raw.get("Level", 1))
        if level is not None and lvl != level:
            continue

        _file_path = ""
        file_name = ""
        if attachments_dir:
            tid = raw.get("task_id", "")
            if tid:
                try:
                    matches = [n for n in os.listdir(attachments_dir) if n.startswith(tid + ".")]
                except OSError:
                    matches = []
                if matches:
                    file_name = matches[0]
                    _file_path = os.path.join(attachments_dir, file_name)

        # Translate webthinker schema to the HF-style keys expected by
        # ``GAIATask.from_hf_row``.
        row = {
            "task_id": raw.get("task_id", ""),
            "Question": raw.get("Question", ""),
            "Level": lvl,
            "Final answer": raw.get("answer", ""),
            "file_name": file_name,
            "file_path": file_name,  # from_hf_row joins with data_dir
            "Annotator Metadata": raw.get("Annotator_Metadata", {}),
        }
        tasks.append(GAIATask.from_hf_row(row, data_dir=attachments_dir or ""))

    if max_tasks:
        tasks = tasks[:max_tasks]

    logger.info(
        "Loaded %d GAIA tasks from %s (level=%s, attachments=%s)",
        len(tasks),
        path,
        level,
        bool(attachments_dir),
    )
    return tasks
