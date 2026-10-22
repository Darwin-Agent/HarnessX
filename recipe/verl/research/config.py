# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
"""Research RL task configuration for verl backend."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass
class ResearchConfig:
    max_steps: int = 12
    sample_timeout: int = 600
    max_new_tokens_per_turn: int = 1024
    early_commit_ratio: float = 0.95

    browser_pool_size: int = 4
    browser_page_timeout: int = 5
    browser_click_timeout: int = 5
    browser_retries: int = 0  # no retry, no fallback for now

    webfetch_timeout: int = 5
    webfetch_browser_timeout: int = 5
    webfetch_download_timeout: int = 10
    webfetch_max_download_size_mb: int = 10
    webfetch_retries: int = 0  # no retry, no fallback for now
    webfetch_concurrency: int = 4
    download_dir: str = field(
        default_factory=lambda: os.environ.get(
            "HARNESSX_DOWNLOAD_DIR", str(Path.home() / ".cache" / "harnessx" / "downloads")
        )
    )

    websearch_concurrency: int = 10
    websearch_timeout: int = 3
    websearch_ddg_timeout: int = 3

    bash_timeout: int = 5
    work_dir: str = field(
        default_factory=lambda: os.environ.get(
            "HARNESSX_WORK_DIR", str(Path.home() / ".local" / "share" / "harnessx" / "bash_work")
        )
    )

    code_timeout: int = 5
    code_concurrency: int = 4
    code_max_output_chars: int = 2000
    code_work_dir: str = field(
        default_factory=lambda: os.environ.get(
            "HARNESSX_CODE_WORK_DIR", str(Path.home() / ".local" / "share" / "harnessx" / "code_work")
        )
    )

    max_content_chars: int = 3000
    tool_call_format: str = "qwen3_coder"
    tool_result_role: str = "user"

    matpo_enabled: bool = True

    skills_dir: str = ""

    def __post_init__(self) -> None:
        if not self.skills_dir:
            self.skills_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "harnessx" / "skills")

    def to_env_dict(self) -> dict[str, str]:
        return {
            "HARNESSX_SAMPLE_TIMEOUT": str(self.sample_timeout),
            "HARNESSX_MAX_NEW_TOKENS_PER_TURN": str(self.max_new_tokens_per_turn),
            "HARNESSX_EARLY_COMMIT_RATIO": str(self.early_commit_ratio),
            "HARNESSX_BROWSER_POOL_SIZE": str(self.browser_pool_size),
            "HARNESSX_BROWSER_PAGE_TIMEOUT": str(self.browser_page_timeout),
            "HARNESSX_BROWSER_CLICK_TIMEOUT": str(self.browser_click_timeout),
            "HARNESSX_WEBFETCH_TIMEOUT": str(self.webfetch_timeout),
            "HARNESSX_WEBFETCH_BROWSER_TIMEOUT": str(self.webfetch_browser_timeout),
            "HARNESSX_WEBFETCH_DOWNLOAD_TIMEOUT": str(self.webfetch_download_timeout),
            "HARNESSX_WEBFETCH_CONCURRENCY": str(self.webfetch_concurrency),
            "HARNESSX_DOWNLOAD_DIR": self.download_dir,
            "HARNESSX_WEBSEARCH_CONCURRENCY": str(self.websearch_concurrency),
            "HARNESSX_WEBSEARCH_TIMEOUT": str(self.websearch_timeout),
            "HARNESSX_WEBSEARCH_DDG_TIMEOUT": str(self.websearch_ddg_timeout),
            "HARNESSX_BASH_TIMEOUT": str(self.bash_timeout),
            "HARNESSX_WORK_DIR": self.work_dir,
            "HARNESSX_CODE_TIMEOUT": str(self.code_timeout),
            "HARNESSX_CODE_CONCURRENCY": str(self.code_concurrency),
            "HARNESSX_CODE_WORK_DIR": self.code_work_dir,
            "HARNESSX_MAX_CONTENT_CHARS": str(self.max_content_chars),
            "HARNESSX_TASK_TYPE": "research",
        }


cfg = ResearchConfig()
