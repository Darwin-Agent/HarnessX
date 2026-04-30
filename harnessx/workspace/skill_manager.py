# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SkillEntry:
    name: str
    source: str  # how it was installed
    path: Path  # path to the directory containing SKILL.md
    is_symlink: bool  # whether the agent's skills/{name} is a symlink


class SkillManager:
    """
    Manages skill installation and agent-local symlinks.

    Args:
        ws_root:        Root workspace directory (e.g. ``.harnessx/pa``).
        agent_id:       Agent whose skills directory is managed.
        builtin_dir:    Path to the built-in skills dir.
                        Defaults to ``extensions/skills/`` at repo root.
    """

    def __init__(
        self,
        ws_root: Path,
        agent_id: str,
        builtin_dir: Path | None = None,
    ) -> None:
        self.ws_root = Path(ws_root)
        self.agent_id = agent_id
        self.agent_skills_dir = self.ws_root / agent_id / "skills"
        self.builtin_dir = builtin_dir or (Path(__file__).parents[2] / "extensions" / "skills")
        # Global skills dir used by the `skills` CLI for claude-code
        self._global_skills_dir = Path.home() / ".claude" / "skills"

    # ── Public API ────────────────────────────────────────────────────────────

    def list_installed(self) -> list[SkillEntry]:
        """Return skills currently active for this agent (from skills dir)."""
        if not self.agent_skills_dir.exists():
            return []
        entries: list[SkillEntry] = []
        for entry in sorted(self.agent_skills_dir.iterdir()):
            skill_md = entry / "SKILL.md"
            if not skill_md.exists():
                continue
            resolved = entry.resolve()
            entries.append(
                SkillEntry(
                    name=entry.name,
                    source=str(resolved),
                    path=resolved,
                    is_symlink=entry.is_symlink(),
                )
            )
        return entries

    def search(self, query: str) -> list[dict[str, str]]:
        """
        Search for installable skills.

        Searches:
          1. skills.sh registry (primary — `https://skills.sh/api/search?q=`)
          2. Built-in harnessx skills matching ``query``

        Returns a list of dicts with keys: name, description, source, type,
        installs (skills.sh results also include install counts).
        """
        results: list[dict[str, str]] = []

        # 1. skills.sh registry
        try:
            results.extend(_search_skills_sh(query))
        except Exception as exc:
            logger.debug("SkillManager: skills.sh search failed: %s", exc)

        # 2. Built-in skills (local fallback)
        if self.builtin_dir.exists():
            from .skill_index import _parse_frontmatter

            for skill_dir in sorted(self.builtin_dir.iterdir()):
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                fm = _parse_frontmatter(skill_md)
                name = fm.get("name", skill_dir.name)
                desc = fm.get("description", "")
                if query.lower() in name.lower() or query.lower() in desc.lower():
                    results.append(
                        {
                            "name": name,
                            "description": desc[:120],
                            "source": f"builtin:{name}",
                            "type": "builtin",
                        }
                    )

        return results

    def install(self, source: str, name: str = "") -> SkillEntry:
        """
        Install a skill from the given source and symlink it for this agent.

        ``source`` is auto-detected:
          - ``owner/repo@skill``   — install via `npx skills add` (skills.sh)
          - ``owner/repo``         — install all skills in repo via `npx skills add`
          - ``builtin:<name>``     — copy from harnessx built-in skills
          - ``https://github.com/...`` — git clone
          - ``https://.../SKILL.md``  — direct URL download
          - ``/absolute/path``         — local directory (symlinked directly)
        """
        source = source.strip()
        resolved_name, store_path = self._fetch(source, name)
        self._link(resolved_name, store_path)
        return SkillEntry(
            name=resolved_name,
            source=source,
            path=store_path,
            is_symlink=True,
        )

    def uninstall(self, name: str) -> None:
        """Remove the skill symlink from this agent (does not delete the store)."""
        link = self.agent_skills_dir / name
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                shutil.rmtree(str(link), ignore_errors=True)
        else:
            raise KeyError(f"Skill {name!r} not found in agent workspace")

    # ── Internal: fetch ───────────────────────────────────────────────────────

    def _fetch(self, source: str, hint_name: str) -> tuple[str, Path]:
        """Resolve and download a skill. Returns (name, path_with_SKILL_md)."""
        if source.startswith("builtin:"):
            return self._fetch_builtin(source[len("builtin:") :], hint_name)

        if source.startswith(("https://github.com/", "http://github.com/", "https://gitlab.com/", "git@")):
            return self._fetch_git(source, hint_name)

        if source.startswith(("https://", "http://")) and source.endswith(".md"):
            return self._fetch_url(source, hint_name)

        if source.startswith(("/", ".", "~")):
            return self._fetch_local(source, hint_name)

        # "owner/repo@skill" or "owner/repo" → skills.sh ecosystem
        if _is_repo_spec(source):
            return self._fetch_npx_skills(source, hint_name)

        # Last resort: try builtin name
        builtin_path = self.builtin_dir / source / "SKILL.md"
        if builtin_path.exists():
            return self._fetch_builtin(source, hint_name or source)

        raise ValueError(
            f"Cannot resolve skill source {source!r}. "
            "Use 'owner/repo@skill', 'builtin:name', a git URL, or a local path."
        )

    def _fetch_builtin(self, name: str, hint_name: str) -> tuple[str, Path]:
        skill_md = self.builtin_dir / name / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"Built-in skill {name!r} not found at {skill_md}")
        resolved = hint_name or name
        # Copy into agent_skills_dir directly (no symlink chain needed for builtins)
        dest = self.agent_skills_dir / resolved
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(skill_md), str(dest / "SKILL.md"))
        return resolved, dest

    def _fetch_npx_skills(self, spec: str, hint_name: str) -> tuple[str, Path]:
        """Install via ``npx skills add`` (global) then return the installed path.

        ``spec`` examples:
          - ``vercel-labs/agent-skills@react``   → one skill
          - ``vercel-labs/agent-skills``         → all skills in the repo
        """
        import subprocess

        # Derive the expected skill directory name after install
        if "@" in spec:
            skill_name = hint_name or spec.split("@")[-1]
        else:
            skill_name = hint_name or spec.split("/")[-1]

        cmd = [
            "npx",
            "--yes",
            "skills",
            "add",
            spec,
            "--agent",
            "claude-code",
            "--global",
            "--yes",
        ]
        logger.info("SkillManager: running %s", " ".join(cmd))
        subprocess.run(cmd, check=True, timeout=120)

        installed = self._global_skills_dir / skill_name
        if not installed.exists():
            raise FileNotFoundError(
                f"Skill {skill_name!r} not found at {installed} after install. Check the skill name in the repo."
            )
        return skill_name, installed

    def _fetch_git(self, repo_url: str, hint_name: str) -> tuple[str, Path]:
        """Clone a git repo, find SKILL.md, copy to agent skills dir."""
        import subprocess

        name = hint_name or repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, tmp],
                check=True,
                capture_output=True,
            )
            skill_md = _find_skill_md(Path(tmp))
            if skill_md is None:
                raise FileNotFoundError(f"No SKILL.md found in {repo_url}")
            dest = self.agent_skills_dir / name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(skill_md), str(dest / "SKILL.md"))
        return name, dest

    def _fetch_url(self, url: str, hint_name: str) -> tuple[str, Path]:
        """Download a raw SKILL.md URL."""
        name = hint_name or url.rstrip("/").split("/")[-2]
        dest = self.agent_skills_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        _download_file(url, dest / "SKILL.md")
        return name, dest

    def _fetch_local(self, path_str: str, hint_name: str) -> tuple[str, Path]:
        """Use a local directory directly (symlink into agent skills dir)."""
        local = Path(path_str).expanduser().resolve()
        if not (local / "SKILL.md").exists():
            raise FileNotFoundError(f"No SKILL.md in {local}")
        name = hint_name or local.name
        return name, local

    # ── Internal: link ────────────────────────────────────────────────────────

    def _link(self, name: str, store_path: Path) -> None:
        """Create/update symlink or skip if path is already inside agent_skills_dir."""
        self.agent_skills_dir.mkdir(parents=True, exist_ok=True)
        link = self.agent_skills_dir / name

        # If store_path IS the link target (builtin copy already placed there), skip
        if link.resolve() == store_path.resolve() and link.exists():
            return

        if link.is_symlink():
            link.unlink()
        elif link.exists():
            shutil.rmtree(str(link))

        # Prefer relative symlink so workspace is relocatable; fall back to absolute
        try:
            rel = Path(
                "../" * len(self.agent_skills_dir.parts) + "/".join(store_path.parts[1:])  # strip leading /
            )
            # Simpler: use os.path.relpath
            import os

            rel = Path(os.path.relpath(store_path, self.agent_skills_dir))
        except ValueError:
            rel = store_path

        link.symlink_to(rel)
        logger.info("SkillManager: linked %s → %s", link, rel)


# ── Helpers ───────────────────────────────────────────────────────────────────

_REPO_SPEC_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-][A-Za-z0-9_.-]*(@[A-Za-z0-9_.:/-]+)?$")


def _is_repo_spec(source: str) -> bool:
    """Return True if source looks like 'owner/repo' or 'owner/repo@skill'."""
    return bool(_REPO_SPEC_RE.match(source))


def _download_file(url: str, dest: Path) -> None:
    try:
        import httpx

        resp = httpx.get(url, follow_redirects=True, timeout=20)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except ImportError:
        import urllib.request

        urllib.request.urlretrieve(url, dest)


def _find_skill_md(root: Path) -> Path | None:
    """Find SKILL.md, preferring the root then searching recursively."""
    if (root / "SKILL.md").exists():
        return root / "SKILL.md"
    for p in root.rglob("SKILL.md"):
        return p
    return None


def _search_skills_sh(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search the skills.sh registry: https://skills.sh/api/search?q=<query>"""
    import json

    url = f"https://skills.sh/api/search?q={query}"
    try:
        import httpx

        resp = httpx.get(url, timeout=10, follow_redirects=True)
        data = resp.json()
    except ImportError:
        import urllib.request

        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())

    results = []
    for skill in data.get("skills", [])[:limit]:
        source = skill.get("source", "")  # e.g. "vercel-labs/agent-skills"
        skill_id = skill.get("skillId", "")  # e.g. "react"
        results.append(
            {
                "name": skill.get("name", skill_id),
                "description": skill.get("description", ""),
                "source": f"{source}@{skill_id}" if source else skill_id,
                "type": "skills.sh",
                "installs": skill.get("installs", 0),
                "repo": source,
            }
        )
    return results
