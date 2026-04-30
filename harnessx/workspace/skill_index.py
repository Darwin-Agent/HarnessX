# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SkillMeta:
    name: str
    description: str  # one-line trigger description from YAML frontmatter
    path: Path  # absolute path to SKILL.md


def _parse_frontmatter(skill_md: Path) -> dict[str, str]:
    """Parse YAML frontmatter from a SKILL.md file.

    Reads only up to the closing '---' — never loads the full markdown body.
    Returns dict with 'name', 'description', etc.
    """
    result: dict[str, str] = {}
    try:
        text = skill_md.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return result
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if ":" in line:
                key, _, val = line.partition(":")
                val = val.strip().strip('"').strip("'")
                result[key.strip()] = val
    except Exception:
        pass
    return result


def expand_skill_roots(roots: list[Path] | None) -> list[Path]:
    """Expand a list of parent dirs into individual skill dirs.

    For each *root* directory that exists, recursively finds every
    ``SKILL.md`` and returns the list of containing directories.
    The output is suitable for :class:`SkillIndex`'s ``extra_dirs``
    argument.

    Returns ``[]`` when *roots* is None/empty.  Missing roots are skipped
    silently (callers may opt-in optimistically).
    """
    if not roots:
        return []
    dirs: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for skill_md in sorted(root.rglob("SKILL.md")):
            dirs.append(skill_md.parent)
    return dirs


def collect_plugin_skill_dirs(home: Path | None) -> list[Path]:
    """Scan ``AGENT_HOME/plugins/*/skills/*/`` for plugin skill directories.

    Each returned path is a directory containing ``SKILL.md`` — ready to be
    passed as ``extra_dirs`` to :class:`SkillIndex`.

    Args:
        home: AGENT_HOME root (e.g. ``~/.harnessx``).  Returns ``[]`` when
              *None* or when the plugins directory does not exist.
    """
    if home is None:
        return []
    plugins_root = Path(home) / "plugins"
    if not plugins_root.is_dir():
        return []
    dirs: list[Path] = []
    for plugin_dir in sorted(plugins_root.iterdir()):
        if not plugin_dir.is_dir():
            continue
        skills_root = plugin_dir / "skills"
        if not skills_root.is_dir():
            continue
        for skill_dir in sorted(skills_root.iterdir()):
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                dirs.append(skill_dir)
    return dirs


class SkillIndex:
    """
    Lightweight index of available skills built from SKILL.md frontmatter.

    Parses only the YAML header of each SKILL.md — never loads the full body.
    Used by SkillRuntimePlugin.SkillSystemPromptProcessor to build the
    ``<available_skills>`` XML block injected into the system prompt on task
    start, and by ProgressiveSkillLoader to inject matching skill content
    per step based on query keyword overlap.

    Args:
        skills_dir:  Path to directory containing skill subdirectories.
                     Defaults to extensions/skills/ at repo root.
        extra_dirs:  Additional individual skill directories (each containing
                     ``SKILL.md``) to include.  Typically plugin skill dirs
                     from ``collect_plugin_skill_dirs()``.  Skills from
                     *skills_dir* take priority over *extra_dirs* when names
                     collide.
    """

    def __init__(
        self,
        skills_dir: Path | None = None,
        extra_dirs: list[Path] | None = None,
    ):
        self.skills_dir = skills_dir or (Path(__file__).parents[2] / "extensions" / "skills")
        self._extra_dirs = extra_dirs or []
        self._cache: list[SkillMeta] | None = None

    def list_skills(self) -> list[SkillMeta]:
        """Return all available skills sorted by name.

        Recursively scans for ``SKILL.md`` files under ``skills_dir``,
        so nested layouts like ``gaia/numeric-answer/SKILL.md`` are
        discovered alongside flat ones like ``docx/SKILL.md``.
        ``extra_dirs`` are appended.  Duplicate names are skipped
        (first-seen wins).
        """
        if self._cache is not None:
            return self._cache
        seen: set[str] = set()
        skills: list[SkillMeta] = []

        # Primary: recursively scan skills_dir
        if self.skills_dir.exists():
            for skill_md in sorted(self.skills_dir.rglob("SKILL.md")):
                fm = _parse_frontmatter(skill_md)
                name = fm.get("name")
                if not name or name in seen:
                    continue
                seen.add(name)
                skills.append(
                    SkillMeta(
                        name=name,
                        description=fm.get("description", ""),
                        path=skill_md,
                    )
                )

        # Extra: individual skill directories (e.g. from plugins)
        for skill_dir in self._extra_dirs:
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            fm = _parse_frontmatter(skill_md)
            name = fm.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            skills.append(
                SkillMeta(
                    name=name,
                    description=fm.get("description", ""),
                    path=skill_md,
                )
            )

        skills.sort(key=lambda s: s.name)
        self._cache = skills
        return skills

    def get_meta(self, name: str) -> SkillMeta | None:
        """Return meta for a specific skill by name."""
        return next((s for s in self.list_skills() if s.name == name), None)

    @staticmethod
    def _short_desc(desc: str, max_len: int = 80) -> str:
        """Truncate description to its first meaningful sentence, max max_len chars."""
        for sep in (".", "This includes", "Triggers"):
            idx = desc.find(sep)
            if idx > 20:
                desc = desc[:idx].rstrip(". ")
                break
        if len(desc) > max_len:
            desc = desc[: max_len - 3] + "..."
        return desc

    def summary_lines(self, n: int = 5) -> list[str]:
        """Return top-n skill one-liners for system prompt injection.

        Format: '- **pptx** (`/abs/path/skills/pptx/SKILL.md`): Create PowerPoint presentations'
        The absolute path lets the model Read the skill file without guessing the CWD.
        """
        return [f"- **{s.name}** (`{s.path}`): {self._short_desc(s.description)}" for s in self.list_skills()[:n]]

    def xml_block(self, n: int = 5) -> str:
        """Return top-n skills as an XML block for system prompt injection.

        Format (OpenClaw-compatible):
            <available_skills>
              <skill>
                <name>deep-research</name>
                <description>Use this skill when...</description>
                <location>/abs/path/skills/deep-research/SKILL.md</location>
              </skill>
              ...
            </available_skills>
        """
        lines = ["<available_skills>"]
        for s in self.list_skills()[:n]:
            lines.append("  <skill>")
            lines.append(f"    <name>{s.name}</name>")
            lines.append(f"    <description>{self._short_desc(s.description)}</description>")
            lines.append(f"    <location>{s.path}</location>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)
