"""
Build DigestAgent HarnessConfig with per-round workspace, write gate, and
cross-round notebook.

All static processors/plugins are declared in default_config.yaml.
This function handles the three runtime concerns:

  1. Workspace  — isolated at session output_dir so the agent can traverse
                  trajectory files freely (read), but cannot escape the
                  session directory.
                  WorkspaceInitializer(prompts_root=digest/prompts/) copies
                  AGENTS.md (the digest system prompt) into the workspace;
                  DefaultSystemPromptBuilder reads it from there at task_start.
                  init_workspace=False is set on HarnessConfig so the default
                  WorkspaceInitializer() call in harness.run() is skipped.

  2. WriteScopeGateProcessor — further restricts writes inside the workspace
                  to *only* the notebook file.  Trajectory JSONL files and
                  all other session artefacts are read-only from the agent's
                  perspective even though the workspace mode is "isolated".

  3. EvolutionNotebookProcessor — injects cross-round pending hypotheses,
                  open todos, and failed interventions into the system prompt
                  so the agent builds on previous rounds' analysis.

Usage:
    config = await build_digest_config(output_dir)
    config = await build_digest_config(
        output_dir,
        skill_dirs=[Path("recipe/my_bench/skills/bench-context")],
    )
    harness = digest_model.agentic(config)
"""
from __future__ import annotations

from pathlib import Path

from harnessx.core.harness import HarnessConfig
from harnessx.workspace.initializer import WorkspaceInitializer
from harnessx.workspace.skill_manager import SkillManager
from harnessx.workspace.workspace import Workspace

_DEFAULT_CONFIG = Path(__file__).parent / "default_config.yaml"
_PROMPTS_DIR = Path(__file__).parent / "prompts"

_WRITE_GATE = "harnessx.experimental.harness_evol.processors.write_scope_gate.WriteScopeGateProcessor"
_NOTEBOOK = "harnessx.experimental.harness_evol.processors.notebook.EvolutionNotebookProcessor"


async def build_digest_config(
    output_dir: Path,
    config_yaml: Path | None = None,
    skill_dirs: list[Path] | None = None,
    notebook_path: Path | None = None,
) -> HarnessConfig:
    """
    Build DigestAgent HarnessConfig for one round.

    Args:
        output_dir:    Session output directory (shared across rounds).
                       Workspace root is set here so the agent can read
                       trajectory files from any round sub-directory.
        config_yaml:   Override default_config.yaml (processors + tools + plugins).
        skill_dirs:    Benchmark-specific skill directories, each containing SKILL.md.
                       Built-in extensions/skills/ are never copied for DigestAgent.
        notebook_path: Shared evolution notebook (DigestAgent writes, EvolveAgent reads).
                       Pass a session-level path to persist notes across rounds.
                       Defaults to output_dir / "digest-workspace/evolution_notebook.md" (per-round).
    """
    notebook_path = notebook_path or (output_dir / "digest-workspace" / "evolution_notebook.md")

    # Workspace root is always inside output_dir so AGENTS.md, sessions/, and skills/
    # are co-located with per-round outputs.
    #
    # Write scope: WriteScopeGateProcessor (configured below) restricts writes to
    # ONLY the notebook file, even though the workspace home may be wider.
    #
    # Home scope: notebook_path may live OUTSIDE output_dir (e.g. at run_dir level for
    # cross-round persistence).  We use mode="home" with home=notebook_path.parent so
    # the Workspace boundary covers both output_dir reads and notebook writes.
    # In practice: notebook at run_dir/evolution_notebook.md → home=run_dir, which
    # also covers output_dir (= run_dir/round_N/evolve/) as a descendant.
    notebook_home = notebook_path.resolve().parent
    digest_workspace_dir = output_dir / "digest-workspace"
    ws = Workspace(agent_id="digest", root=digest_workspace_dir, mode="home", home=str(notebook_home))

    # Initialise workspace with digest-specific prompts (AGENTS.md = digest system prompt).
    # copy_skills=False: DigestAgent never needs built-in extensions/skills/ (docx/pdf/etc.).
    # Benchmark-specific skills are installed below via SkillManager when skill_dirs is given.
    # init_workspace=False below prevents harness.run() from re-running the default init.
    await WorkspaceInitializer(prompts_root=_PROMPTS_DIR).initialize(
        ws, copy_skills=False
    )

    if skill_dirs:
        # agent_id="" → installs to ws_root/skills/ which matches workspace root + AGENTS.md discovery
        mgr = SkillManager(ws_root=digest_workspace_dir, agent_id="")
        for sd in skill_dirs:
            mgr.install(str(sd.resolve()))

    base = HarnessConfig.from_yaml_file(config_yaml or _DEFAULT_CONFIG)

    # Substitute runtime values into the placeholder processor dicts from YAML.
    nb_str = str(notebook_path)
    updated = []
    for p in base.processors:
        target = p.get("_target_", "")
        if target == _WRITE_GATE:
            p = {**p, "allowed_files": [nb_str]}
        elif target == _NOTEBOOK:
            p = {**p, "notebook_path": nb_str}
        updated.append(p)

    return base.copy(workspace=ws, processors=updated, init_workspace=False)
