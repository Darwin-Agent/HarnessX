"""L5: validate_applied_config enforces prompt-bucket scratch-dir reference.

A prompt-bucket candidate MUST have at least one ``template_path`` in the
applied config that resolves INSIDE the per-candidate ``applied_scratch_dir``
— otherwise the "modification" is a silent no-op: R_{n+1} would run against
the unchanged shared template.
"""
import pytest
import yaml
from pathlib import Path

from harnessx.aegis.apply import validate_applied_config, ApplyError


def _write_cfg_with_template(
    out: Path, template_path: str, other_processors: list | None = None
) -> None:
    """Emit a minimal HarnessConfig YAML whose single processor references
    ``template_path``. The YAML is the raw serialized form — the canonicalize
    step happens at load-time and doesn't hit the template's contents, so
    the file doesn't have to actually exist for this test."""
    cfg = {
        "tool_registry": None,
        "tracer": None,
        "processors": [
            {
                "_target_": "harnessx.processors.context.system_prompt.SystemPromptProcessor",
                "system_builder": {
                    "_target_": "harnessx.processors.context.strategies.system_prompt.template.TemplateSystemPromptBuilder",
                    "template_path": template_path,
                },
            },
            *(other_processors or []),
        ],
        "workspace": None,
        "workspace_template": "default",
        "init_workspace": True,
        "step_snapshots": True,
        "sandbox_provider": None,
        "sandbox_hint_id": None,
        "plugins": [],
    }
    out.write_text(yaml.safe_dump(cfg))


def test_prompt_bucket_without_scratch_template_rejects(tmp_path):
    scratch = tmp_path / "applied" / "C-R1-01"
    scratch.mkdir(parents=True)
    shared_template = tmp_path / "shared_templates" / "agent.j2"
    shared_template.parent.mkdir(parents=True, exist_ok=True)
    shared_template.write_text("system prompt")

    applied = scratch / "config.yaml"
    _write_cfg_with_template(applied, str(shared_template))

    with pytest.raises(ApplyError) as exc:
        validate_applied_config(
            applied, expected_bucket="prompt", scratch_dir=scratch,
        )
    assert "scratch-dir template_path" in str(exc.value)


def test_prompt_bucket_with_scratch_template_accepts(tmp_path):
    scratch = tmp_path / "applied" / "C-R1-01"
    scratch.mkdir(parents=True)
    scratch_tpl = scratch / "gaia_agent.j2"
    scratch_tpl.write_text("system prompt (mutated)")

    applied = scratch / "config.yaml"
    _write_cfg_with_template(applied, str(scratch_tpl))

    result = validate_applied_config(
        applied, expected_bucket="prompt", scratch_dir=scratch,
    )
    assert result.canonicalized is True


def test_non_prompt_bucket_skips_scratch_check(tmp_path):
    """Tools/config/processor buckets don't need scratch-dir template_path."""
    scratch = tmp_path / "applied" / "C-R1-02"
    scratch.mkdir(parents=True)
    from harnessx.core.builder import HarnessBuilder
    cfg = HarnessBuilder().build()
    out = scratch / "config.yaml"
    cfg.to_yaml_file(out)
    # Tools bucket — no scratch enforcement; should pass despite no
    # template_path at all.
    result = validate_applied_config(
        out, expected_bucket="tools", scratch_dir=scratch,
    )
    assert result.canonicalized is True


def test_default_no_expected_bucket_keeps_legacy_behaviour(tmp_path):
    """Calling validate_applied_config with no bucket kwarg behaves as before."""
    from harnessx.core.builder import HarnessBuilder
    cfg = HarnessBuilder().build()
    out = tmp_path / "good.yaml"
    cfg.to_yaml_file(out)
    result = validate_applied_config(out)  # no expected_bucket / scratch_dir
    assert result.canonicalized is True
