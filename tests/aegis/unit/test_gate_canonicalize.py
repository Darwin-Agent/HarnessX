from pathlib import Path
from harnessx.aegis.gates.canonicalize import check_canonicalize


def test_valid_yaml_passes(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "processors: []\n"
        "tool_registry:\n"
        "  builtin: []\n"
        "  custom: []\n"
    )
    result = check_canonicalize(cfg)
    assert result.ok


def test_malformed_yaml_fails(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("processors: [\n  unclosed")
    result = check_canonicalize(cfg)
    assert not result.ok
