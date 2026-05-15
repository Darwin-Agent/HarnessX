import pytest
from pathlib import Path
from harnessx.aegis.apply import validate_applied_config, ApplyError


def test_missing_file_raises(tmp_path):
    with pytest.raises(ApplyError):
        validate_applied_config(tmp_path / "nope.yaml")


def test_malformed_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("this is not yaml: :::: broken")
    with pytest.raises(ApplyError):
        validate_applied_config(p)


def test_valid_minimal_config_passes(tmp_path):
    from harnessx.core.builder import HarnessBuilder
    cfg = HarnessBuilder().build()
    p = tmp_path / "good.yaml"
    cfg.to_yaml_file(p)
    result = validate_applied_config(p)
    assert result.canonicalized is True
    assert result.applied_path == p
