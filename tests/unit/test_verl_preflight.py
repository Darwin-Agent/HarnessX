import importlib.metadata

import pytest

from recipe.verl.launch.preflight import check_versions


def test_supported_versions_accept_cuda_build_suffix():
    versions = {"sglang": "0.5.6", "torch": "2.9.1+cu128", "transformers": "4.57.1"}
    check_versions(versions.__getitem__)


@pytest.mark.parametrize(
    "name, version",
    [("sglang", "0.5.11"), ("torch", "2.8.0"), ("transformers", "5.0.0"), ("transformers", "4.50.0")],
)
def test_unsupported_versions_fail_before_starting_ray(name, version):
    versions = {"sglang": "0.5.6", "torch": "2.9.1", "transformers": "4.57.1", name: version}
    with pytest.raises(RuntimeError, match=f"Expected {name}"):
        check_versions(versions.__getitem__)


def test_missing_training_dependency_has_actionable_error():
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    with pytest.raises(RuntimeError, match="Missing training dependency: sglang"):
        check_versions(missing)
