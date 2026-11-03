import io
import json
import tarfile

import pytest

from recipe.verl import bootstrap


def make_archive(path, members=None):
    with tarfile.open(path, "w:gz") as archive:
        for name, value in (
            members
            or {
                "upstream/verl/__init__.py": b"value = 1\n",
                "upstream/LICENSE": b"license",
                "upstream/Notice.txt": b"notice",
            }
        ).items():
            member = tarfile.TarInfo(name)
            if value is None:
                member.type = tarfile.SYMTYPE
                member.linkname = "/tmp/outside"
                archive.addfile(member)
            else:
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))


@pytest.fixture
def pinned_source(tmp_path, monkeypatch):
    archive = tmp_path / "source.tar.gz"
    make_archive(archive)
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    lock = {"version": "test", "commit": "test", "sha256": bootstrap.file_sha256(archive)}
    (dependencies / "verl.lock.json").write_text(json.dumps(lock))
    (dependencies / "harnessx.patch").write_text(
        "--- a/verl/__init__.py\n+++ b/verl/__init__.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    monkeypatch.setattr(bootstrap, "DEPENDENCIES", dependencies)
    return archive, tmp_path / "checkout"


def test_bootstrap_applies_patch_preserves_licenses_and_is_idempotent(pinned_source):
    archive, destination = pinned_source
    assert bootstrap.materialize(archive, destination) == destination
    assert (destination / "verl/__init__.py").read_text() == "value = 2\n"
    assert (destination / "LICENSE").read_text() == "license"
    assert (destination / "Notice.txt").read_text() == "notice"
    bootstrap.check_installation(destination)
    assert bootstrap.materialize(archive, destination) == destination


def test_bootstrap_rejects_checksum_mismatch_without_partial_checkout(pinned_source):
    archive, destination = pinned_source
    archive.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA256"):
        bootstrap.materialize(archive, destination)
    assert not destination.exists()


@pytest.mark.parametrize("member,value", [("../../outside", b"bad"), ("/absolute", b"bad"), ("upstream/link", None)])
def test_bootstrap_rejects_unsafe_archive(pinned_source, member, value):
    archive, destination = pinned_source
    make_archive(archive, {member: value})
    lock_path = bootstrap.DEPENDENCIES / "verl.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["sha256"] = bootstrap.file_sha256(archive)
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="Unsafe"):
        bootstrap.materialize(archive, destination)
    assert not destination.exists()


def test_bootstrap_detects_modified_source_and_patch(pinned_source):
    archive, destination = pinned_source
    bootstrap.materialize(archive, destination)
    source = destination / "verl/__init__.py"
    source.write_text("modified\n")
    with pytest.raises(RuntimeError, match="modified"):
        bootstrap.check_installation(destination)
    source.write_text("value = 2\n")
    (bootstrap.DEPENDENCIES / "harnessx.patch").write_text("changed patch")
    with pytest.raises(RuntimeError, match="pin or patch changed"):
        bootstrap.materialize(archive, destination)


def test_checked_in_lock_identifies_immutable_source_and_license():
    lock = bootstrap.expected_stamp()
    assert lock["version"] == "0.7.1"
    assert len(lock["commit"]) == 40
    assert lock["commit"] in lock["url"]
    assert len(lock["sha256"]) == 64
    assert "Apache License" in (bootstrap.DEPENDENCIES / "LICENSE").read_text()
    assert (bootstrap.DEPENDENCIES / "Notice.txt").is_file()
