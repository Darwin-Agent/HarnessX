"""Materialize the pinned veRL source and local patch without editing site-packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = Path(__file__).resolve().parent / "dependencies"
DEFAULT_DESTINATION = ROOT / ".deps" / "verl"


def file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def expected_stamp():
    lock = json.loads((DEPENDENCIES / "verl.lock.json").read_text())
    return {**lock, "patch_sha256": file_sha256(DEPENDENCIES / "harnessx.patch")}


def check_installation(destination=DEFAULT_DESTINATION):
    destination = Path(destination)
    stamp_path = destination / ".harnessx-source.json"
    if not stamp_path.exists():
        raise RuntimeError("Pinned veRL is missing. Run: python -m recipe.verl.bootstrap")
    stamp = json.loads(stamp_path.read_text())
    if stamp.get("source") != expected_stamp():
        raise RuntimeError("veRL source pin or patch changed; rebuild with recipe.verl.bootstrap")
    for relative, digest in stamp["files"].items():
        path = destination / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"Pinned veRL source was modified: {relative}")


def materialize(archive=None, destination=DEFAULT_DESTINATION):
    destination = Path(destination).resolve()
    if destination.exists():
        check_installation(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = expected_stamp()
    with tempfile.TemporaryDirectory(prefix=".verl-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        if archive is None:
            archive = temporary / "upstream.tar.gz"
            with urllib.request.urlopen(expected["url"], timeout=120) as response, archive.open("wb") as target:
                shutil.copyfileobj(response, target)
        archive = Path(archive)
        if file_sha256(archive) != expected["sha256"]:
            raise ValueError("Upstream veRL archive SHA256 does not match the pinned source")
        extracted = temporary / "source"
        extracted.mkdir()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                target = (extracted / member.name).resolve()
                if not target.is_relative_to(extracted) or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsafe upstream archive member: {member.name}")
            tar.extractall(extracted, filter="data")
        roots = list(extracted.iterdir())
        if len(roots) != 1 or not (roots[0] / "verl" / "__init__.py").is_file():
            raise ValueError("Unexpected veRL archive layout")
        source = roots[0]
        subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-i", str(DEPENDENCIES / "harnessx.patch")],
            cwd=source,
            check=True,
        )
        for name in ("LICENSE", "Notice.txt"):
            if not (source / name).is_file():
                raise ValueError(f"Upstream license file is missing: {name}")
        files = {
            str(path.relative_to(source)): file_sha256(path)
            for path in sorted(source.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
        }
        (source / ".harnessx-source.json").write_text(json.dumps({"source": expected, "files": files}, indent=2) + "\n")
        os.rename(source, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local, checksum-verified upstream archive")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--check", action="store_true", help="Verify an existing checkout without changing it")
    arguments = parser.parse_args()
    if arguments.check:
        check_installation(arguments.destination)
    else:
        print(materialize(arguments.archive, arguments.destination))


if __name__ == "__main__":
    main()
