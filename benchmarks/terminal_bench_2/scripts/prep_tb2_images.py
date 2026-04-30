#!/usr/bin/env python3
"""Pre-build TB2 task images with curl and uv 0.9.5 pre-installed.

Each task image (alexgshaw/<task>:20251031) is rebuilt from scratch with:
  - curl installed via apt
  - uv binary copied from the host (no internet download needed)
  - /root/.local/bin/env activation script written

The rebuilt image is tagged as the SAME original name, so OpenSandbox
picks up the local image without any config changes (pull-if-not-present
policy means the local image wins).

This is a one-time operation (or re-run when adding new task images).

Usage:
  # Prepare all locally cached images
  python prep_tb2_images.py

  # Prepare all images (pulls missing ones first)
  python prep_tb2_images.py --all

  # Prepare specific images
  python prep_tb2_images.py --image alexgshaw/code-from-image:20251031

  # Dry run to see what would be built
  python prep_tb2_images.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_DEFAULT_UV = os.environ.get("TB2_UV_BINARY", "") or "/root/.local/bin/uv"
_HARBOR_TASK_CACHE = "/root/.cache/harbor/tasks"

# Dockerfile template.  The uv binary is COPY-ed from the build context
# (no internet needed).  curl is installed via apt (internet needed once,
# during the one-time build — not at every evaluation run).
_DOCKERFILE = """\
FROM {image}
ARG http_proxy
ARG https_proxy
COPY uv /usr/local/bin/uv
RUN set -eux; \\
    chmod +x /usr/local/bin/uv && \\
    ln -sf /usr/local/bin/uv /usr/local/bin/uvx && \\
    mkdir -p /root/.local/bin && \\
    ln -sf /usr/local/bin/uv /root/.local/bin/uv && \\
    ln -sf /usr/local/bin/uv /root/.local/bin/uvx && \\
    printf '#!/bin/sh\\nexport PATH="/root/.local/bin:/usr/local/bin:$PATH"\\n' \\
        > /root/.local/bin/env && \\
    chmod +x /root/.local/bin/env && \\
    apt-get update -qq && \\
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && \\
    rm -rf /var/lib/apt/lists/*
"""


def find_task_images() -> list[str]:
    """Discover all task docker images from harbor task.toml files."""
    result = subprocess.run(
        ["find", _HARBOR_TASK_CACHE, "-name", "task.toml"],
        capture_output=True,
        text=True,
    )
    images: set[str] = set()
    for path in result.stdout.splitlines():
        path = path.strip()
        if not path:
            continue
        try:
            content = Path(path).read_text()
            for line in content.splitlines():
                if line.strip().startswith("docker_image"):
                    image = line.split("=", 1)[1].strip().strip('"')
                    images.add(image)
        except Exception:
            pass
    return sorted(images)


def is_image_local(image: str) -> bool:
    r = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
    )
    return r.returncode == 0


def is_image_prepped(image: str) -> bool:
    """Return True if this image was already built by us (has the prep label)."""
    r = subprocess.run(
        ["docker", "image", "inspect", "--format", '{{index .Config.Labels "tb2.prepped"}}', image],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def pull_image(image: str) -> bool:
    print(f"  Pulling {image} ...", flush=True)
    r = subprocess.run(["docker", "pull", image])
    return r.returncode == 0


def prep_image(image: str, uv_path: str, dry_run: bool = False, force: bool = False) -> bool:
    """Build a prepared version of `image` with uv + curl pre-installed.

    Tags the result as the original image name so OpenSandbox uses it
    automatically (no config changes needed).
    """
    if not force and is_image_prepped(image):
        print(f"  SKIP {image} (already prepped, use --force to rebuild)", flush=True)
        return True

    print(f"  Building {image} ...", flush=True)
    if dry_run:
        print(f"    [dry-run] would: docker build -t {image} <context>", flush=True)
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write Dockerfile
        Path(tmpdir, "Dockerfile").write_text(_DOCKERFILE.format(image=image))
        # Copy uv binary into build context (avoids network fetch at build time)
        shutil.copy2(uv_path, Path(tmpdir, "uv"))
        # Add label so we can detect already-prepped images
        label_arg = ["--label", "tb2.prepped=true"]

        # Pass proxy through to apt-get inside the build only from standard system proxy vars
        # (OPENSANDBOX_PROXY is for OpenSandbox API calls only, not general internet)
        proxy_args: list[str] = []
        proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
        if proxy:
            proxy_args = ["--build-arg", f"http_proxy={proxy}", "--build-arg", f"https_proxy={proxy}"]

        r = subprocess.run(
            ["docker", "build", "--network=host", *label_arg, *proxy_args, "--tag", image, tmpdir],
        )
        if r.returncode != 0:
            print(f"  ERROR: build failed for {image}", file=sys.stderr)
            return False

    print(f"  OK {image}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-build TB2 task images with uv + curl (SWE-bench-docker style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--uv",
        default=_DEFAULT_UV,
        help=f"Path to the host uv binary (default: {_DEFAULT_UV})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Pull and prepare all task images, not just locally cached ones",
    )
    parser.add_argument(
        "--image",
        action="append",
        dest="images",
        metavar="IMAGE",
        help="Specific image(s) to prepare (may be repeated)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even images already marked as prepped",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    uv_path = args.uv
    if not Path(uv_path).is_file():
        print(f"ERROR: uv binary not found at {uv_path}", file=sys.stderr)
        print("Set TB2_UV_BINARY env var or pass --uv <path>", file=sys.stderr)
        sys.exit(1)

    images = args.images or find_task_images()
    if not images:
        print("No task images found. Make sure harbor task cache is populated.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} task image(s). uv binary: {uv_path}")
    ok = skip = fail = pulled = 0

    for image in images:
        local = is_image_local(image)
        if not local:
            if args.all:
                if not args.dry_run:
                    if not pull_image(image):
                        print(f"  ERROR: pull failed for {image}", file=sys.stderr)
                        fail += 1
                        continue
                    pulled += 1
                else:
                    print(f"  [dry-run] would pull {image}", flush=True)
                    pulled += 1
            else:
                print(f"  SKIP {image} (not cached locally; use --all to pull first)", flush=True)
                skip += 1
                continue

        if prep_image(image, uv_path, dry_run=args.dry_run, force=args.force):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} prepped, {skip} skipped (not local), {fail} failed" + (f", {pulled} pulled" if pulled else ""))
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
