# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from harnessx.tools.builtin import build_gaia_tools


def test_build_gaia_tools_contains_full_fs_toolset():
    reg = build_gaia_tools()
    names = set(reg.list_names())
    assert {"WebSearch", "WebFetch", "Browser", "Read", "Bash"} <= names
    assert {"Write", "Edit", "Glob", "Grep"} <= names, f"Missing filesystem tools — got {sorted(names)}"
