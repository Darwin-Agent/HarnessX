# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import pytest

from examples.sem_weave import code_tools


@pytest.mark.parametrize("failure", ["connect", "tool"])
async def test_error_closes_registry_and_does_not_print_success(failure, capsys):
    class Registry:
        closed = False

        async def connect(self):
            if failure == "connect":
                raise RuntimeError("connection refused")

        async def execute(self, name, arguments):
            return SimpleNamespace(error="edit refused", output="not a success")

        async def disconnect(self):
            self.closed = True

    registry = Registry()
    with pytest.raises(RuntimeError):
        await code_tools.call(registry, "weave_edit", {"edits": []})
    assert registry.closed
    assert capsys.readouterr().out == ""
