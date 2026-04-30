# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import warnings

from harnessx.tools.builtin import browser


class TestBrowserAtexit:
    def test_atexit_close_handles_event_loop_creation_failure(self, monkeypatch) -> None:
        def _boom():
            raise RuntimeError("loop init failed")

        monkeypatch.setattr(browser.asyncio, "new_event_loop", _boom)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            browser._atexit_close()

        assert not any("was never awaited" in str(w.message) for w in captured)

    def test_atexit_close_closes_loop_even_when_run_interrupted(self, monkeypatch) -> None:
        class _FakeLoop:
            def __init__(self) -> None:
                self._calls = 0
                self.closed = False

            def run_until_complete(self, _coro):
                self._calls += 1
                if self._calls == 1:
                    raise KeyboardInterrupt()
                raise RuntimeError("shutdown interrupted")

            async def shutdown_asyncgens(self):
                return None

            def close(self) -> None:
                self.closed = True

        fake = _FakeLoop()
        monkeypatch.setattr(browser.asyncio, "new_event_loop", lambda: fake)

        browser._atexit_close()
        assert fake.closed is True
