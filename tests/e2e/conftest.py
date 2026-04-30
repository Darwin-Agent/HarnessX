# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
import asyncio
import os
import pytest
from ._utils import load_provider

_SCENARIO_TIMEOUT = 300  # 5 minutes per test scenario


def pytest_configure(config):
    """Set a sane per-request timeout for all e2e tests."""
    os.environ.setdefault("HARNESSX_REQUEST_TIMEOUT", "30")
    from ._utils import get_test_home

    os.environ.setdefault("HARNESSX_HOME", str(get_test_home()))


@pytest.fixture(autouse=True)
async def scenario_timeout():
    """Hard 5-minute ceiling on every async e2e test."""
    async with asyncio.timeout(_SCENARIO_TIMEOUT):
        yield


@pytest.fixture
def provider():
    """Model provider built from tests/e2e/.env (ANTHROPIC_*, OPENAI_*, or LITELLM_* keys)."""
    return load_provider()
