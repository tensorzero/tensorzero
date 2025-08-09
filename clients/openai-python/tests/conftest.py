# type: ignore
"""
Shared test fixtures for TensorZero OpenAI client tests
"""

import os

import pytest_asyncio
from openai import AsyncOpenAI

TEST_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../tensorzero-core/tests/e2e/tensorzero.toml",
)


@pytest_asyncio.fixture
async def async_client():
    async with AsyncOpenAI(
        api_key="donotuse", base_url="http://localhost:3000/openai/v1"
    ) as client:
        yield client
