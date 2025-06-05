import inspect
import os

import pytest
import pytest_asyncio
from tensorzero import (
    AsyncTensorZeroGateway,
    TensorZeroGateway,
)

TEST_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../tensorzero-internal/tests/e2e/tensorzero.toml",
)


@pytest.fixture
def embedded_sync_client():
    with TensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
    ) as client:
        yield client


@pytest_asyncio.fixture
async def embedded_async_client():
    client_fut = AsyncTensorZeroGateway.build_embedded(
        config_file=TEST_CONFIG_FILE,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
    )
    assert inspect.isawaitable(client_fut)
    async with await client_fut as client:
        yield client
