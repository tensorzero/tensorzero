import inspect
import os
from enum import Enum

import pytest
import pytest_asyncio
from pytest import FixtureRequest
from tensorzero import (
    AsyncTensorZeroGateway,
    TensorZeroGateway,
)

TEST_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../tensorzero-core/tests/e2e/tensorzero.toml",
)


class ClientType(Enum):
    HttpGateway = 0
    EmbeddedGateway = 1


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


# Shared fixtures for both HTTP and embedded clients
@pytest_asyncio.fixture(params=[ClientType.HttpGateway, ClientType.EmbeddedGateway])
async def async_client(request: FixtureRequest):
    if request.param == ClientType.HttpGateway:
        client_fut = AsyncTensorZeroGateway.build_http(
            gateway_url="http://localhost:3000",
            verbose_errors=True,
        )
        assert inspect.isawaitable(client_fut)
        async with await client_fut as client:
            yield client
    else:
        client_fut = AsyncTensorZeroGateway.build_embedded(
            config_file=TEST_CONFIG_FILE,
            clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
        )
        assert inspect.isawaitable(client_fut)
        async with await client_fut as client:
            yield client


@pytest.fixture(params=[ClientType.HttpGateway, ClientType.EmbeddedGateway])
def sync_client(request: FixtureRequest):
    if request.param == ClientType.HttpGateway:
        with TensorZeroGateway.build_http(
            gateway_url="http://localhost:3000",
            verbose_errors=True,
        ) as client:
            yield client
    else:
        with TensorZeroGateway.build_embedded(
            config_file=TEST_CONFIG_FILE,
            clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero-python-e2e",
        ) as client:
            yield client
