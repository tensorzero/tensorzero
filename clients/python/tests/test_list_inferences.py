import os
from uuid import UUID

import pytest
import pytest_asyncio
from tensorzero import (
    TensorZeroGateway,
)

from fixtures import embedded_sync_client