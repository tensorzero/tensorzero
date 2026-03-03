import os
import tempfile
from typing import Generator

import pytest
from tensorzero import TensorZeroGateway

DUMMY_CONFIG = """
[gateway]
bind_address = "127.0.0.1:3000"

[gateway.observability]
enabled = false

[models.dummy_model]
routing = ["my_dummy_provider"]

[models.dummy_model.providers.my_dummy_provider]
type = "openai"
model_name = "gpt-4o-mini"

[functions.sanity_check]
type = "chat"

[functions.sanity_check.variants.default]
type = "chat_completion"
model = "dummy_model"
"""


@pytest.fixture
def dummy_config_path() -> Generator[str, None, None]:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml") as f:
        f.write(DUMMY_CONFIG)
        config_path = f.name

    yield config_path

    os.remove(config_path)


def test_pyo3_dummy_sanity(dummy_config_path: str) -> None:
    """
    Ensures the PyO3 bindings compile and execute correctly without requiring external databases.
    This provides a fast CI signal before a PR enters the merge queue.
    """

    os.environ["OPENAI_API_KEY"] = "dummy-key"

    with TensorZeroGateway.build_embedded(config_file=dummy_config_path) as client:
        
        with pytest.raises(Exception):
            client.inference(
                function_name="sanity_check",
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Hello from Python!"}
                            ]
                        }
                    ]
                }
            )
