# This is intentionally not a 'test_' file, since setting os.environ doesn't
# seem to work properly in pytest.

import os

os.environ["RUST_LOG"] = "foo=bar"

try:
    import tensorzero  # noqa # pyright: ignore[reportUnusedImport]

    raise Exception("TensorZero import succeeded - this should not happen")
except Exception as e:
    assert (
        "Internal TensorZero Error: Invalid `RUST_LOG` environment variable: invalid filter directive"
        in str(e)
    )
    print("Successfully caught exception: ", e)
