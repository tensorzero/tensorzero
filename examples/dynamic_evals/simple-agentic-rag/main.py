import asyncio
import os

from dataset import load_beerqa
from tensorzero import AsyncTensorZeroGateway


async def main():
    # We initialize an embedded TensorZero client with our configuration file.
    t0 = await AsyncTensorZeroGateway.build_embedded(
        config_file="config/tensorzero.toml",
        clickhouse_url=os.getenv("TENSORZERO_CLICKHOUSE_URL"),
    )

    data = load_beerqa()


if __name__ == "__main__":
    asyncio.run(main())
