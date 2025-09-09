# %%
# type: ignore

# %% [markdown]
# # Dynamic In-Context Learning
#
# This recipe allows TensorZero users to set up a dynamic in-context learning variant for any function.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to query a set of good examples and retrieve the most relevant ones to put them into context for future inferences.
# Since TensorZero allows users to add demonstrations for any inference it is also easy to include them in the set of examples as well.
# This recipe will show use the OpenAI embeddings API only, but we have support for other embeddings providers as well.
#

# %% [markdown]
# To get started:
#
# - Set the `TENSORZERO_CLICKHOUSE_URL` environment variable. For example: `TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero"`
# - Set the `OPENAI_API_KEY` environment variable.
# - Update the following parameters
# - Uncomment query filters as appropriate
#

# %%
from typing import Optional

CONFIG_PATH = "../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

METRIC_NAME: Optional[str] = None

MAX_EXAMPLES = 1000

# The name of the DICL variant you will want to use. Set this to a meaningful name that does not conflict
# with other variants for the function selected above.
DICL_VARIANT_NAME = "gpt_4o_mini_dicl"

# The model to use for the DICL variant. Should match the name of the embedding model defined in your config
DICL_EMBEDDING_MODEL = "openai::text-embedding-3-small"

# The model to use for generation in the DICL variant
DICL_GENERATION_MODEL = "openai::gpt-4o-2024-08-06"

# The number of examples to retrieve for the DICL variant
DICL_K = 10

# If the metric is a float metric, you can set the threshold to filter the data
FLOAT_METRIC_THRESHOLD = 0.5

# Whether to use demonstrations for DICL examples
USE_DEMONSTRATIONS = True

# %%
import os
from asyncio import Semaphore

import pandas as pd
import toml
from clickhouse_connect import get_client
from openai import AsyncOpenAI
from tensorzero import TensorZeroGateway, patch_openai_client
from tensorzero.util import uuid7
from tqdm.asyncio import tqdm_asyncio

# %% [markdown]
# Initialize the ClickHouse client.
#

# %%
assert "TENSORZERO_CLICKHOUSE_URL" in os.environ, (
    "TENSORZERO_CLICKHOUSE_URL environment variable not set"
)

clickhouse_client = get_client(dsn=os.environ["TENSORZERO_CLICKHOUSE_URL"])

# %% [markdown]
# Initialize the TensorZero Client
#

# %%
t0 = TensorZeroGateway.build_embedded(
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"], config_file=CONFIG_PATH
)

# %%
openai_client = await patch_openai_client(
    AsyncOpenAI(),
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
    config_file=CONFIG_PATH,
    async_setup=True,
)

# %%
filters = None
# To filter on a boolean metric, you can uncomment the following line
# filters = BooleanMetricFilter(metric_name=METRIC_NAME, value=True) # or False as needed

# To filter on a float metric, you can uncomment the following line
# filters = FloatMetricFilter(metric_name=METRIC_NAME, value=0.5, comparison_operator=">")
# or any other float value as needed
# You can even use AND, OR, and NOT operators to combine multiple filters

# %%
inferences = t0.experimental_list_inferences(
    function_name=FUNCTION_NAME,
    filters=filters,
    output_source="demonstration",
    # or "inference" if you don't want to use (or don't have) demonstrations
    # if you use "demonstration" we will restrict to the subset of infereences
    # that have demonstrations
    limit=MAX_EXAMPLES,
)


# %%
async def get_embedding(
    text: str, semaphore: Semaphore, model: str = "text-embedding-3-small"
) -> Optional[list[float]]:
    try:
        async with semaphore:
            response = await openai_client.embeddings.create(
                input=text, model=f"tensorzero::embedding_model_name::{model}"
            )
            return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


# %%
MAX_CONCURRENT_EMBEDDING_REQUESTS = 50
semaphore = Semaphore(MAX_CONCURRENT_EMBEDDING_REQUESTS)

# %%
# Embed the 'input' column using the get_embedding function
tasks = [
    get_embedding(str(inference.input), semaphore, DICL_EMBEDDING_MODEL)
    for inference in inferences
]
embeddings = await tqdm_asyncio.gather(*tasks, desc="Embedding inputs")

# %%
data = []
for inference, embedding in zip(inferences, embeddings):
    data.append(
        {
            "input": str(inference.input),
            "output": str(inference.output),
            "embedding": embedding,
            "function_name": FUNCTION_NAME,
            "variant_name": DICL_VARIANT_NAME,
            "id": uuid7(),
        }
    )
example_df = pd.DataFrame(data)
example_df.head()

# %% [markdown]
# Prepare the data for the DynamicInContextLearningExample table
# The table schema is as follows:
#
# ```
# CREATE TABLE tensorzero.DynamicInContextLearningExample
# (
#     `id` UUID,
#     `function_name` LowCardinality(String),
#     `variant_name` LowCardinality(String),
#     `namespace` String,
#     `input` String,
#     `output` String,
#     `embedding` Array(Float32),
#     `timestamp` DateTime MATERIALIZED UUIDv7ToDateTime(id)
# )
# ENGINE = MergeTree
# ORDER BY (function_name, variant_name, namespace)
# ```
#

# %%
# Insert the data into the DiclExample table
result = clickhouse_client.insert_df(
    "DynamicInContextLearningExample",
    example_df,
)
print(result)

# %% [markdown]
# Finally, add a new variant to your function configuration to try out the Dynamic In-Context Learning variant in practice!
#
# If your embedding model name or generation model name in the config is different from the one you used above, you might have to update the config.
# Be sure and also give the variant some weight and if you are using a JSON function set the json_mode field to "strict" if you want.
#
# > **Tip:** DICL variants support additional parameters like system instructions or strict JSON mode. See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference).
#

# %%
variant_config = {
    "type": "experimental_dynamic_in_context_learning",
    "embedding_model": DICL_EMBEDDING_MODEL,
    "model": DICL_GENERATION_MODEL,
    "k": DICL_K,
}
full_variant_config = {
    "functions": {FUNCTION_NAME: {"variants": {DICL_VARIANT_NAME: variant_config}}}
}

print(toml.dumps(full_variant_config))
