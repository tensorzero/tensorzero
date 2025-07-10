# %%
# type: ignore

# %% [markdown]
# # Dynamic In-Context Learning
#
# This recipe allows TensorZero users to set up a dynamic in-context learning variant for any function.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to query a set of good examples and retrieve the most relevant ones to put them into context for future inferences.
# Since TensorZero allows users to add demonstrations for any inference it is also easy to include them in the set of examples as well.
# This recipe will show use the OpenAI embeddings API only, but we are working towards support for all embedding providers over time as well.
#

# %% [markdown]
# To get started:
#
# - Set the `TENSORZERO_CLICKHOUSE_URL` environment variable. For example: `TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero"`
# - Set the `OPENAI_API_KEY` environment variable.
# - Update the following parameters:
#

# %%
from typing import Optional

CONFIG_PATH = "../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

# Can also set this to None if you do not want to use a metric and only want to use demonstrations
METRIC_NAME: Optional[str] = None

# The name of the DICL variant you will want to use. Set this to a meaningful name that does not conflict
# with other variants for the function selected above.
DICL_VARIANT_NAME = "gpt_4o_mini_dicl"

# The model to use for the DICL variant.
DICL_EMBEDDING_MODEL = "text-embedding-3-small"

# The model to use for generation in the DICL variant.
DICL_GENERATION_MODEL = "gpt-4o-mini-2024-07-18"

# The number of examples to retrieve for the DICL variant.
DICL_K = 10

# If the metric is a float metric, you can set the threshold to filter the data
FLOAT_METRIC_THRESHOLD = 0.5

# Whether to use demonstrations for DICL examples
USE_DEMONSTRATIONS = True

# %%
import os
from asyncio import Semaphore
from pathlib import Path

import pandas as pd
import toml
from clickhouse_connect import get_client
from openai import AsyncOpenAI
from tensorzero.util import uuid7
from tqdm.asyncio import tqdm_asyncio

# %% [markdown]
# Load the TensorZero configuration file.
#

# %%
config_path = Path(CONFIG_PATH)

assert config_path.exists(), f"{CONFIG_PATH} does not exist"
assert config_path.is_file(), f"{CONFIG_PATH} is not a file"

with config_path.open("r") as f:
    config = toml.load(f)

# %% [markdown]
# Retrieve the configuration for the function we are interested in.
#

# %%
assert "functions" in config, "No `[functions]` section found in config"
assert FUNCTION_NAME in config["functions"], (
    f"No function named `{FUNCTION_NAME}` found in config"
)

function_config = config["functions"][FUNCTION_NAME]
function_type = function_config["type"]

# %% [markdown]
# Retrieve the metric configuration.
#

# %%
if METRIC_NAME is None:
    metric = None
else:
    assert "metrics" in config, "No `[metrics]` section found in config"
    assert METRIC_NAME in config["metrics"], (
        f"No metric named `{METRIC_NAME}` found in config"
    )
    metric = config["metrics"][METRIC_NAME]

metric

# %% [markdown]
# Initialize the ClickHouse client.
#

# %%
assert "TENSORZERO_CLICKHOUSE_URL" in os.environ, (
    "TENSORZERO_CLICKHOUSE_URL environment variable not set"
)

clickhouse_client = get_client(dsn=os.environ["TENSORZERO_CLICKHOUSE_URL"])

# %% [markdown]
# Determine the ClickHouse table name for the function.
#

# %%
inference_table_name = {"chat": "ChatInference", "json": "JsonInference"}.get(
    function_type
)

if inference_table_name is None:
    raise ValueError(f"Unsupported function type: {function_type}")

# %% [markdown]
# Determine the ClickHouse table name for the metric.
#

# %%
feedback_table_name = (
    {
        "float": "FloatMetricFeedback",
        "boolean": "BooleanMetricFeedback",
    }.get(metric["type"])
    if metric is not None
    else None
)

if feedback_table_name is None and metric is not None:
    raise ValueError(f"Unsupported metric type: {metric['type']}")

# %% [markdown]
# Determine the correct join key to use for the metric on the inference table.
#

# %%
inference_join_key = (
    {
        "episode": "episode_id",
        "inference": "id",
    }.get(metric["level"])
    if metric is not None
    else None
)

if inference_join_key is None and metric is not None:
    raise ValueError(f"Unsupported metric level: {metric['level']}")

# %%
if metric is not None:
    assert "optimize" in metric, "Metric is missing the `optimize` field"

    threshold = FLOAT_METRIC_THRESHOLD if metric["type"] == "float" else 0.5
    comparison_operator = ">=" if metric["optimize"] == "max" else "<="

    query = f"""
    SELECT
        i.input,
        i.output,
    FROM
        {inference_table_name} i
    JOIN
        (SELECT
            target_id,
            value,
            ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
        FROM
            {feedback_table_name}
        WHERE
            metric_name = %(metric_name)s
            AND value {comparison_operator} %(threshold)s
        ) f ON i.{inference_join_key} = f.target_id and f.rn = 1
    WHERE
        i.function_name = %(function_name)s
    """

    params = {
        "function_name": FUNCTION_NAME,
        "metric_name": METRIC_NAME,
        "comparison_operator": comparison_operator,
        "threshold": threshold,
    }

    metric_df = clickhouse_client.query_df(query, params)

    metric_df.head()
else:
    metric_df = None

# %%
query = f"""
SELECT
    i.input,
    f.value AS output
FROM
    {inference_table_name} i
JOIN
    (SELECT
        inference_id,
        value,
        ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn
    FROM
        DemonstrationFeedback
    ) f ON i.id = f.inference_id AND f.rn = 1
WHERE
    i.function_name = %(function_name)s
"""

params = {
    "function_name": FUNCTION_NAME,
}

if USE_DEMONSTRATIONS:
    demonstration_df = clickhouse_client.query_df(query, params)

    demonstration_df.head()
else:
    demonstration_df = None

# %%
# Combine metric_df and demonstration_df into example_df
example_df = pd.concat(
    [df for df in [metric_df, demonstration_df] if df is not None], ignore_index=True
)

# Assert that at least one of the dataframes is not None
assert example_df is not None and not example_df.empty, (
    "Both metric_df and demonstration_df are None or empty"
)

# Display the first few rows of the combined dataframe
example_df.head()

# %%
openai_client = AsyncOpenAI()


# %%
async def get_embedding(
    text: str, semaphore: Semaphore, model: str = "text-embedding-3-small"
) -> Optional[list[float]]:
    try:
        async with semaphore:
            response = await openai_client.embeddings.create(input=text, model=model)
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
    get_embedding(str(input_text), semaphore, DICL_EMBEDDING_MODEL)
    for input_text in example_df["input"]
]
embeddings = await tqdm_asyncio.gather(*tasks, desc="Embedding inputs")

# %%
# Add the embeddings as a new column to the dataframe
example_df["embedding"] = embeddings

# Display the first few rows to verify the new column
print(example_df[["input", "embedding"]].head())

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
# Add a new column 'function_name' with the value FUNCTION_NAME for every row
example_df["function_name"] = FUNCTION_NAME

# Overwrite the 'variant_name' column with the value DICL_VARIANT_NAME for every row
example_df["variant_name"] = DICL_VARIANT_NAME

# Add a new column 'id' with a UUID for every row
example_df["id"] = [uuid7() for _ in range(len(example_df))]

# %%
example_df.head()

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

# %% [markdown]
# If you haven't, also include the embedding model in the config.
#

# %%
embedding_model_config = {
    "embedding_models": {
        DICL_EMBEDDING_MODEL: {
            "routing": ["openai"],
            "providers": {
                "openai": {"type": "openai", "model_name": DICL_EMBEDDING_MODEL}
            },
        }
    }
}

print(toml.dumps(embedding_model_config))
