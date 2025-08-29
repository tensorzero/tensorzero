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
# - Update the following parameters
# - Uncomment query filters as appropriate
#

# %%
from typing import Optional

CONFIG_PATH = "../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

METRIC_NAME: Optional[str] = None

MAX_EXAMPLES = 1000

# The name of the variant to use to grab the templates used for fine-tuning
TEMPLATE_VARIANT_NAME = "gpt_4o_mini"

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

import toml
from tensorzero import DiclOptimizationConfig, TensorZeroGateway

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

# %% [markdown]
# Initialize the TensorZero Client
#

# %%
t0 = TensorZeroGateway.build_embedded(
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"], config_file=CONFIG_PATH
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
stored_inferences = t0.experimental_list_inferences(
    function_name=FUNCTION_NAME,
    filters=filters,
    output_source="demonstration",
    # or "inference" if you don't want to use (or don't have) demonstrations
    # if you use "demonstration" we will restrict to the subset of infereences
    # that have demonstrations
    limit=MAX_EXAMPLES,
)

# %%
rendered_samples = t0.experimental_render_samples(
    stored_samples=stored_inferences,
    variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME},
)

# %%
optimization_config = DiclOptimizationConfig(
    embedding_model=DICL_EMBEDDING_MODEL,
    variant_name=DICL_VARIANT_NAME,
    function_name=FUNCTION_NAME,
    k=DICL_K,
    model=DICL_GENERATION_MODEL,
)
job_handle = t0.experimental_launch_optimization(
    train_samples=rendered_samples,
    val_samples=None,
    optimization_config=optimization_config,
)

# %%
job_info = t0.experimental_poll_optimization(job_handle=job_handle)

# %% [markdown]
# Finally, add a new variant to your function configuration to try out the Dynamic In-Context Learning variant in practice!
#
# If your embedding model name or generation model name in the config is different from the one you used above, you might have to update the config.
# Be sure and also give the variant some weight and if you are using a JSON function set the json_mode field to "strict" if you want.
#
# > **Tip:** DICL variants support additional parameters like system instructions or strict JSON mode. See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference).
#

# %%
full_variant_config = {
    "functions": {
        FUNCTION_NAME: {"variants": {DICL_VARIANT_NAME: job_info.output["content"]}}
    }
}

print(toml.dumps(full_variant_config))
