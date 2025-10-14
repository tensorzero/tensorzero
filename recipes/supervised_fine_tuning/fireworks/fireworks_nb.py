# %%
# type: ignore

# %% [markdown]
# # Fireworks Supervised Fine-Tuning
#
# This recipe allows TensorZero users to fine-tune open-source LLMs using their own data.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to fine-tune a model using your own data and any prompt you want.
# We follow the Fireworks [docs](https://docs.fireworks.ai/fine-tuning/fine-tuning-via-api) on fine-tuning a model.
#

# %% [markdown]
# To get started:
#
# - Set the `TENSORZERO_CLICKHOUSE_URL`, `FIREWORKS_API_KEY`, and `FIREWORKS_ACCOUNT_ID` environment variable. See the `.env.example` file.
# - Update the following parameters:
#

# %%
import os
import sys

from dotenv import load_dotenv

load_dotenv()

CLICKHOUSE_URL = os.getenv("TENSORZERO_CLICKHOUSE_URL")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
account_id = os.getenv("FIREWORKS_ACCOUNT_ID")

assert CLICKHOUSE_URL is not None, "TENSORZERO_CLICKHOUSE_URL is not set"
assert FIREWORKS_API_KEY is not None, "FIREWORKS_API_KEY is not set"
assert account_id is not None, "FIREWORKS_ACCOUNT_ID is not set"

tensorzero_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
if tensorzero_path not in sys.path:
    sys.path.append(tensorzero_path)

# %%
CONFIG_PATH = "../../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

METRIC_NAME = "jaccard_similarity"

# The name of the variant to use to grab the templates used for fine-tuning
TEMPLATE_VARIANT_NAME = "gpt_4o_mini"  # It's OK that this variant uses a different model than the one we're fine-tuning

# If the metric is a float metric, you can set the threshold to filter the data
FLOAT_METRIC_THRESHOLD = 0.5

# Fraction of the data to use for validation
VAL_FRACTION = 0.2

# Number of epochs to train for
NUM_EPOCHS = 1

# Maximum number of samples to use for fine-tuning (for Fireworks, NUM_EPOCHS * MAX_SAMPLES should be <= 3,000,000)
MAX_SAMPLES = 100_000

# The name of the model to fine-tune (supported models: https://docs.fireworks.ai/fine-tuning/fine-tuning-models#supported-base-models)
MODEL_NAME = "accounts/fireworks/models/llama-v3p1-8b-instruct"

# At the time of writing, Fireworks does not support tool call content blocks in assistant messages. Or the tool role.
# We will drop these invalid messages from the dataset by default.
# You can set this to False to keep the invalid messages in the dataset.
DROP_INVALID_MESSAGES = True

# %%
from time import sleep

import toml
from IPython.display import clear_output
from tensorzero import (
    FireworksSFTConfig,
    FloatMetricFilter,
    OptimizationJobStatus,
    TensorZeroGateway,
)

from recipes.util import train_val_split

# %% [markdown]
# Initialize the embedded TensorZero client
#

# %%
t0 = TensorZeroGateway.build_embedded(
    config_file=CONFIG_PATH,
    clickhouse_url=CLICKHOUSE_URL,
)

# %% [markdown]
# Query for stored examples
#

# %%
filters = FloatMetricFilter(metric_name=METRIC_NAME, value=FLOAT_METRIC_THRESHOLD, comparison_operator=">")
# from tensorzero import BooleanMetricFilter
# filters = BooleanMetricFilter(metric_name=METRIC_NAME, value=True)
# You could also train on demonstrations by changing the output_source to "demonstration"
stored_samples = t0.experimental_list_inferences(
    function_name=FUNCTION_NAME,
    filters=filters,
    output_source="inference",
    limit=MAX_SAMPLES,
)

# %% [markdown]
# Template the data using the variant we chose above.
#

# %%
rendered_samples = t0.experimental_render_samples(
    stored_samples=stored_samples, variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME}
)

# %% [markdown]
# Split the data into training and validation sets for fine-tuning.

# %%
train_samples, val_samples = train_val_split(
    rendered_samples,
    val_size=VAL_FRACTION,
    last_inference_only=True,
)

# %% [markdown]
# Launch the fine tuning job

# %%
optimization_config = FireworksSFTConfig(
    model=MODEL_NAME,
    account_id=account_id,
)

job_handle = t0.experimental_launch_optimization(
    train_samples=train_samples,
    val_samples=val_samples,
    optimization_config=optimization_config,
)

# %% [markdown]
# Wait for the fine-tuning job to complete.
#
# This cell will take a while to run.

# %%
while True:
    clear_output(wait=True)

    try:
        job_info = t0.experimental_poll_optimization(job_handle=job_handle)
        print(job_info)
        if job_info.status in (
            OptimizationJobStatus.Completed,
            OptimizationJobStatus.Failed,
        ):
            break
    except Exception as e:
        print(f"Error: {e}")

    sleep(10)

# %% [markdown]
# Once the fine-tuning job is complete, you can add the fine-tuned model to your config file.

# %%
fine_tuned_model = job_info.output["routing"][0]
model_config = {
    "models": {
        fine_tuned_model: {
            "routing": ["fireworks"],
            "providers": {"fireworks": {"type": "fireworks", "model_name": fine_tuned_model}},
        }
    }
}

print(toml.dumps(model_config))

# %% [markdown]
# Finally, add a new variant to your function to use the fine-tuned model.
#

# %% [markdown]
# You're all set!
#
# You can change the weight to enable a gradual rollout of the new model.
#
# You might also add other parameters (e.g. `max_tokens`, `temperature`) to the variant section in the config file.
#
