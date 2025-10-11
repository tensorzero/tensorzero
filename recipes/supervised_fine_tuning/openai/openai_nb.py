# %%
# type: ignore

# %% [markdown]
# # OpenAI Supervised Fine-Tuning
#
# This recipe allows TensorZero users to fine-tune OpenAI models using their own data.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to fine-tune a model using your own data and any prompt you want.
#

# %% [markdown]
# To get started:
#
# - Set the `TENSORZERO_CLICKHOUSE_URL` environment variable. For example: `TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero"`
# - Set the `OPENAI_API_KEY` environment variable.
# - Update the following parameters:
#

# %%
CONFIG_PATH = "../../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

METRIC_NAME = "jaccard_similarity"

# The name of the variant to use to grab the templates used for fine-tuning
TEMPLATE_VARIANT_NAME = "gpt_4o_mini"

# If the metric is a float metric, you can set the threshold to filter the data
FLOAT_METRIC_THRESHOLD = 0.5

# Fraction of the data to use for validation
VAL_FRACTION = 0.2

# Maximum number of samples to use for fine-tuning
MAX_SAMPLES = 100_000

# The name of the model to fine-tune (supported models: https://platform.openai.com/docs/guides/fine-tuning)
MODEL_NAME = "gpt-4o-mini-2024-07-18"

# %%
import os
import sys

tensorzero_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
if tensorzero_path not in sys.path:
    sys.path.append(tensorzero_path)

# %%
import time

import toml
from IPython.display import clear_output
from tensorzero import (
    FloatMetricFilter,
    OpenAISFTConfig,
    OptimizationJobStatus,
    TensorZeroGateway,
)

from recipes.util import train_val_split

# %% [markdown]
# Initialize the TensorZero client
#

# %%
tensorzero_client = TensorZeroGateway.build_embedded(
    config_file=CONFIG_PATH,
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
    timeout=15,
)

# %% [markdown]
# Set the metric filter as needed
#

# %%
comparison_operator = ">="
metric_node = FloatMetricFilter(
    metric_name=METRIC_NAME,
    value=FLOAT_METRIC_THRESHOLD,
    comparison_operator=comparison_operator,
)

# from tensorzero import BooleanMetricFilter

# metric_node = BooleanMetricFilter(
#     metric_name=METRIC_NAME,
#     value=True  # or False
# )

metric_node

# %% [markdown]
# Query the inferences from ClickHouse
#

# %%
stored_inferences = tensorzero_client.experimental_list_inferences(
    function_name=FUNCTION_NAME,
    variant_name=None,
    output_source="inference",  # could also be "demonstration"
    filters=metric_node,
    limit=MAX_SAMPLES,
)

# %% [markdown]
# Render the inputs using the templates.
#

# %%
rendered_samples = tensorzero_client.experimental_render_inferences(
    stored_inferences=stored_inferences,
    variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME},
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
optimization_config = OpenAISFTConfig(
    model=MODEL_NAME,
)

job_handle = tensorzero_client.experimental_launch_optimization(
    train_samples=train_samples,
    val_samples=val_samples,
    optimization_config=optimization_config,
)

# %% [markdown]
# Wait for the fine-tuning job to complete.
#
# This cell will take a while to run.
#

# %%
while True:
    clear_output(wait=True)

    try:
        job_info = tensorzero_client.experimental_poll_optimization(job_handle=job_handle)
        print(job_info)
        if job_info.status in (
            OptimizationJobStatus.Completed,
            OptimizationJobStatus.Failed,
        ):
            break
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(10)

# %% [markdown]
# Once the fine-tuning job is complete, you can add the fine-tuned model to your config file.
#

# %%
fine_tuned_model = job_info.output["routing"][0]
model_config = {
    "models": {
        fine_tuned_model: {
            "routing": ["openai"],
            "providers": {"openai": {"type": "openai", "model_name": fine_tuned_model}},
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
# You might also add other parameters (e.g. `temperature`) to the variant section in the config file.
#
