# %%
# type: ignore

# %% [markdown]
# # Together Supervised Fine-Tuning
#
# This recipe allows TensorZero users to fine-tune Together models using their own data.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to fine-tune a model using your own data and any prompt you want.
#

# %% [markdown]
# To get started:
#
# - Set the `TENSORZERO_CLICKHOUSE_URL` environment variable. For example: `TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero"`
# - Set the `TOGETHER_API_KEY` environment variable.
# - Update the following parameters:
#

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

# Maximum number of samples to use for fine-tuning
MAX_SAMPLES = 100_000

# The name of the model to fine-tune (supported models: https://docs.together.ai/docs/fine-tuning-models)
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"

# At the time of writing, Together.ai does not support tool call content blocks in assistant messages. Or the tool role.
# We will drop these invalid messages from the dataset by default.
# You can set this to False to keep the invalid messages in the dataset.
DROP_INVALID_MESSAGES = True

# %%
import os
import sys

tensorzero_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
if tensorzero_path not in sys.path:
    sys.path.append(tensorzero_path)

# %%
import json
import os
import subprocess
import tempfile
import time
from pprint import pprint
from typing import Any, Dict, List

import requests
import toml
from IPython.display import clear_output
from tensorzero import (
    FloatMetricFilter,
    TensorZeroGateway,
)

from recipes.util import tensorzero_rendered_samples_to_conversations, train_val_split

# %% [markdown]
# Initialize the TensorZero client
#

# %%
tensorzero_client = TensorZeroGateway.build_embedded(
    config_file=CONFIG_PATH,
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
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
# Render the inputs using the templates in the template variant.
#

# %%
rendered_samples = tensorzero_client.experimental_render_samples(
    stored_samples=stored_inferences,
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
# Convert the rendered samples to openai format

# %%
train_samples = tensorzero_rendered_samples_to_conversations(
    train_samples, conversation_key="messages", join_text_blocks=True
)
val_samples = tensorzero_rendered_samples_to_conversations(
    val_samples, conversation_key="messages", join_text_blocks=True
)


# %% [markdown]
# We'll write the training and validation messages to temporary files for the Together CLI
#


# %%
def upload_dataset_to_together(samples: List[Dict[str, Any]]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write the conversational_messages to the temporary file
        for item in samples:
            json.dump(item, f)
            f.write("\n")
        f.flush()

        dataset_path = f.name
        result = subprocess.run(["together", "files", "upload", dataset_path], capture_output=True)
        print("Stdout:")
        print(result.stdout.decode())
        print("Stderr:")
        print(result.stderr.decode())
        together_result = json.loads(result.stdout)
        return together_result["id"]


train_file_object_id = upload_dataset_to_together(train_samples)
val_file_object_id = upload_dataset_to_together(val_samples)

# %% [markdown]
# Launch the fine-tuning job.
#

# %%
url = "https://api.together.xyz/v1/fine-tunes"
print("MODEL: ", MODEL_NAME)
print("Train: ", train_file_object_id)
print("Val: ", val_file_object_id)

payload = {
    "training_file": train_file_object_id,
    "validation_file": val_file_object_id,
    "model": MODEL_NAME,
    "n_epochs": 1,
    "n_checkpoints": 1,
    "n_evals": 0,
    "batch_size": 16,
    "learning_rate": 0.00001,
    "lr_scheduler": {"lr_scheduler_args": {"min_lr_ratio": 0}},
    "warmup_ratio": 0,
    "max_grad_norm": 1,
    "weight_decay": 0,
    "train_on_inputs": "auto",
    "training_type": {"type": "Lora", "lora_r": 8, "lora_alpha": 32},
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}",
}

response = requests.post(url, json=payload, headers=headers)
print("Response status: ", response.status_code)
print("Response body: ")
print(response.text)
response_json = json.loads(response.text)
fine_tune_id = response_json["id"]

# %% [markdown]
# Wait for the fine-tuning job to complete.
#
# This cell will take a while to run.
#

# %%
while True:
    clear_output(wait=True)

    try:
        job_status = requests.get(
            f"https://api.together.xyz/v1/fine-tunes/{fine_tune_id}",
            headers={
                "accept": "application/json",
                "authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}",
            },
        ).json()
        pprint(job_status)
        print("Status: ", job_status["status"])
        if job_status["status"] in ("completed", "failed", "cancelled"):
            break
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(10)

# %% [markdown]
# Once the fine-tuning job is complete, you can add the fine-tuned model to your config file.
#

# %%
fine_tuned_model = job_status["model_output_name"]
model_config = {
    "models": {
        fine_tuned_model: {
            "routing": ["together"],
            "providers": {"together": {"type": "together", "model_name": fine_tuned_model}},
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
