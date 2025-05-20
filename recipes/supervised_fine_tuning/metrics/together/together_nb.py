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
CONFIG_PATH = "../../../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

METRIC_NAME = "exact_match"

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

# %%
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
import toml
from clickhouse_connect import get_client
from IPython.display import clear_output
from minijinja import Environment

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
# Retrieve the metric configuration.
#

# %%
assert "metrics" in config, "No `[metrics]` section found in config"
assert METRIC_NAME in config["metrics"], (
    f"No metric named `{METRIC_NAME}` found in config"
)

metric = config["metrics"][METRIC_NAME]

metric

# %% [markdown]
# Retrieve the configuration for the variant with the templates we'll use for fine-tuning.
#

# %%
assert "functions" in config, "No `[functions]` section found in config"
assert FUNCTION_NAME in config["functions"], (
    f"No function named `{FUNCTION_NAME}` found in config"
)
assert "variants" in config["functions"][FUNCTION_NAME], (
    f"No variants section found for function `{FUNCTION_NAME}`"
)
assert TEMPLATE_VARIANT_NAME in config["functions"][FUNCTION_NAME]["variants"], (
    f"No variant named `{TEMPLATE_VARIANT_NAME}` found in function `{FUNCTION_NAME}`"
)

function_type = config["functions"][FUNCTION_NAME]["type"]
variant = config["functions"][FUNCTION_NAME]["variants"][TEMPLATE_VARIANT_NAME]

variant

# %% [markdown]
# Retrieve the system, user, and assistant templates in the variant (if any), and initialize a minijinja environment with them.
#

# %%
templates = {}

if "assistant_template" in variant:
    assistant_template_path = config_path.parent / variant["assistant_template"]
    with assistant_template_path.open("r") as f:
        templates["assistant"] = f.read()

if "system_template" in variant:
    system_template_path = config_path.parent / variant["system_template"]
    with system_template_path.open("r") as f:
        templates["system"] = f.read()

if "user_template" in variant:
    user_template_path = config_path.parent / variant["user_template"]
    with user_template_path.open("r") as f:
        templates["user"] = f.read()

env = Environment(templates=templates)

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
feedback_table_name = {
    "float": "FloatMetricFeedback",
    "boolean": "BooleanMetricFeedback",
}.get(metric["type"])

if feedback_table_name is None:
    raise ValueError(f"Unsupported metric type: {metric['type']}")

# %% [markdown]
# Determine the correct join key to use for the metric on the inference table.
#

# %%
inference_join_key = {
    "episode": "episode_id",
    "inference": "id",
}.get(metric["level"])

if inference_join_key is None:
    raise ValueError(f"Unsupported metric level: {metric['level']}")

# %% [markdown]
# Query the inferences and feedback from ClickHouse.
#
# If the metric is a float metric, we need to filter the data based on the threshold.
#

# %%
assert "optimize" in metric, "Metric is missing the `optimize` field"

threshold = FLOAT_METRIC_THRESHOLD if metric["type"] == "float" else 0.5
comparison_operator = ">=" if metric["optimize"] == "max" else "<="

query = f"""
SELECT
    i.variant_name,
    i.input,
    i.output,
    f.value,
    i.episode_id
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
LIMIT %(max_samples)s
"""

params = {
    "function_name": FUNCTION_NAME,
    "metric_name": METRIC_NAME,
    "comparison_operator": comparison_operator,
    "threshold": threshold,
    "max_samples": MAX_SAMPLES,
}

df = clickhouse_client.query_df(query, params)

df.head()


# %% [markdown]
# Render the inputs using the templates.
#


# %%
def render_message(content: List[Dict[str, Any]], role: str) -> str:
    assert role in ["user", "assistant"], f"Invalid role: {role}"

    if len(content) != 1:
        raise ValueError(f"Message must have exactly one content block: {content}")

    if content[0]["type"] != "text":
        raise ValueError(f"Content block must be of type text: {content}")

    content = content[0]["value"]

    if isinstance(content, str):
        return content
    else:
        return env.render_template(role, **content)


def sample_to_conversational_messages(sample) -> List[Dict[str, str]]:
    function_input = json.loads(sample["input"])

    rendered_messages = []

    # Add the system message to the rendered messages
    # If there is data passed in or a system template there must be a system message
    system = function_input.get("system", {})
    if len(system) > 0 or system_template_path:
        if system_template_path:
            system_message = env.render_template("system", **system)
            rendered_messages.append({"role": "system", "content": system_message})
        else:
            rendered_messages.append({"role": "system", "content": system})

    # Add the input messages to the rendered messages
    for message in function_input["messages"]:
        rendered_message = render_message(message["content"], message["role"])
        rendered_messages.append({"role": message["role"], "content": rendered_message})

    # Add the output to the messages
    output = json.loads(sample["output"])

    if function_type == "chat":
        if len(output) != 1:
            raise ValueError(f"Output {output} must have exactly one content block.")

        if output[0]["type"] != "text":
            raise ValueError(f"Output {output} must be a text block.")

        rendered_messages.append({"role": "assistant", "content": output[0]["text"]})
    elif function_type == "json":
        rendered_messages.append({"role": "assistant", "content": output["raw"]})
    else:
        raise ValueError(f"Unsupported function type: {function_type}")

    return {"messages": rendered_messages}


df["conversational_messages"] = df.apply(sample_to_conversational_messages, axis=1)

df.head()

# %% [markdown]
# Split the data into training and validation sets for fine-tuning.
#

# %%
# Get unique episode_ids
unique_episode_ids = df["episode_id"].unique()

# Shuffle the unique episode_ids
np.random.seed(42)
np.random.shuffle(unique_episode_ids)

# Calculate the split index for episode_ids
split_index = int(len(unique_episode_ids) * (1 - VAL_FRACTION))

# Split the episode_ids into training and validation sets
train_episode_ids = unique_episode_ids[:split_index]
val_episode_ids = unique_episode_ids[split_index:]

# Create training and validation DataFrames based on episode_ids
train_df = df[df["episode_id"].isin(train_episode_ids)]
val_df = df[df["episode_id"].isin(val_episode_ids)]

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Actual validation fraction: {len(val_df) / len(df):.2f}")


# %% [markdown]
# We'll write the training and validation messages to temporary files for the Together CLI


# %%
def upload_dataset_to_together(df: pd.DataFrame) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write the conversational_messages to the temporary file
        for item in df["conversational_messages"]:
            json.dump(item, f)
            f.write("\n")
        f.flush()

        dataset_path = f.name
        result = subprocess.run(
            ["together", "files", "upload", dataset_path], capture_output=True
        )
        print("Stdout:")
        print(result.stdout.decode())
        print("Stderr:")
        print(result.stderr.decode())
        together_result = json.loads(result.stdout)
        return together_result["id"]


train_file_object_id = upload_dataset_to_together(train_df)
val_file_object_id = upload_dataset_to_together(val_df)

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
            "providers": {
                "together": {"type": "together", "model_name": fine_tuned_model}
            },
        }
    }
}

print(toml.dumps(model_config))

# %% [markdown]
# Finally, add a new variant to your function to use the fine-tuned model.
#

# %%
variant_config = {
    "type": "chat_completion",
    "weight": 0,
    "model": fine_tuned_model,
}

system_template = variant.get("system_template")
if system_template:
    variant_config["system_template"] = system_template

user_template = variant.get("user_template")
if user_template:
    variant_config["user_template"] = user_template

assistant_template = variant.get("assistant_template")
if assistant_template:
    variant_config["assistant_template"] = assistant_template

full_variant_config = {
    "functions": {FUNCTION_NAME: {"variants": {fine_tuned_model: variant_config}}}
}

print(toml.dumps(full_variant_config))

# %% [markdown]
# You're all set!
#
# You can change the weight to enable a gradual rollout of the new model.
#
# You might also add other parameters (e.g. `max_tokens`, `temperature`) to the variant section in the config file.
#
