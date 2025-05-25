# %%
# type: ignore

# %% [markdown]
# # Google Vertex Supervised Fine-Tuning
#
# This recipe allows TensorZero users to fine-tune Gemini models using their own data.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to fine-tune a model using your own data and any prompt you want.
#

# %% [markdown]
# To get started:
#
# - Set the `TENSORZERO_CLICKHOUSE_URL` environment variable. For example: `TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero"`
# - Update the following parameters:
#

# %%
CONFIG_PATH = "../../../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

METRIC_NAME = "exact_match"

# The name of the variant to use to grab the templates used for fine-tuning
TEMPLATE_VARIANT_NAME = "gpt_4o_mini"

# If the metric is a float metric, you can set the threshold to filter the data
FLOAT_METRIC_THRESHOLD = 0.5

# Fraction of the data to use for validation
VAL_FRACTION = 0.2

# Maximum number of samples to use for fine-tuning
MAX_SAMPLES = 100_000

# The name of the model to fine-tune (supported models: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning)
MODEL_NAME = "gemini-2.0-flash-lite-001"

# Google Cloud Variables
PROJECT_ID = "<your-project-id>"
LOCATION = "us-central1"
BUCKET_NAME = "tensorzero-supervised-fine-tuning"

# %%
import json
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import toml
import vertexai
from clickhouse_connect import get_client
from google.cloud import storage
from google.cloud.aiplatform_v1.types import JobState
from IPython.display import clear_output
from minijinja import Environment
from tensorzero.util import uuid7
from vertexai.tuning import sft

# %% [markdown]
# Initialize Vertex AI

# %%
vertexai.init(project=PROJECT_ID, location=LOCATION)

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
role_map = {
    "user": "user",
    "assistant": "model",
    "system": "system",  # The role field of systemInstruction is ignored and doesn't affect the performance of the model.
}


def render_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    role = message["role"]
    assert role in ["user", "assistant"], f"Invalid role: {role}"
    content: List[Dict[str, Any]] = []

    for content_block in message["content"]:
        if content_block["type"] == "text":
            parsed_content = content_block["value"]
            if not isinstance(parsed_content, str):
                parsed_content = env.render_template(role, **parsed_content)
            content.append({"text": parsed_content})
        elif content_block["type"] == "raw_text":
            content.append({"text": content_block["value"]})
        elif content_block["type"] == "thought":
            content.append({"text": f"<think>{content_block['text']}</think>"})
        elif content_block["type"] == "tool_call" and role == "assistant":
            content.append(
                {
                    "functionCall": {
                        "name": content_block["name"],
                        "args": json.loads(content_block["arguments"]),
                    }
                }
            )
        elif content_block["type"] == "tool_result" and role == "user":
            content.append(
                {
                    "functionResponse": {
                        "name": content_block["name"],
                        "response": {"result": content_block["result"]},
                    }
                }
            )
        else:
            warnings.warn(
                f"We do not support content block type: {content_block['type']}, dropping example.",
                UserWarning,
            )
            return None

    return {
        "role": role_map[role],
        "parts": content,
    }


def render_output(
    output: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Parses the assistant message from an observation using the provided function configuration.
    """
    content: List[Dict[str, Any]] = []

    if function_type == "json":
        content.append({"text": output["raw"]})
    elif function_type == "chat":
        for content_block in output:
            if content_block["type"] == "text":
                content.append({"text": content_block["text"]})
            elif content_block["type"] == "thought":
                content.append({"text": f"<think>{content_block['text']}</think>"})
            elif content_block["type"] == "tool_call":
                content.append(
                    {
                        "functionCall": {
                            "name": content_block["name"],
                            "args": content_block["arguments"],
                        }
                    }
                )
            else:
                warnings.warn(
                    f"We do not support content block type: {content_block['type']}, dropping example.",
                    UserWarning,
                )
                return None
    else:
        raise ValueError(f"Unsupported function type: {function_type}")

    return {"role": role_map["assistant"], "parts": content}


def sample_to_google_messages(sample) -> List[Dict[str, Any]]:
    function_input = json.loads(sample["input"])

    rendered_messages = []

    # Add the system message to the rendered messages
    # If there is data passed in or a system template there must be a system message
    system = function_input.get("system", {})
    if len(system) > 0 or system_template_path:
        if system_template_path:
            system_message = {
                "role": role_map["system"],
                "parts": [{"text": env.render_template("system", **system)}],
            }
        else:
            system_message = {"role": role_map["system"], "parts": [{"text": system}]}

    # Add the input messages to the rendered messages
    for message in function_input["messages"]:
        rendered_message = render_message(message)
        if rendered_message is None:
            # `render_message` will return None if the message contains an unknown or unsupported content block.
            # The entire example is dropped if this is the case.
            return None
        rendered_messages.append(render_message(message))

    # Add the output to the messages
    output = json.loads(sample["output"])
    rendered_output = render_output(output)
    if rendered_output is None:
        # `render_output` will return None if the output contains an unknown or unsupported content block.
        # The entire example is dropped if this is the case.
        return None
    rendered_messages.append(rendered_output)

    return {
        "systemInstruction": system_message,
        "contents": rendered_messages,
    }


df["google_messages"] = df.apply(sample_to_google_messages, axis=1)

# Drop null rows
df = df[df["google_messages"].notna()]

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
# Create a google clous storage bucket

# %% [markdown]
# Upload the training and validation datasets to GCP

# %%
def upload_dataset_to_gcp(
    df: pd.DataFrame, dataset_name: str, gcp_client: storage.Client
) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write the openai_messages to the temporary file
        for item in df["google_messages"]:
            json.dump(item, f)
            f.write("\n")
        f.flush()

        bucket = gcp_client.bucket(BUCKET_NAME)
        if not bucket.exists():
            bucket.storage_class = "STANDARD"
            bucket = gcp_client.create_bucket(bucket, location="us")
            print(
                "Created bucket {} in {} with storage class {}".format(
                    bucket.name, bucket.location, bucket.storage_class
                )
            )
        blob = bucket.blob(dataset_name)

        generation_match_precondition = 0
        blob.upload_from_filename(
            f.name, if_generation_match=generation_match_precondition
        )


gcp_client = storage.Client(project=PROJECT_ID)

train_file_name = f"train_{uuid7()}.jsonl"
val_file_name = f"val_{uuid7()}.jsonl"


upload_dataset_to_gcp(train_df, train_file_name, gcp_client)
upload_dataset_to_gcp(val_df, val_file_name, gcp_client)

# %% [markdown]
# Launch the fine-tuning job.

# %%
sft_tuning_job = sft.train(
    source_model=MODEL_NAME,
    train_dataset=f"gs://{BUCKET_NAME}/{train_file_name}",
    validation_dataset=f"gs://{BUCKET_NAME}/{val_file_name}",
)

# %% [markdown]
# Wait for the fine-tuning job to complete.
#
# This cell will take a while to run.

# %%
response = sft.SupervisedTuningJob(sft_tuning_job.resource_name)
while True:
    clear_output(wait=True)

    try:
        job_state = response.state
        print(job_state)
        if job_state in (
            JobState.JOB_STATE_SUCCEEDED,
            JobState.JOB_STATE_FAILED,
            JobState.JOB_STATE_CANCELLED,
        ):
            break
    except Exception as e:
        print(f"Error: {e}")
    response.refresh()
    time.sleep(10)

# %% [markdown]
# Once the fine-tuning job is complete, you can add the fine-tuned model to your config file.
#

# %%
fine_tuned_model = response.tuned_model_endpoint_name.split("/")[-1]
model_config = {
    "models": {
        fine_tuned_model: {
            "routing": ["gcp_vertex_gemini"],
            "providers": {
                "gcp_vertex_gemini": {
                    "type": "gcp_vertex_gemini",
                    "model_name": fine_tuned_model,
                    "location": LOCATION,
                    "project_id": PROJECT_ID,
                }
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
