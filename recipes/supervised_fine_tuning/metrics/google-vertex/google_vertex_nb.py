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
# - Create local authentication credentials `gcloud auth application-default login`
# - You may need to [Create a Bucket](https://cloud.google.com/storage/docs/creating-buckets) on GCP, if you do not already have one.
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
PROJECT_ID = "alpine-realm-415615"
LOCATION = "us-central1"
BUCKET_NAME = "tensorzero-fine-tuning"

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
from google.cloud import storage
from google.cloud.aiplatform_v1.types import JobState
from IPython.display import clear_output
from tensorzero import (
    BooleanMetricNode,
    FloatMetricNode,
    RawText,
    TensorZeroGateway,
    Text,
    Thought,
    ToolCall,
    ToolResult,
)
from tensorzero.util import uuid7
from vertexai.tuning import sft

# %% [markdown]
# Initialize Vertex AI

# %%
vertexai.init(project=PROJECT_ID, location=LOCATION)

# %% [markdown]
# Initialize the TensorZero client

# %%
tensorzero_client = TensorZeroGateway.build_embedded(
    config_file=CONFIG_PATH,
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
    timeout=15,
)

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
# Valudate config

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

variant = config["functions"][FUNCTION_NAME]["variants"][TEMPLATE_VARIANT_NAME]

variant

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
# Set the metric filter

# %%
assert "optimize" in metric, "Metric is missing the `optimize` field"

if metric.get("type") == "float":
    comparison_operator = ">=" if metric["optimize"] == "max" else "<="
    metric_node = FloatMetricNode(
        metric_name=METRIC_NAME,
        value=FLOAT_METRIC_THRESHOLD,
        comparison_operator=comparison_operator,
    )
elif metric.get("type") == "boolean":
    metric_node = BooleanMetricNode(
        metric_name=METRIC_NAME,
        value=True if metric["optimize"] == "max" else False,
    )

metric_node

# %% [markdown]
# Query the inferences and feedback from ClickHouse.

# %%
stored_inferences = tensorzero_client.experimental_list_inferences(
    function_name=FUNCTION_NAME,
    variant_name=None,
    filters=metric_node,
    limit=MAX_SAMPLES,
)

# %% [markdown]
# Render the stored inferences

# %%
rendered_inferences = tensorzero_client.experimental_render_inferences(
    stored_inferences=stored_inferences,
    variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME},
)

# %% [markdown]
# Convert inferences to vertex format

# %%
role_map = {
    "user": "user",
    "assistant": "model",
    "system": "system",
}


def merge_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive messages with the same role into a single message.
    """
    merged: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        parts = msg.get("parts", [])
        if merged and merged[-1]["role"] == role:
            merged[-1]["parts"].extend(parts)
        else:
            merged.append({"role": role, "parts": list(parts)})
    return merged


def render_chat_message(
    role: str,
    content_blocks: List[
        Any
    ],  # instances of Text, RawText, Thought, ToolCall, ToolResult
) -> Optional[Dict[str, Any]]:
    """
    Render a single chat message into Google “parts” format.
    """
    parts: List[Dict[str, Any]] = []
    for blk in content_blocks:
        # plain text
        if isinstance(blk, Text):
            parts.append({"text": blk.text})
        elif isinstance(blk, RawText):  # Verify if needed
            parts.append({"text": blk.value})
        # internal “thoughts”
        elif isinstance(blk, Thought):
            parts.append({"text": f"<think>{blk.text}</think>"})
        # function call (assistant only)
        elif isinstance(blk, ToolCall) and role == "assistant":
            args = blk.raw_arguments
            # raw_arguments might already be a dict or JSON string
            if isinstance(args, str):
                args = json.loads(args)
            parts.append(
                {
                    "functionCall": {
                        "name": blk.name,
                        "args": args,
                    }
                }
            )
        # function result (user only)
        elif isinstance(blk, ToolResult) and role == "user":
            parts.append(
                {
                    "functionResponse": {
                        "name": blk.name,
                        "response": {"result": blk.result},
                    }
                }
            )
        else:
            warnings.warn(
                f"Unsupported block type {type(blk)} in role={role}, skipping inference.",
                UserWarning,
            )
            return None
    return {"role": role_map[role], "parts": parts}


def inference_to_google(
    inf,
) -> Optional[Dict[str, Any]]:
    """
    Convert a single rendered_inference into the Google Vertex format dict.
    """
    model_input = inf.input
    rendered_msgs: List[Dict[str, Any]] = []

    # 1) systemInstruction
    if model_input.system:
        system_instruction = {
            "role": role_map["system"],
            "parts": [{"text": model_input.system}],
        }
    else:
        system_instruction = None

    # 2) all user/assistant messages
    for msg in model_input.messages:
        rendered = render_chat_message(msg.role, msg.content)
        if rendered is None:
            return None
        rendered_msgs.append(rendered)

    # 3) the assistant’s output
    #    (same logic as render_chat_message but without ToolResult)
    out_parts: List[Dict[str, Any]] = []
    for blk in inf.output:
        if isinstance(blk, Text):
            out_parts.append({"text": blk.text})
        elif isinstance(blk, Thought):
            out_parts.append({"text": f"<think>{blk.text}</think>"})
        elif isinstance(blk, ToolCall):
            args = blk.raw_arguments
            if isinstance(args, str):
                args = json.loads(args)
            out_parts.append(
                {
                    "functionCall": {
                        "name": blk.name,
                        "args": args,
                    }
                }
            )
        else:
            warnings.warn(
                f"Unsupported output block {type(blk)}, skipping inference.",
                UserWarning,
            )
            return None
    rendered_msgs.append({"role": role_map["assistant"], "parts": out_parts})

    # 4) merge any consecutive roles and return
    contents = merge_messages(rendered_msgs)
    result = {"google_messages": {"contents": contents}}
    if system_instruction:
        result["google_messages"]["systemInstruction"] = system_instruction
    # optionally keep track of episode
    if hasattr(inf, "episode_id"):
        result["episode_id"] = inf.episode_id
    return result


google_payloads = []
for inf in rendered_inferences:
    payload = inference_to_google(inf)
    if payload is not None:
        google_payloads.append(payload)

df = pd.DataFrame(google_payloads)
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
            JobState.JOB_STATE_SUCCEEDED.value,
            JobState.JOB_STATE_FAILED.value,
            JobState.JOB_STATE_CANCELLED.value,
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
                    "endpoint_id": fine_tuned_model,
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
