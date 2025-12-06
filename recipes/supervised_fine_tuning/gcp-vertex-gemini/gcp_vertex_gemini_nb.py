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
# - Set the `GCP_VERTEX_CREDENTIALS_PATH`, `GCP_PROJECT_ID`, `GCP_LOCATION`, and `GCP_BUCKET_NAME` environment variables.
# - Create local authentication credentials `gcloud auth application-default login`
# - You may need to [Create a Bucket](https://cloud.google.com/storage/docs/creating-buckets) on GCP, if you do not already have one.
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

# The name of the model to fine-tune (supported models: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning)
MODEL_NAME = "gemini-2.0-flash-lite-001"

# %%
import os
import sys

tensorzero_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
if tensorzero_path not in sys.path:
    sys.path.append(tensorzero_path)

# %%
import json
import tempfile
import time
import warnings
from typing import Any, Dict, List, Optional

import toml
import vertexai
from google.cloud import storage
from google.cloud.aiplatform_v1.types import JobState
from IPython.display import clear_output
from tensorzero import (
    FloatMetricFilter,
    RawText,
    TensorZeroGateway,
    Text,
    Thought,
    ToolCall,
    ToolResult,
)
from tensorzero.util import uuid7
from vertexai.tuning import sft

from recipes.util import train_val_split

# %% [markdown]
# Initialize Vertex AI
#

# %%
vertexai.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["GCP_LOCATION"])

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
# Set the metric filter
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
# Query the inferences from ClickHouse.
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
# Render the stored inferences
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
# Convert inferences to vertex format
#

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
    content_blocks: List[Any],  # instances of Text, RawText, Thought, ToolCall, ToolResult
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
    result = {"contents": contents}
    if system_instruction:
        result.update({"systemInstruction": system_instruction})
    return result


# %%
train_data = [inference_to_google(sample) for sample in train_samples]
val_data = [inference_to_google(sample) for sample in val_samples]


# %% [markdown]
# Upload the training and validation datasets to GCP
#


# %%
def upload_dataset_to_gcp(data: List[Dict[str, Any]], dataset_name: str, gcp_client: storage.Client) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write the openai_messages to the temporary file
        for item in data:
            json.dump(item, f)
            f.write("\n")
        f.flush()

        bucket = gcp_client.bucket(os.environ["GCP_BUCKET_NAME"])
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
        blob.upload_from_filename(f.name, if_generation_match=generation_match_precondition)


gcp_client = storage.Client(project=os.environ["GCP_PROJECT_ID"])

train_file_name = f"train_{uuid7()}.jsonl"
val_file_name = f"val_{uuid7()}.jsonl"


upload_dataset_to_gcp(train_data, train_file_name, gcp_client)
upload_dataset_to_gcp(val_data, val_file_name, gcp_client)

# %% [markdown]
# Launch the fine-tuning job.
#

# %%
sft_tuning_job = sft.train(
    source_model=MODEL_NAME,
    train_dataset=f"gs://{os.environ['GCP_BUCKET_NAME']}/{train_file_name}",
    validation_dataset=f"gs://{os.environ['GCP_BUCKET_NAME']}/{val_file_name}",
)

# %% [markdown]
# Wait for the fine-tuning job to complete.
#
# This cell will take a while to run.
#

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
                    "location": os.environ["GCP_LOCATION"],
                    "project_id": os.environ["GCP_PROJECT_ID"],
                }
            },
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
