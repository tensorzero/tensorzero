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

# Maximum number of tokens per example to OpenAI
MAX_TOKENS_PER_EXAMPLE = 65_536

# The name of the model to fine-tune (supported models: https://platform.openai.com/docs/guides/fine-tuning)
MODEL_NAME = "gpt-4o-mini-2024-07-18"

# %%
import json
import os
import tempfile
import time
import warnings
from pprint import pprint
from typing import Any, Dict, List, Optional

import openai
import toml
from IPython.display import clear_output
from tensorzero import (
    FloatMetricFilter,
    TensorZeroGateway,
)

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
rendered_inferences = tensorzero_client.experimental_render_inferences(
    stored_inferences=stored_inferences,
    variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME},
)


# %%
def render_message(message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    role = message["role"]
    assert role in ["user", "assistant"], f"Invalid role: {role}"
    content: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    rendered_messages: List[Dict[str, Any]] = []

    for content_block in message["content"]:
        if content_block["type"] == "text":
            parsed_content = content_block["value"]
            content.append({"type": "text", "text": parsed_content})
        elif content_block["type"] == "raw_text":
            content.append({"type": "text", "text": content_block["value"]})
        elif content_block["type"] == "thought":
            content.append(
                {"type": "text", "text": f"<think>{content_block['text']}</think>"}
            )
        elif content_block["type"] == "tool_call" and role == "assistant":
            tool_calls.append(
                {
                    "function": {
                        "arguments": json.dumps(content_block["arguments"]),
                        "name": content_block["name"],
                    },
                    "id": content_block["id"],
                    "type": "function",
                }
            )
        elif content_block["type"] == "tool_result" and role == "user":
            # Tool results get priority so that they follow the tool call in the conversation.
            # Any other "user" content will be appended in another message below.
            rendered_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": content_block["id"],
                    "content": content_block["result"],
                }
            )
        else:
            warnings.warn(
                f"We do not support content block type: {content_block['type']}, dropping example.",
                UserWarning,
            )
            return None

    if content or tool_calls:
        role_message: Dict[str, Any] = {"role": role}
        if content:
            role_message["content"] = content
        if tool_calls:
            role_message["tool_calls"] = tool_calls
        rendered_messages.append(role_message)

    return rendered_messages


def render_output(
    output: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Parses the assistant message from an observation using the provided function configuration.
    """
    content: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for content_block in output:
        if content_block["type"] == "text":
            content.append({"type": "text", "text": content_block["text"]})
        elif content_block["type"] == "thought":
            content.append(
                {"type": "text", "text": f"<think>{content_block['text']}</think>"}
            )
        elif content_block["type"] == "tool_call":
            tool_calls.append(
                {
                    "function": {
                        "arguments": json.dumps(content_block["arguments"]),
                        "name": content_block["name"],
                    },
                    "id": content_block["id"],
                    "type": "function",
                }
            )
        else:
            warnings.warn(
                f"We do not support content block type: {content_block['type']}, dropping example.",
                UserWarning,
            )
            return None

    # Once we finish collecting all blocks, create one assistant message.
    output_message: Dict[str, Any] = {"role": "assistant"}
    if content:
        output_message["content"] = content
    if tool_calls:
        output_message["tool_calls"] = tool_calls

    return output_message


def sample_to_openai_messages(sample) -> List[Dict[str, Any]]:
    rendered_messages = []

    # Add the system message to the rendered messages
    # If there is data passed in or a system template there must be a system message
    system = sample.system
    if system:
        rendered_messages.append({"role": "system", "content": system})

    # Add the input messages to the rendered messages
    for message in sample.messages:
        rendered_message = render_message(message)
        if rendered_message is None:
            # `render_message` will return None if the message contains an unknown or unsupported content block.
            # The entire example is dropped if this is the case.
            return None
        rendered_messages.extend(rendered_message)

    # Add the output to the messages
    output = sample.output
    rendered_output = render_output(output)
    if rendered_output is None:
        # `render_output` will return None if the output contains an unknown or unsupported content block.
        # The entire example is dropped if this is the case.
        return None
    rendered_messages.append(rendered_output)

    return {"messages": rendered_messages}


# %%
openai_samples = [sample_to_openai_messages(sample) for sample in stored_inferences]

# %% [markdown]
# Split the data into training and validation sets for fine-tuning.
#

# %%
# Get unique episode_ids
episode_ids = list(set(sample.episode_id for sample in stored_inferences))
split_index = int(len(episode_ids) * VAL_FRACTION)
train_episode_ids = episode_ids[:split_index]
val_episode_ids = episode_ids[split_index:]

train_samples = [
    sample for sample in stored_inferences if sample.episode_id in train_episode_ids
]
val_samples = [
    sample for sample in stored_inferences if sample.episode_id in val_episode_ids
]


print(f"Training set size: {len(train_samples)}")
print(f"Validation set size: {len(val_samples)}")
print(f"Actual validation fraction: {len(val_samples) / len(stored_inferences):.2f}")


# %% [markdown]
# Upload the training and validation datasets to OpenAI.
#


# %%
def upload_dataset_to_openai(samples, openai_client: openai.OpenAI) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write the openai_messages to the temporary file
        for item in samples:
            json.dump(item, f)
            f.write("\n")
        f.flush()

        # Upload the file to OpenAI
        with open(f.name, "rb") as file:
            file_object = openai_client.files.create(file=file, purpose="fine-tune")

        return file_object.id


openai_client = openai.OpenAI()

train_file_object_id = upload_dataset_to_openai(train_samples, openai_client)
val_file_object_id = upload_dataset_to_openai(val_samples, openai_client)

# %% [markdown]
# Launch the fine-tuning job.
#

# %%
fine_tuning_job = openai_client.fine_tuning.jobs.create(
    training_file=train_file_object_id,
    validation_file=val_file_object_id,
    model=MODEL_NAME,
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
        job_status = openai_client.fine_tuning.jobs.retrieve(fine_tuning_job.id)
        pprint(job_status.to_dict())
        if job_status.status in ("succeeded", "failed", "cancelled"):
            break
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(10)

# %% [markdown]
# Once the fine-tuning job is complete, you can add the fine-tuned model to your config file.
#

# %%
fine_tuned_model = job_status.fine_tuned_model
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
# You might also add other parameters (e.g. `MAX_TOKENS_PER_EXAMPLE`, `temperature`) to the variant section in the config file.
#
