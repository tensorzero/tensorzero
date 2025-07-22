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

from dotenv import load_dotenv

load_dotenv()

CLICKHOUSE_URL = os.getenv("TENSORZERO_CLICKHOUSE_URL")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
account_id = os.getenv("FIREWORKS_ACCOUNT_ID")

assert CLICKHOUSE_URL is not None, "TENSORZERO_CLICKHOUSE_URL is not set"
assert FIREWORKS_API_KEY is not None, "FIREWORKS_API_KEY is not set"
assert account_id is not None, "FIREWORKS_ACCOUNT_ID is not set"

# %%
CONFIG_PATH = "../../../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

METRIC_NAME = "exact_match"

# The name of the variant to use to grab the templates used for fine-tuning
TEMPLATE_VARIANT_NAME = "gpt_4o_mini"  # It's OK that this variant uses a different model than the one we're fine-tuning

# If the metric is a float metric, you can set the threshold to filter the data
FLOAT_METRIC_THRESHOLD = 0.5

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
import json
import tempfile
import warnings
from time import sleep
from typing import Any, Dict, List, Optional

import requests
import toml
from IPython.display import clear_output
from tensorzero import (
    FloatMetricFilter,
    RenderedSample,
    TensorZeroGateway,
)
from tensorzero.util import uuid7

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
filters = FloatMetricFilter(
    metric_name=METRIC_NAME, value=FLOAT_METRIC_THRESHOLD, comparison_operator=">"
)
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
# Convert the rendered samples to the format Fireworks expects. This is handled automatically with our built-in `experimental_launch_optimization` method but we do it explicitly here.
#


# %%
def warning_message(role: str) -> str:
    return (
        f"Fireworks does not support multiple content blocks per message. "
        f"We have chosen to concatenate the text across all content blocks for the message with role '{role}'. "
        f"You may want to manually review this behavior."
    )


def render_message(message) -> Optional[List[Dict[str, Any]]]:
    role = message.role
    assert role in ["user", "assistant"], f"Invalid role: {role}"
    content: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    rendered_messages: List[Dict[str, Any]] = []

    for content_block in message.content:
        if content_block.type not in ["text", "raw_text"] and DROP_INVALID_MESSAGES:
            warnings.warn(
                f"Fireworks may not support content block type: {content_block['type']}, dropping example.",
                UserWarning,
            )
            return None
        if content_block.type == "text":
            parsed_content = content_block.text
            content.append({"type": "text", "text": parsed_content})
        elif content_block.type == "raw_text":
            content.append({"type": "text", "text": content_block.value})
        elif content_block.type == "thought":
            content.append(
                {"type": "text", "text": f"<think>{content_block.text}</think>"}
            )
        elif (
            content_block.type == "tool_call"
            and role == "assistant"
            and not DROP_INVALID_MESSAGES
        ):
            warnings.warn(
                "Fireworks may not support tool calls in assistant messages. Including it may cause the fine-tuning job to fail.",
                UserWarning,
            )
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
        elif (
            content_block.type == "tool_result"
            and role == "user"
            and not DROP_INVALID_MESSAGES
        ):
            warnings.warn(
                "Fireworks may not support tool results in user messages. Including it may cause the fine-tuning job to fail.",
                UserWarning,
            )
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
            if len(content) > 1:
                warnings.warn(warning_message(role), UserWarning)
            role_message["content"] = "\n".join([c["text"] for c in content])
        if tool_calls:
            role_message["tool_calls"] = tool_calls
        rendered_messages.append(role_message)

    return rendered_messages


def render_output(output) -> Optional[Dict[str, Any]]:
    """
    Parses the assistant message from an observation using the provided function configuration.
    """
    content: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []

    for content_block in output:
        if content_block.type != "text" and DROP_INVALID_MESSAGES:
            warnings.warn(
                f"Fireworks may not support content block type: {content_block['type']}, dropping example.",
                UserWarning,
            )
            return None
        if content_block.type == "text":
            content.append({"type": "text", "text": content_block.text})
        elif content_block.type == "thought":
            content.append(
                {"type": "text", "text": f"<think>{content_block.text}</think>"}
            )
        elif content_block.type == "tool_call" and not DROP_INVALID_MESSAGES:
            warnings.warn(
                "Fireworks may not support tool calls in assistant messages. Including it may cause the fine-tuning job to fail.",
                UserWarning,
            )
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
                f"We do not support content block type: {content_block.type}, dropping example.",
                UserWarning,
            )
            return None

    # Once we finish collecting all blocks, create one assistant message.
    output_message: Dict[str, Any] = {"role": "assistant"}
    if content:
        if len(content) > 1:
            warnings.warn(warning_message("assistant"), UserWarning)
        output_message["content"] = "\n".join([c["text"] for c in content])
    if tool_calls:
        output_message["tool_calls"] = tool_calls

    return output_message


def rendered_sample_to_fireworks(sample: RenderedSample) -> List[Dict[str, Any]]:
    function_input = sample.input

    rendered_messages = []

    # Add the system message to the rendered messages
    # If there is data passed in or a system template there must be a system message
    system = function_input.system
    if system:
        rendered_messages.append({"role": "system", "content": system})

    # Add the input messages to the rendered messages
    for message in function_input.messages:
        rendered_message = render_message(message)
        if rendered_message is None:
            # `render_message` will return None if the message contains an unknown or unsupported content block.
            # The entire example is dropped if this is the case.
            return None
        rendered_messages.extend(rendered_message)

    # Add the output to the messages
    rendered_output = render_output(sample.output)
    if rendered_output is None:
        # `render_output` will return None if the output contains an unknown or unsupported content block.
        # The entire example is dropped if this is the case.
        return None
    rendered_messages.append(rendered_output)

    return rendered_messages


# %%
fireworks_samples = []
for sample in rendered_samples:
    rendered_sample = rendered_sample_to_fireworks(sample)
    if rendered_sample is not None:
        fireworks_samples.append(rendered_sample)

print(f"Found {len(fireworks_samples)} samples to fine-tune on")

# %% [markdown]
# We'll write the conversational messages to a temporary file for the Fireworks API
#

# %%
dataset_id = f"t0-{uuid7()}"
api_base = "https://api.fireworks.ai/v1"
base_headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}"}
json_headers = base_headers.copy()
json_headers.update({"Content-Type": "application/json"})

with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
    for row in fireworks_samples:
        f.write((json.dumps(row) + "\n").encode("utf-8"))
    create_record_url = f"{api_base}/accounts/{account_id}/datasets/"

    # Create dataset
    create_record_result = requests.post(
        create_record_url,
        json={
            "datasetId": dataset_id,
            "dataset": {
                "displayName": dataset_id,
                "format": "CHAT",
                "exampleCount": len(fireworks_samples),
            },
        },
        headers=json_headers,
    )
    print(create_record_result)
    # Upload dataset
    upload_file_url = f"{api_base}/accounts/{account_id}/datasets/{dataset_id}:upload"

    with open(f.name, "r") as file:
        dataset = {"file": file}
        upload_file_result = requests.post(
            upload_file_url, headers=base_headers, files=dataset
        )
        print(upload_file_result)

# %%
check_state_url = f"{api_base}/accounts/{account_id}/datasets/{dataset_id}"
result = requests.get(check_state_url, headers=base_headers)
print(json.dumps(result.json(), indent=2))

# %% [markdown]
# Now we start the fine-tuning job. This cell will block until the job is done.
#

# %%
create_sft_url = f"{api_base}/accounts/{account_id}/supervisedFineTuningJobs"
json_to_send = {
    "dataset": f"accounts/{account_id}/datasets/{dataset_id}",
    "base_model": MODEL_NAME,
}
if NUM_EPOCHS is not None:
    json_to_send["epochs"] = NUM_EPOCHS
result = requests.post(url=create_sft_url, headers=json_headers, json=json_to_send)
if result.status_code != 200:
    print(json.dumps(result.json(), indent=2))
else:
    response = result.json()
    print(json.dumps(response, indent=2))
    job_id = response["name"]
    print(f"job_id: {job_id}")

# %%
get_job_status_url = f"{api_base}/{job_id}"
while True:
    clear_output(wait=True)

    try:
        result = requests.get(
            url=get_job_status_url,
            headers=base_headers,
        )
        response = result.json()
        print(json.dumps(result.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

    if response["state"] == "JOB_STATE_FAILED":
        raise ValueError("Fine-tuning job failed")

    if response["state"] == "JOB_STATE_COMPLETED":
        break

    sleep(5)

# %% [markdown]
# Now that the model is done training, we need to [deploy](https://docs.fireworks.ai/fine-tuning/fine-tuning-models#deploying-and-using-a-model) it to Fireworks serverless inference. If you need high or guaranteed throughput you can also deploy the model to [reserved capacity](https://docs.fireworks.ai/deployments/reservations) or an on-demand [deployment](https://docs.fireworks.ai/guides/ondemand-deployments).
#

# %%
model_id = response["outputModel"]
deploy_model_url = f"{api_base}/accounts/{account_id}/deployedModels"
result = requests.post(
    url=deploy_model_url,
    headers=json_headers,
    json={
        "model": model_id,
        "default": True,
        "serverless": True,
        "public": False,
    },
)

# %%
model_identifier = model_id

assert model_identifier

model_identifier

# %% [markdown]
# Once the fine-tuning job is complete, you can add the fine-tuned model to your config file.
#

# %%
model_config = {
    "models": {
        model_identifier: {
            "routing": ["fireworks"],
            "providers": {
                "fireworks": {"type": "fireworks", "model_name": model_identifier}
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
# You might also add other parameters (e.g. `max_tokens`, `temperature`) to the variant section in the config file.
#
