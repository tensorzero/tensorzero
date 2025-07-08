# %%
# type: ignore

# %% [markdown]
# # Fireworks Supervised Fine-Tuning
#
# This recipe allows TensorZero users to fine-tune open-source LLMs using their own data.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to fine-tune a model using your own data and any prompt you want.
# We follow the Fireworks [docs](https://docs.fireworks.ai/fine-tuning/fine-tuning-models) on fine-tuning a model.
#

# %% [markdown]
# To get started:
#
# - Set the `TENSORZERO_CLICKHOUSE_URL` environment variable. For example: `TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero"`
# - You'll also need to [install](https://docs.fireworks.ai/tools-sdks/firectl/firectl) the CLI tool `firectl` on your machine and sign in with `firectl signin`. You can test that this all worked with `firectl whoami`.
# - Update the following parameters:
#

# %%
import os

CLICKHOUSE_URL = os.getenv("TENSORZERO_CLICKHOUSE_URL")

assert CLICKHOUSE_URL is not None, "TENSORZERO_CLICKHOUSE_URL is not set"

# %%
CONFIG_PATH = "../../../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

# The name of the variant to use to grab the templates used for fine-tuning
TEMPLATE_VARIANT_NAME = "gpt_4o_mini"  # It's OK that this variant uses a different model than the one we're fine-tuning

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
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional

import toml
from clickhouse_connect import get_client
from IPython.display import clear_output
from minijinja import Environment
from tensorzero.util import uuid7

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
clickhouse_client = get_client(dsn=CLICKHOUSE_URL)

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
# Query the inferences and demonstrations from ClickHouse.
#

# %%
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
        inference_id,
        value,
        ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn
    FROM
        DemonstrationFeedback
    ) f ON i.id = f.inference_id AND f.rn = 1
WHERE
    i.function_name = %(function_name)s
LIMIT %(max_samples)s
"""

params = {
    "function_name": FUNCTION_NAME,
    "max_samples": MAX_SAMPLES,
}

df = clickhouse_client.query_df(query, params)

df.sample(5)[["output", "value"]]


# %% [markdown]
# Render the inputs using the templates.
#


# %%
def warning_message(role: str) -> str:
    return (
        f"Fireworks does not support multiple content blocks per message. "
        f"We have chosen to concatenate the text across all content blocks for the message with role '{role}'. "
        f"You may want to manually review this behavior."
    )


def render_message(message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    role = message["role"]
    assert role in ["user", "assistant"], f"Invalid role: {role}"
    content: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    rendered_messages: List[Dict[str, Any]] = []

    for content_block in message["content"]:
        if content_block["type"] not in ["text", "raw_text"] and DROP_INVALID_MESSAGES:
            warnings.warn(
                f"Fireworks may not support content block type: {content_block['type']}, dropping example.",
                UserWarning,
            )
            return None
        if content_block["type"] == "text":
            parsed_content = content_block["value"]
            if not isinstance(parsed_content, str):
                parsed_content = env.render_template(role, **parsed_content)
            content.append({"type": "text", "text": parsed_content})
        elif content_block["type"] == "raw_text":
            content.append({"type": "text", "text": content_block["value"]})
        elif content_block["type"] == "thought":
            content.append(
                {"type": "text", "text": f"<think>{content_block['text']}</think>"}
            )
        elif (
            content_block["type"] == "tool_call"
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
            content_block["type"] == "tool_result"
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


def render_output(
    output: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Parses the assistant message from an observation using the provided function configuration.
    """
    content: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    if function_type == "json":
        return {"role": "assistant", "content": output["raw"]}
    elif function_type == "chat":
        for content_block in output:
            if content_block["type"] != "text" and DROP_INVALID_MESSAGES:
                warnings.warn(
                    f"Fireworks may not support content block type: {content_block['type']}, dropping example.",
                    UserWarning,
                )
                return None
            if content_block["type"] == "text":
                content.append({"type": "text", "text": content_block["text"]})
            elif content_block["type"] == "thought":
                content.append(
                    {"type": "text", "text": f"<think>{content_block['text']}</think>"}
                )
            elif content_block["type"] == "tool_call" and not DROP_INVALID_MESSAGES:
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
                    f"We do not support content block type: {content_block['type']}, dropping example.",
                    UserWarning,
                )
                return None
    else:
        raise ValueError(f"Unsupported function type: {function_type}")

    # Once we finish collecting all blocks, create one assistant message.
    output_message: Dict[str, Any] = {"role": "assistant"}
    if content:
        if len(content) > 1:
            warnings.warn(warning_message("assistant"), UserWarning)
        output_message["content"] = "\n".join([c["text"] for c in content])
    if tool_calls:
        output_message["tool_calls"] = tool_calls

    return output_message


def sample_to_conversational_messages(sample) -> List[Dict[str, Any]]:
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
        rendered_message = render_message(message)
        if rendered_message is None:
            # `render_message` will return None if the message contains an unknown or unsupported content block.
            # The entire example is dropped if this is the case.
            return None
        rendered_messages.extend(rendered_message)

    # Add the output to the messages
    output = json.loads(sample["value"])
    rendered_output = render_output(output)
    if rendered_output is None:
        # `render_output` will return None if the output contains an unknown or unsupported content block.
        # The entire example is dropped if this is the case.
        return None
    rendered_messages.append(rendered_output)

    return {"messages": rendered_messages}


df["conversational_messages"] = df.apply(sample_to_conversational_messages, axis=1)

# Drop null rows
df = df[df["conversational_messages"].notna()]

df.head()

# %%
for _, row in df.iterrows():
    print(row["value"])
    print(row["conversational_messages"])
    break

# %% [markdown]
# We'll write the conversational messages to a temporary file for the Fireworks CLI
#

# %%
dataset_id = f"t0-{uuid7()}"

with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
    for _, row in df.iterrows():
        f.write((json.dumps(row["conversational_messages"]) + "\n").encode("utf-8"))

    dataset_path = f.name
    result = subprocess.run(
        ["firectl", "create", "dataset", dataset_id, dataset_path], capture_output=True
    )
print(result.stdout)

# %%
result = subprocess.run(["firectl", "get", "dataset", dataset_id], capture_output=True)
print(result.stdout.decode("utf-8"))


# %%
def get_job_id(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.strip().startswith("Name:"):
            return line.split("/")[-1].strip()
    raise ValueError("Job ID not found in output")


# %% [markdown]
# Now we start the fine-tuning job. This cell will block until the job is done.
#

# %%
command = [
    "firectl",
    "create",
    "sftj",
    "--display-name",
    f"tensorzero-ft-job-{dataset_id}",
    "--dataset",
    dataset_id,
    "--base-model",
    MODEL_NAME,
]

if NUM_EPOCHS is not None:
    command.append("--epochs")
    command.append(str(NUM_EPOCHS))

print("Command: ", " ".join(command))

result = subprocess.run(command, capture_output=True)

if result.returncode != 0:
    print(result.stderr.decode("utf-8"))
else:
    stdout = result.stdout.decode("utf-8")
    print(stdout)
    job_id = get_job_id(stdout)
    print(f"job_id: {job_id}")

# %%
while True:
    clear_output(wait=True)

    try:
        command = ["firectl", "get", "sftj", job_id]
        result = subprocess.run(command, capture_output=True)
        stdout = result.stdout.decode("utf-8")
        print(stdout)
    except Exception as e:
        print(f"Error: {e}")

    if "State: JOB_STATE_FAILED" in stdout:
        raise ValueError("Fine-tuning job failed")

    if "State: JOB_STATE_COMPLETED" in stdout:
        break

    sleep(5)


# %%
def get_model_id(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.strip().startswith("Output Model:"):
            return line.split(":")[1].strip()
    raise ValueError("Model ID not found in output")


model_id = get_model_id(stdout)

assert model_id

model_id

# %% [markdown]
# Now that the model is done training, we need to [deploy](https://docs.fireworks.ai/fine-tuning/fine-tuning-models#deploying-and-using-a-model) it to Fireworks serverless inference. If you need high or guaranteed throughput you can also deploy the model to [reserved capacity](https://docs.fireworks.ai/deployments/reservations) or an on-demand [deployment](https://docs.fireworks.ai/guides/ondemand-deployments).
#

# %%
command = ["firectl", "deploy", model_id]
print(" ".join(command))
result = subprocess.run(command, capture_output=True)
if result.returncode != 0:
    print(result.stderr.decode("utf-8"))
else:
    stdout = result.stdout.decode("utf-8")
    print(stdout)


# %%
def get_model_identifier(model_id: str) -> str:
    command = ["firectl", "get", "model", model_id]
    result = subprocess.run(command, capture_output=True)
    stdout = result.stdout.decode("utf-8")
    for line in stdout.splitlines():
        if line.strip().startswith("Name:"):
            return line.split(":")[1].strip()
    raise ValueError("Model identifier not found in output")


model_identifier = get_model_identifier(model_id)

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

# %%
variant_config = {
    "type": "chat_completion",
    "weight": 0,
    "model": model_identifier,
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
    "functions": {FUNCTION_NAME: {"variants": {model_identifier: variant_config}}}
}

print(toml.dumps(full_variant_config))

# %% [markdown]
# You're all set!
#
# You can change the weight to enable a gradual rollout of the new model.
#
# You might also add other parameters (e.g. `max_tokens`, `temperature`) to the variant section in the config file.
#
