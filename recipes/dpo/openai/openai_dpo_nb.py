# type: ignore


# %% [markdown]
# # OpenAI Supervised Fine-Tuning using Direct Preference Optimization (DPO)
#
# This recipe allows TensorZero users to fine-tune OpenAI models using Direct Preference Optimization (DPO) and their own data. Since TensorZero automatically logs all inferences and feedback, it is straightforward to fine-tune a model using your own data and any prompt you want.
#

# %% [markdown]
# To get started:
#
# - Set the `TENSORZERO_CLICKHOUSE_URL` environment variable. For example: `TENSORZERO_CLICKHOUSE_URL`=`"http://chuser:chpassword@localhost:8123/tensorzero"`
# - Set the `OPENAI_API_KEY` environment variable.
# - Update the following parameters:
#

# %%
CONFIG_PATH = "../../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

# The name of the variant to use to grab the templates used for fine-tuning
TEMPLATE_VARIANT_NAME = "gpt_4o_mini"  # It's OK that this variant uses a different model than the one we're fine-tuning

# Fraction of the data to use for validation
VAL_FRACTION = 0.2

# Maximum number of samples to use for fine-tuning
MAX_SAMPLES = 1000

#  Model "gpt-4o-2024-08-06" is to our knowledge the only base model supported for this method.
#  You can can use the base model as below or fine-tunes derived from it for this recipe.
MODEL_NAME = "gpt-4o-2024-08-06"

# %%
import json
import os
import tempfile
import time
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

import numpy as np
import openai
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
# Ensure that the function and variant being fine-tuned are present in the provided config.
#

# %%
assert "functions" in config, "No `[functions]` section found in config"
assert "variants" in config["functions"][FUNCTION_NAME], (
    f"No variants section found for function `{FUNCTION_NAME}`"
)
assert TEMPLATE_VARIANT_NAME in config["functions"][FUNCTION_NAME]["variants"], (
    f"No variant named `{TEMPLATE_VARIANT_NAME}` found in function `{FUNCTION_NAME}`"
)

# %% [markdown]
# Retrieve the configuration for the variant with the templates we will use for fine-tuning.
#

# %%
function_type = config["functions"][FUNCTION_NAME]["type"]
variant = config["functions"][FUNCTION_NAME]["variants"][TEMPLATE_VARIANT_NAME]

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
inference_table_name = {"json": "JsonInference"}.get(function_type)

if inference_table_name is None:
    raise ValueError(f"Unsupported function type: {function_type}")

# %% [markdown]
# Query ClickHouse for inference, feedback, and metric.
#

# %%
query = f"""
SELECT
    i.variant_name AS variant,
    i.episode_id AS episode_id,
    i.input AS input,
    i.output AS non_preferred_output,
    d.value AS preferred_output
FROM
    {inference_table_name} AS i
INNER
    JOIN DemonstrationFeedback AS d ON i.id = d.inference_id
WHERE
    (i.function_name = %(function_name)s)
LIMIT %(max_samples)s
"""

params = {
    "max_samples": MAX_SAMPLES,
    "function_name": FUNCTION_NAME,
}

df = clickhouse_client.query_df(query, params)

df.head()


# %% [markdown]
# OpenAI requires the fine-tuning data (for DPO) to be structured in this [format](https://platform.openai.com/docs/guides/fine-tuning#preference)
#
# ```
# {
#   "input": {
#     "messages": [
#       {
#         "role": "user",
#         "content": "<string>"
#       }
#     ],
#     "tools": [],
#     "parallel_tool_calls": true
#   },
#   "preferred_output": [
#     {
#       "role": "assistant",
#       "content": "<string>"
#     }
#   ],
#   "non_preferred_output": [
#     {
#       "role": "assistant",
#       "content": "<string>"
#     }
#   ]
# }
#
# ```
#


# %%
def render_message(content: List[Dict[str, Any]], role: str) -> str:
    assert role in ["user", "assistant"], f"Invalid role: {role}"

    if len(content) != 1:
        raise ValueError(f"Message must have exactly one content block: {content}")

    if content[0]["type"] != "text":
        raise ValueError(f"Content block must be of the type text: {content}")

    content = content[0]["value"]

    if isinstance(content, str):
        return content
    else:
        return env.render_template(role, **content)


def render_output_message(output):
    if function_type == "chat":
        if len(output) != 1:
            raise ValueError(f"Output {output} must have exactly one content block")
        if output[0]["type"] != "text":
            raise ValueError(f"Output {output} must be a text block")
        return {"role": "assistant", "content": output[0]["text"]}
    elif function_type == "json":
        return {"role": "assistant", "content": output["raw"]}
    else:
        raise ValueError(f"Unsupported function type: {function_type}")


def sample_to_openai_messages(sample) -> List[Dict[str, str]]:
    function_input = json.loads(sample["input"])

    result = {
        "input": {"messages": [], "tools": [], "parallel_tool_calls": True},
        "preferred_output": [],
        "non_preferred_output": [],
    }

    # Add the system message to the rendered messages
    # If there is data passed in or a system template there must be a system message
    system = function_input.get("system", {})
    if len(system) > 0 or system_template_path:
        if system_template_path:
            system_message = env.render_template("system", **system)
            result["input"]["messages"].append(
                {"role": "system", "content": system_message}
            )
        else:
            result["input"]["messages"].append(
                {"role": "system", "content": system_message}
            )

    # Add the input messages to the rendered messages
    for message in function_input["messages"]:
        rendered_message = render_message(message["content"], message["role"])
        result["input"]["messages"].append(
            {"role": message["role"], "content": rendered_message}
        )

    # Add the demonstration (preferred output)
    preferred_output = json.loads(sample["preferred_output"])
    result["preferred_output"].append(render_output_message(preferred_output))

    # Add the inference output (non-preferred output)
    non_preferred_output = json.loads(sample["non_preferred_output"])
    result["non_preferred_output"].append(render_output_message(non_preferred_output))

    return result


df["openai_messages"] = df.apply(sample_to_openai_messages, axis=1)

df.head()

# %% [markdown]
# Split data into training and validation sets for fine-tuning
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
# Upload the prepared datasets to OpenAI.
#


# %%
def upload_dataset_to_openai(df, openai_client) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in df["openai_messages"]:
            json.dump(item, f)
            f.write("\n")
        f.flush()

        print(f"File persisted on path [{f.name}]")

        with open(f.name, "rb") as file:
            file_object = openai_client.files.create(file=file, purpose="fine-tune")

        return file_object.id


openai_client = openai.OpenAI()

dpo_fine_tuning_object_id = upload_dataset_to_openai(train_df, openai_client)
val_file_object_id = upload_dataset_to_openai(val_df, openai_client)

# %% [markdown]
# Launch the fine-tuning job and wait for it to complete.
#
# NOTE : This step takes a while and you can monitor the progress and estimated completion time using OpenAI's fine-tuning [dashboard](https://platform.openai.com/finetune/)
#

# %%
fine_tuning_job = openai_client.fine_tuning.jobs.create(
    training_file=dpo_fine_tuning_object_id,
    validation_file=val_file_object_id,
    model=MODEL_NAME,
    method={
        "type": "dpo",
        "dpo": {
            "hyperparameters": {"beta": 0.2},
        },
    },
)

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

print(f"The fine-tuning job has compeleted with result {job_status.status}")

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
# You might also add other parameters (e.g. max_tokens, temperature) to the variant section in the config file.
#
