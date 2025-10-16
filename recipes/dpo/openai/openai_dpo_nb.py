# %%
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
CONFIG_PATH = "../../../ui/fixtures/config/tensorzero.toml"

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
import random
import tempfile
import time
from pprint import pprint
from typing import Any, Dict, List

import openai
import toml
from IPython.display import clear_output
from tensorzero import ContentBlock, RenderedSample, TensorZeroGateway

# %%
assert "TENSORZERO_CLICKHOUSE_URL" in os.environ, "TENSORZERO_CLICKHOUSE_URL environment variable not set"

# %% [markdown]
# Initialize the TensorZero client
#

# %%
t0 = TensorZeroGateway.build_embedded(clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"], config_file=CONFIG_PATH)

# %%
inferences = t0.experimental_list_inferences(
    function_name=FUNCTION_NAME,
    output_source="demonstration",  # Since we're using DPO we need pairwise data so we must use demonstrations
    limit=MAX_SAMPLES,
)

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
rendered_samples = t0.experimental_render_samples(
    stored_samples=inferences, variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME}
)

# %% [markdown]
# Split data into training and validation sets for fine-tuning
#

# %%
random.shuffle(rendered_samples)
train_samples = rendered_samples[: int(len(rendered_samples) * (1 - VAL_FRACTION))]
val_samples = rendered_samples[int(len(rendered_samples) * (1 - VAL_FRACTION)) :]

print(f"Training set size: {len(train_samples)}")
print(f"Validation set size: {len(val_samples)}")
print(f"Actual validation fraction: {len(val_samples) / len(rendered_samples):.2f}")


# %%
def prepare_output(output: List[ContentBlock]) -> Dict[str, Any]:
    content = []
    tool_calls = []

    for block in output:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "thought":
            content.append({"type": "text", "text": f"<think>{block.text}</think>"})
        elif block.type == "tool_call":
            tool_calls.append(
                {
                    "function": {
                        "arguments": json.dumps(block.arguments),
                        "name": block.name,
                    },
                    "id": block.id,
                    "type": "function",
                }
            )
        else:
            raise ValueError(f"Unsupported content type: {block.type}")

    output_message: Dict[str, Any] = {"role": "assistant"}
    if content:
        output_message["content"] = content
    if tool_calls:
        output_message["tool_calls"] = tool_calls

    return output_message


def sample_to_openai_messages(sample: RenderedSample) -> Dict[str, Any]:
    result = {
        "input": {"messages": [], "tools": [], "parallel_tool_calls": True},
        "preferred_output": [],
        "non_preferred_output": [],
    }

    if sample.input.system:
        result["input"]["messages"].append({"role": "system", "content": sample.input.system})
    for message in sample.input.messages:
        content = []
        for part in message.content:
            if part.type == "text":
                content.append(part.text)
            else:
                raise ValueError(f"Unsupported content type: {part.type}")
        if len(content) != 1:
            raise ValueError(f"Expected exactly one content part for message {message}, got {len(content)}")
        result["input"]["messages"].append({"role": message.role, "content": content[0]})

    result["preferred_output"].append(prepare_output(sample.output))
    if len(sample.dispreferred_outputs) != 1:
        raise ValueError(
            f"Expected exactly one dispreferred output for sample {sample}, got {len(sample.dispreferred_outputs)}"
        )
    result["non_preferred_output"].append(prepare_output(sample.dispreferred_outputs[0]))

    return result


def prepare_samples(samples: List[RenderedSample]) -> List[Dict[str, Any]]:
    return [sample_to_openai_messages(sample) for sample in samples]


# %%
prepared_train_samples = prepare_samples(train_samples)
prepared_val_samples = prepare_samples(val_samples)


# %% [markdown]
# Upload the prepared datasets to OpenAI.
#


# %%
def upload_dataset_to_openai(samples, openai_client) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in samples:
            json.dump(item, f)
            f.write("\n")
        f.flush()

        print(f"File persisted on path [{f.name}]")

        with open(f.name, "rb") as file:
            file_object = openai_client.files.create(file=file, purpose="fine-tune")

        return file_object.id


openai_client = openai.OpenAI()

dpo_fine_tuning_object_id = upload_dataset_to_openai(prepared_train_samples, openai_client)
val_file_object_id = upload_dataset_to_openai(prepared_val_samples, openai_client)

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
# You'll need to add this model to a new variant you define in your config.
#
# Then, you're all set!
#
# You can change the weight to enable a gradual rollout of the new model.
#
# You might also add other parameters (e.g. max_tokens, temperature) to the variant section in the config file.
#
