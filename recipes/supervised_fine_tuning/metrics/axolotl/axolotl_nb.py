# %%
# type: ignore

# %% [markdown]
# # Axolotl Supervised Fine-Tuning
#
# This recipe allows TensorZero users to fine-tune models using [Axolotl](https://docs.axolotl.ai) and their own data.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to fine-tune a model using your own data and any prompt you want.
#
# We demonstrate how to deploy a LoRA fine-tuned model for serverless inference using [Fireworks](https://fireworks.ai). Full instructions to deploy LoRA or full fine-tuned models are provided by [Fireworks](https://docs.fireworks.ai/fine-tuning/fine-tuning-models), [Together](https://docs.together.ai/docs/deploying-a-fine-tuned-model), and other inference providers. You can also use [vLLM](https://docs.vllm.ai/en/latest/examples/online_serving/api_client.html) to serve your fine-tuned model locally. The TensorZero client seemlessly integrates inference using your fine-tuned model for any of these approaches.
#
# To get started:
#
# - Set your `TENSORZERO_CLICKHOUSE_URL` enironment variable to point to the database containing the historical inferences you'd like to train on.
# - Set your `HF_TOKEN` to use Llama or Gemma models downloaded through huggingface.
# - You'll also need to [install](https://docs.fireworks.ai/tools-sdks/firectl/firectl) the CLI tool `firectl` on your machine and sign in with `firectl signin`. You can test that this all worked with `firectl whoami`.
# - Update the following parameters:

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

# Random seed
SEED = 42

# %% [markdown]
# Select a model to fine tune

# %%
# The huggingface name of the model to fine-tune (Axolotl supports various models like LLaMA, Mistral, Mixtral, Pythia, and more)
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# The name of the chat template to use
# - tokenizer_default: Uses the chat template that is available in the tokenizer_config.json. If the chat template is not available in the tokenizer, it will raise an error.
# - alpaca/inst/chatml/gemma/cohere/llama3/phi_3/deepseek_v2/jamba: These chat templates are available in the axolotl codebase at src/axolotl/utils/chat_templates.py
CHAT_TEMPLATE = "llama3"

# Whether to use LoRA or not. Set to False for full model fine-tuning
# If set to False, SEVERLESS must also be False as you will need to create your own deployment
USE_LORA = True

# Whether to use a serverless deployment.
# Set to False is full model fine tuning or using LoRA for a model without serverless support
SERVERLESS = True

# Can add "user" to the list to fine-tune on user messages also
ROLES_TO_TRAIN = ["assistant"]

# Number of server nodes to use
DISTRIBUTED = False  # Only set to True if multiple GPUs are available. DeepSpeed will throw an error if only one GPU is available.

# %% [markdown]
# Set the tuning parameters. A complete list of all [configuration options](https://docs.axolotl.ai/docs/config.html) is provided by Axolotl.

# %%
from tensorzero.util import uuid7

TUNE_CONFIG = {
    "output_dir": f"./outputs/{MODEL_NAME}/{uuid7()}",
    # Model
    "base_model": MODEL_NAME,  # This can also be a relative path to a model on disk
    "tokenizer_type": "AutoTokenizer",
    "load_in_8bit": True,  # Set to false for full fine-tuning
    "load_in_4bit": False,
    "sequence_len": 8192,
    "sample_packing": True,
    "eval_sample_packing": False,
    "pad_to_sequence_len": True,
    # Optimization
    "gradient_accumulation_steps": 4,
    "micro_batch_size": 2,
    "num_epochs": 4,
    "optimizer": "adamw_bnb_8bit",
    "lr_scheduler": "cosine",
    "learning_rate": 0.0002,  # May want to set lower for full fine-tuning. e.g., 2e-5
    "warmup_steps": 10,  # May want to increase for full fine-tuning. e.g., 100
    "weight_decay": 0.0,
    "bf16": "auto",
    "tf32": False,
    # Logging
    "gradient_checkpointing": True,
    "resume_from_checkpoint": None,
    "logging_steps": 1,
    "flash_attention": True,
    "evals_per_epoch": 2,
    "save_strategy": "no",
    "special_tokens": {"pad_token": "<|end_of_text|>"},
    # WandB configuration
    "wandb_project": None,
    "wandb_entity": None,
    "wandb_watch": None,
    "wandb_name": None,
    "wandb_log_model": None,
}

# %% [markdown]
# Optionally, use Low Rank Adaptation.
#
# Some [Fireworks Models]() support [serverless LoRA deployment](https://docs.fireworks.ai/fine-tuning/fine-tuning-models), but full fine-tuning usually needs some form of reserved capacity.

# %%
if USE_LORA:
    TUNE_CONFIG.update(
        {
            "adapter": "lora",
            "lora_model_dir": None,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
            ],
        }
    )

# %%
import json
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import toml
import yaml
from clickhouse_connect import get_client
from tensorzero import (
    ContentBlock,
    RawText,
    StoredChatInference,
    StoredJsonInference,
    TensorZeroGateway,
    Text,
    Thought,
    ToolCall,
    ToolResult,
)
from tensorzero.internal import OutputMessage
from tensorzero.util import uuid7

# %%
TENSORZERO_GATEWAY_URL = "http://localhost:3000"

# %% [markdown]
# Load the TensorZero configuration file.

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
# Initialize the ClickHouse client.
#

# %%
assert "TENSORZERO_CLICKHOUSE_URL" in os.environ, (
    "TENSORZERO_CLICKHOUSE_URL environment variable not set"
)

clickhouse_client = get_client(dsn=os.environ["TENSORZERO_CLICKHOUSE_URL"])

# %% [markdown]
# Determine the ClickHouse table name for the function.

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

# %%
assert "optimize" in metric, "Metric is missing the `optimize` field"

threshold = FLOAT_METRIC_THRESHOLD if metric["type"] == "float" else 0.5
comparison_operator = ">=" if metric["optimize"] == "max" else "<="

inference_col = "tool_params" if function_type == "chat" else "output_schema"

query = f"""
SELECT
    i.variant_name,
    i.input,
    i.output,
    f.value,
    i.episode_id,
    i.id,
    i.{inference_col},
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
# Load and render the stored inferences

# %%
tensorzero_client = TensorZeroGateway.build_embedded(
    config_file=CONFIG_PATH,
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
    timeout=15,
)


# %%
def double_parse_arguments(example_input):
    for message in example_input["messages"]:
        for block in message["content"]:
            if block["type"] == "tool_call":
                block["arguments"] = json.loads(block["arguments"])


stored_inferences = []
for _, row in df.iterrows():
    input_data = json.loads(row["input"])
    double_parse_arguments(input_data)
    output_data = json.loads(row["output"])
    if function_type == "chat":
        stored_inferences.append(
            StoredChatInference(
                function_name=FUNCTION_NAME,
                variant_name=row["variant_name"],
                input=input_data,
                output=output_data,
                episode_id=row["episode_id"],
                inference_id=row["id"],
                tool_params=json.loads(row["tool_params"]),
            )
        )
    elif function_type == "json":
        stored_inferences.append(
            StoredJsonInference(
                function_name=FUNCTION_NAME,
                variant_name=row["variant_name"],
                input=input_data,
                output=output_data,
                episode_id=row["episode_id"],
                inference_id=row["id"],
                output_schema=json.loads(row["output_schema"]),
            )
        )

# %%
rendered_inferences = tensorzero_client.experimental_render_inferences(
    stored_inferences=stored_inferences,
    variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME},
)


# %% [markdown]
# Reformat the rendered inferences to ChatML


# %%
def message_to_chatml(message: OutputMessage) -> Optional[List[Dict[str, Any]]]:
    chatml_messages: List[Dict[str, Any]] = []
    assert message.role in ["user", "assistant"], f"Invalid role: {message.role}"
    content: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for content_block in message.content:
        if isinstance(content_block, Text):
            assert content_block.arguments is None, "Arguments should be None"
            content.append(content_block.text)
        elif isinstance(content_block, RawText):
            content.append(content_block.value)
        elif isinstance(content_block, Thought):
            content.append(f"<think>{content_block['text']}</think>")
        elif isinstance(content_block, ToolCall):
            tool_calls.append(
                {
                    "function": {
                        "arguments": content_block.raw_arguments,
                        "name": content_block.name,
                    },
                    "id": content_block.id,
                    "type": "function",
                }
            )
        elif isinstance(content_block, ToolResult):
            # Tool results get priority so that they follow the tool call in the conversation.
            # Any other "user" content will be appended in another message below.
            chatml_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": content_block.id,
                    "content": content_block.result,
                }
            )
        else:
            warnings.warn(
                f"We do not support content block type: {type(content_block)}, dropping example.",
                UserWarning,
            )
            return [None]
    if len(tool_calls) > 1:
        return [None]
    if content or tool_calls:
        chatml_message: Dict[str, Any] = {"role": message.role}
        if content:
            chatml_message["content"] = "\n".join(content)
        if tool_calls:
            chatml_message["tool_calls"] = tool_calls
        chatml_messages.append(chatml_message)

    return chatml_messages


def output_to_chatml(output: List[ContentBlock]) -> Dict[str, Any]:
    content: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for content_block in output:
        if isinstance(content_block, Text):
            assert content_block.arguments is None, "Arguments should be None"
            content.append(content_block.text)
        elif isinstance(content_block, Thought):
            content.append(f"<think>{content_block['text']}</think>")
        elif isinstance(content_block, ToolCall):
            tool_calls.append(
                {
                    "function": {
                        "arguments": content_block.raw_arguments,
                        "name": content_block.name,
                    },
                    "id": content_block.id,
                    "type": "function",
                }
            )
        else:
            warnings.warn(
                f"We do not support content block type: {type(content_block)}, dropping example.",
                UserWarning,
            )
            return None

    if len(tool_calls) > 1:
        return None
    # Once we finish collecting all blocks, create one assistant message.
    output_message: Dict[str, Any] = {"role": "assistant"}
    if content:
        output_message["content"] = "\n".join(content)
    if tool_calls:
        output_message["tool_calls"] = tool_calls

    return output_message


conversations = []
for rendered_inference in rendered_inferences:
    messages = []
    model_input = rendered_inference.input
    if model_input.system is not None:
        messages.append({"role": "system", "content": model_input.system})
    for message in model_input.messages:
        messages.extend(message_to_chatml(message))
    messages.append(output_to_chatml(rendered_inference.output))

    # Drop conversations that have unknown content
    if all(msg is not None for msg in messages):
        conversations.append(
            {
                "conversation": {"messages": messages},
                "episode_id": rendered_inference.episode_id,
            }
        )

conversations = pd.DataFrame(conversations)

# %% [markdown]
# Split the data into training and validation sets for fine-tuning.
#

# %%
# Get unique episode_ids
unique_episode_ids = conversations["episode_id"].unique()

# Shuffle the unique episode_ids
np.random.seed(SEED)
np.random.shuffle(unique_episode_ids)

# Calculate the split index for episode_ids
split_index = int(len(unique_episode_ids) * (1 - VAL_FRACTION))

# Split the episode_ids into training and validation sets
train_episode_ids = unique_episode_ids[:split_index]
val_episode_ids = unique_episode_ids[split_index:]

# Create training and validation DataFrames based on episode_ids
train_df = conversations[conversations["episode_id"].isin(train_episode_ids)]
val_df = conversations[conversations["episode_id"].isin(val_episode_ids)]

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Actual validation fraction: {len(val_df) / len(df):.2f}")

# %% [markdown]
# Set up distributed computing using [DeepSpeed](https://www.deepspeed.ai) if specified. See Axolotl for [distributed computing guidance](https://docs.axolotl.ai/docs/multi-gpu.html).

# %%
if DISTRIBUTED:
    command = [
        "axolotl",
        "fetch",
        "deepspeed_configs",
    ]
    try:
        subprocess.run(command, check=True)
        TUNE_CONFIG["deepspeed"] = "deepspeed_configs/zero1.json"
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)

# %% [markdown]
# Fine tune

# %%
with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir = Path(temp_dir)

    # Write training JSONL
    train_json_path = temp_dir / "train.jsonl"
    with train_json_path.open("w") as f:
        for item in train_df["conversation"]:
            json.dump(item, f)
            f.write("\n")

    # Write evaluation JSONL
    val_json_path = temp_dir / "eval.jsonl"
    with val_json_path.open("w") as f:
        for item in val_df["conversation"]:
            json.dump(item, f)
            f.write("\n")

    # Write YAML config
    config_path = temp_dir / "config.yaml"
    TUNE_CONFIG["datasets"] = [
        {
            "path": str(train_json_path),
            "type": "chat_template",
            "chat_template": CHAT_TEMPLATE,
            "field_messages": "messages",
            "field_system": "system",
            "roles_to_train": ROLES_TO_TRAIN,
        }
    ]
    TUNE_CONFIG["test_datasets"] = [
        {
            "path": str(val_json_path),
            "ds_type": "json",
            "split": "train",
            "type": "chat_template",
            "chat_template": CHAT_TEMPLATE,
            "data_files": [str(val_json_path)],
        }
    ]
    with open(config_path, "w") as fp:
        yaml.safe_dump(
            TUNE_CONFIG,
            fp,
            sort_keys=False,
            default_flow_style=False,  # expand lists/dicts in block style
        )
    print(f"Config written to {config_path}")
    command = [
        "axolotl",
        "train",
        str(config_path),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)

# %% [markdown]
# Now that the model is done training, we need to [deploy](https://docs.fireworks.ai/fine-tuning/fine-tuning-models#deploying-and-using-a-model) it to Fireworks serverless inference. If you need high or guaranteed throughput you can also deploy the model to [reserved capacity](https://docs.fireworks.ai/deployments/reservations) or an on-demand [deployment](https://docs.fireworks.ai/guides/ondemand-deployments).

# %%
base_model_id = "llama-v3p1-8b-instruct"
base_model_path = f"accounts/fireworks/models/{base_model_id}"

fine_tuned_model_id = f"{MODEL_NAME.lower().replace('/', '-').replace('.', 'p')}-{str(uuid7()).split('-')[-1]}"

command = [
    "firectl",
    "create",
    "model",
    fine_tuned_model_id,
    TUNE_CONFIG["output_dir"],
    "--base-model",
    base_model_path,
]
try:
    result = subprocess.run(command, capture_output=True)
    stdout = result.stdout.decode("utf-8")
    print("Command output:", stdout)
except subprocess.CalledProcessError as e:
    print("Error occurred:", e.stderr)


# %%
def get_model_id(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.strip().startswith("Name:"):
            return line.split(":")[1].strip()
    raise ValueError("Model ID not found in output")


model_identifier = get_model_id(stdout)

model_identifier

# %% [markdown]
# Create a deployment if not using a model with serverless support, if it does not support serveless addons, or if you are doing full fine-tuning.

# %%
if not SERVERLESS:
    command = ["firectl", "create", "deployment", model_identifier]
    print(" ".join(command))
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode("utf-8"))
    else:
        stdout = result.stdout.decode("utf-8")
        print(stdout)

# %% [markdown]
# Load the LoRA addon

# %%
if USE_LORA:
    command = ["firectl", "load-lora", model_identifier]
    print(" ".join(command))
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode("utf-8"))
    else:
        stdout = result.stdout.decode("utf-8")
        print(stdout)

# %% [markdown]
# Once the model is deployed, you can add the fine-tuned model to your config file.

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
