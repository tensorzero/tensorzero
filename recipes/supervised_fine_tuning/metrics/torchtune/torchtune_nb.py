# %%
# type: ignore

# %% [markdown]
# # torchtune Supervised Fine-Tuning
#
# This recipe allows TensorZero users to fine-tune models using [torchtune](https://docs.pytorch.org/torchtune/main/) and their own data.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to fine-tune a model using your own data and any prompt you want.
#
# We demonstrate how to deploy a LoRA fine-tuned model for serverless inference using [Fireworks](https://fireworks.ai). Full instructions to deploy LoRA or full fine-tuned models are provided by [Fireworks](https://docs.fireworks.ai/fine-tuning/fine-tuning-models), [Together](https://docs.together.ai/docs/deploying-a-fine-tuned-model), and other inference providers. You can also use [vLLM](https://docs.vllm.ai/en/latest/examples/online_serving/api_client.html) to serve your fine-tuned model locally. The TensorZero client seemlessly integrates inference using your fine-tuned model for any of these approaches.
#
# To get started:
#
# - Set your `TENSORZERO_CLICKHOUSE_URL` enironment variable to point to the database containing the historical inferences you'd like to train on.
# - Set your `HF_TOKEN` to use Llama or Gemma models downloaded through huggingface.
# - Set the environment variable `CHECKPOINT_HOME` to a path with sufficient storage to save the base LLM checkpoints.
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

# Fraction of the data to use for validation
VAL_FRACTION = 0.2

# Maximum number of samples to use for fine-tuning
MAX_SAMPLES = 100_000

# Random seed
SEED = 42

# %% [markdown]
# Select a model to fine tune

# %%
# The name of the model to fine-tune (supported models: https://docs.pytorch.org/torchtune/main/api_ref_models.html)
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Whether to use LoRA or not. Set to False for full model fine-tuning
# If set to False, SEVERLESS must also be False as you will need to create your own deployment
USE_LORA = True

# Whether to use a serverless deployment.
# Set to False is full model fine tuning or using LoRA for a model without serverless support
SERVERLESS = True

# Set to true if you want to include system and user messages in loss calculation
TRAIN_ON_INPUT = False

# Number of server nodes to use
NNODES = 1

# Number of devices (e.g., GPUs) to use per node
NPROC_PER_NODE = 1

# Set the directory where you would like to save the fine-tuned model
OUTPUT_DIR = "fine-tuned"

# %% [markdown]
# Download the model

# %%
import os
import subprocess
from pathlib import Path

from utils import MODELS, list_checkpoints

# %%
assert "CHECKPOINT_HOME" in os.environ, "CHECKPOINT_HOME environment variable not set"
assert "HF_TOKEN" in os.environ, "HF_TOKEN environment variable not set"

checkpoint_home = Path(os.environ["CHECKPOINT_HOME"])
checkpoint_dir = checkpoint_home / MODEL_NAME

command = [
    "tune",
    "download",
    MODEL_NAME,
    "--output-dir",
    f"{checkpoint_dir}",
]
print(" ".join(command))
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print("Error occurred:", e.stderr)

# %% [markdown]
# Optionally, use Low Rank Adaptation.
#
# Some [Fireworks Models]() support [serverless LoRA deployment](https://docs.fireworks.ai/fine-tuning/fine-tuning-models), but full fine-tuning usually needs some form of reserved capacity.

# %%
if USE_LORA:
    MODEL_CONFIG = {
        "_component_": MODELS[MODEL_NAME]["modules"]["lora"],
        "lora_attn_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "output_proj",
        ],
        "apply_lora_to_mlp": True,
        "apply_lora_to_output": False,
        "lora_rank": 8,  # higher increases accuracy and memory
        "lora_alpha": 16,  # usually alpha=2*rank
        "lora_dropout": 0.0,
    }
else:
    MODEL_CONFIG = {
        "_component_": MODELS[MODEL_NAME]["modules"]["full"],
    }

# %% [markdown]
# Set the training parameters

# %%
TOKENIZER_CONFIG = MODELS[MODEL_NAME]["tokenizer"]
if TOKENIZER_CONFIG.get("path"):
    TOKENIZER_CONFIG["path"] = str(checkpoint_dir / TOKENIZER_CONFIG["path"])
if TOKENIZER_CONFIG.get("merges_file"):
    TOKENIZER_CONFIG["merges_file"] = str(
        checkpoint_dir / TOKENIZER_CONFIG["merges_file"]
    )
TOKENIZER_CONFIG["max_seq_len"] = (
    None  # Can set to an integer value to reduce your memory footprint
)

TUNING_CONFIG = {
    "output_dir": OUTPUT_DIR,
    # Tokenizer
    "tokenizer": TOKENIZER_CONFIG,
    # Model Arguments
    "model": MODEL_CONFIG,
    "checkpointer": {
        "_component_": "torchtune.training.FullModelHFCheckpointer",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_files": list_checkpoints(checkpoint_dir),
        "recipe_checkpoint": None,
        "output_dir": OUTPUT_DIR,
        "model_type": MODELS[MODEL_NAME]["checkpointer"]["model_type"],
    },
    "resume_from_checkpoint": False,
    "save_adapter_weights_only": USE_LORA,
    # Optimizer and Scheduler
    "optimizer": {
        "_component_": "torch.optim.AdamW",
        "fused": True,
        "weight_decay": 0.01,
        "lr": 1e-4,
    },
    "lr_scheduler": {
        "_component_": "torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup",
        "num_warmup_steps": 100,
    },
    "loss": {
        "_component_": "torchtune.modules.loss.LinearCrossEntropyLoss",
    },
    # Training
    "epochs": 1,
    "batch_size": 2,
    "batch_size_val": 2,
    "max_steps_per_epoch": None,
    "gradient_accumulation_steps": 8,  # Use to increase effective batch size
    "clip_grad_norm": None,
    "compile": False,  # torch.compile the model + loss: True increases speed + decreases memory
    "run_val_every_n_steps": 10,
    "seed": SEED,
    "shuffle": True,
    # Logging
    "log_every_n_steps": 1,
    "log_peak_memory_stats": True,
    "log_level": "INFO",  # DEBUG, WARN, etc.
    # Environment
    "device": "cuda",
    "dtype": "bf16",
    "enable_activation_checkpointing": True,  # True reduces memory
    "enable_activation_offloading": True,  # True reduces memory
}

# %%
import json
import tempfile
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import toml
import yaml
from tensorzero import (
    BooleanMetricNode,
    ContentBlock,
    FloatMetricNode,
    RawText,
    TensorZeroGateway,
    Text,
    Thought,
    ToolCall,
    ToolResult,
)
from tensorzero.internal import OutputMessage
from tensorzero.util import uuid7

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
# Load and render the stored inferences

# %%
tensorzero_client = TensorZeroGateway.build_embedded(
    config_file=CONFIG_PATH,
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
    timeout=15,
)

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
            return None
    if content or tool_calls:
        chatml_message: Dict[str, Any] = {"role": message.role}
        if content:
            chatml_message["content"] = "\n".join(content)
        if tool_calls:
            chatml_message["tool_calls"] = tool_calls
            if len(content) == 0:
                chatml_message["content"] = ""
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

    # Once we finish collecting all blocks, create one assistant message.
    output_message: Dict[str, Any] = {"role": "assistant"}
    if content:
        output_message["content"] = "\n".join(content)
    if tool_calls:
        output_message["tool_calls"] = tool_calls
        if len(content) == 0:
            output_message["content"] = ""

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
print(f"Actual validation fraction: {len(val_df) / len(conversations):.2f}")

# %%
with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir = Path(temp_dir)

    # Write training JSONL
    train_json_path = temp_dir / "train.json"
    with train_json_path.open("w") as f:
        for item in train_df["conversation"]:
            json.dump(item, f)
            f.write("\n")

    # Write evaluation JSONL
    val_json_path = temp_dir / "eval.json"
    with val_json_path.open("w") as f:
        for item in val_df["conversation"]:
            json.dump(item, f)
            f.write("\n")

    # Write YAML config
    config_path = temp_dir / "custom_8B_lora_single_device.yaml"
    TUNING_CONFIG["dataset"] = {
        "_component_": "torchtune.datasets.chat_dataset",
        "source": "json",
        "packed": False,  # True increases speed
        "data_files": str(train_json_path),
        "conversation_column": "messages",
        "conversation_style": "openai",
        "train_on_input": TRAIN_ON_INPUT,
    }
    TUNING_CONFIG["dataset_val"] = {
        "_component_": "torchtune.datasets.chat_dataset",
        "source": "json",
        "packed": False,  # True increases speed
        "data_files": str(val_json_path),
        "conversation_column": "messages",
        "conversation_style": "openai",
        "train_on_input": TRAIN_ON_INPUT,
    }
    TUNING_CONFIG["metric_logger"] = {
        "_component_": "torchtune.training.metric_logging.DiskLogger",
        "log_dir": str(temp_dir / "logs"),
    }
    TUNING_CONFIG["profiler"] = {  # Disabled
        "_component_": "torchtune.training.setup_torch_profiler",
        "enabled": False,
        "output_dir": str(temp_dir / "profiling_outputs"),
        "cpu": True,
        "cuda": True,
        "profile_memory": False,
        "with_stack": False,
        "record_shapes": True,
        "with_flops": False,
        "wait_steps": 5,
        "warmup_steps": 3,
        "active_steps": 2,
        "num_cycles": 1,
    }
    with open(config_path, "w") as fp:
        yaml.safe_dump(
            TUNING_CONFIG,
            fp,
            sort_keys=False,
            default_flow_style=False,  # expand lists/dicts in block style
        )
    print(f"Config written to {config_path}")
    command = [
        "tune",
        "run",
        "--nnodes",
        str(NNODES),
        "--nproc_per_node",
        str(NPROC_PER_NODE),
        "lora_finetune_distributed" if USE_LORA else "full_finetune_distributed",
        "--config",
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

checkpoint_dir = Path(OUTPUT_DIR) / f"epoch_{TUNING_CONFIG['epochs'] - 1}"

command = [
    "firectl",
    "create",
    "model",
    fine_tuned_model_id,
    str(checkpoint_dir),
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
