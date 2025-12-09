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
# - You'll also need to [install](https://docs.fireworks.ai/tools-sdks/firectl/firectl) the CLI tool `firectl` on your machine and sign in with `firectl signin`. You can test that this all worked with `firectl whoami`. We use `firectl` for deployment to Fireworks in this example but you can serve the model however you prefer.
# - Update the following parameters:

# %%
CONFIG_PATH = "../../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

METRIC_NAME = "jaccard_similarity"

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
# Select a [supported model](https://docs.pytorch.org/torchtune/main/api_ref_models.html) to fine tune.
#
#
# If you want to use a supported model that we have not yet included in the MODELS dictionary in utils.py, you will need to update the dictionary.
#
# Alternatively, you can write the `TUNING_CONFIG` defined below from scratch following the [example configs](https://github.com/pytorch/torchtune/tree/main/recipes/configs) provided by torchtune.

# %%
# The name of the model to fine-tune.
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
import sys

tensorzero_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if tensorzero_path not in sys.path:
    sys.path.append(tensorzero_path)

# %%
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
    TOKENIZER_CONFIG["merges_file"] = str(checkpoint_dir / TOKENIZER_CONFIG["merges_file"])
TOKENIZER_CONFIG["max_seq_len"] = None  # Can set to an integer value to reduce your memory footprint

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

import toml
import yaml
from tensorzero import (
    FloatMetricFilter,
    TensorZeroGateway,
)
from tensorzero.util import uuid7
from util import tensorzero_rendered_samples_to_conversations, train_val_split

# %% [markdown]
# Load and render the stored inferences

# %%
tensorzero_client = TensorZeroGateway.build_embedded(
    config_file=CONFIG_PATH,
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
    timeout=15,
)

# %% [markdown]
# Set the metric filter as needed

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
# Query the inferences and feedback from ClickHouse.

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

# %%
rendered_samples = tensorzero_client.experimental_render_samples(
    stored_samples=stored_inferences,
    variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME},
)

# %% [markdown]
# Split the data into training and validation sets for fine-tuning.

# %%
train_samples, eval_samples = train_val_split(
    rendered_samples,
    val_size=VAL_FRACTION,
    last_inference_only=True,
)

# %% [markdown]
# Convert the rendered samples to conversations for tokenization

# %%
train_conversations = tensorzero_rendered_samples_to_conversations(train_samples, conversation_key="messages")
eval_conversations = tensorzero_rendered_samples_to_conversations(eval_samples, conversation_key="messages")

# %%
with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir = Path(temp_dir)

    # Write training JSONL
    train_json_path = temp_dir / "train.json"
    with train_json_path.open("w") as f:
        for item in train_conversations:
            json.dump(item, f)
            f.write("\n")

    # Write evaluation JSONL
    val_json_path = temp_dir / "eval.json"
    with val_json_path.open("w") as f:
        for item in eval_conversations:
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
base_model_id = "llama-v3p3-70b-instruct"
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
            "providers": {"fireworks": {"type": "fireworks", "model_name": model_identifier}},
        }
    }
}

print(toml.dumps(model_config))

# %% [markdown]
# You're all set!
#
# You can change the weight to enable a gradual rollout of the new model.
