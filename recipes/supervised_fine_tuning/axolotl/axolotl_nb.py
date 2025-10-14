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
# - [Install](https://docs.fireworks.ai/tools-sdks/firectl/firectl) the CLI tool `firectl` on your machine and sign in with `firectl signin`. You can test that this all worked with `firectl whoami`. We use `firectl` for deployment to Fireworks in this example but you can serve the model however you prefer.
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
import os
import sys

tensorzero_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
if tensorzero_path not in sys.path:
    sys.path.append(tensorzero_path)

# %%
import json
import subprocess
import tempfile
from pathlib import Path

import toml
import yaml
from tensorzero import (
    FloatMetricFilter,
    TensorZeroGateway,
)
from tensorzero.util import uuid7

from recipes.util import tensorzero_rendered_samples_to_conversations, train_val_split

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
# from tensorzero import BooleanMetricFilter
# metric_node = BooleanMetricFilter(
#     metric_name=METRIC_NAME,
#     value=True  # or False
# )

metric_node

# %% [markdown]
# Query the inferences from ClickHouse
#

# %%
stored_samples = tensorzero_client.experimental_list_inferences(
    function_name=FUNCTION_NAME,
    variant_name=None,
    output_source="inference",  # could also be "demonstration"
    filters=metric_node,
    limit=MAX_SAMPLES,
)

# %% [markdown]
# Render the inputs using the templates in the template variant.
#

# %%
rendered_samples = tensorzero_client.experimental_render_samples(
    stored_samples=stored_samples,
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
# Convert the rendered samples to openai format

# %%
train_conversations = tensorzero_rendered_samples_to_conversations(train_samples, conversation_key="messages")
eval_conversations = tensorzero_rendered_samples_to_conversations(eval_samples, conversation_key="messages")

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
        for item in train_conversations:
            json.dump(item, f)
            f.write("\n")

    # Write evaluation JSONL
    val_json_path = temp_dir / "eval.jsonl"
    with val_json_path.open("w") as f:
        for item in eval_conversations:
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
    TUNE_CONFIG["dataset_prepared_path"] = str(temp_dir / "prepared")
    with open(config_path, "w") as fp:
        yaml.safe_dump(
            TUNE_CONFIG,
            fp,
            sort_keys=False,
            default_flow_style=False,  # expand lists/dicts in block style
        )
    print(f"Config written to {config_path}")
    # preprocess dataset
    command = [
        "axolotl",
        "preprocess",
        str(config_path),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)
    # train
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
            "providers": {"fireworks": {"type": "fireworks", "model_name": model_identifier}},
        }
    }
}

print(toml.dumps(model_config))

# %% [markdown]
# You're all set!
#
# You can change the weight to enable a gradual rollout of the new model.
