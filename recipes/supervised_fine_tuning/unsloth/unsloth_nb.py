# %%
# type: ignore

# %% [markdown]
# # Unsloth Supervised Fine-Tuning
#
# This recipe allows TensorZero users to fine-tune models using [Unsloth](https://unsloth.ai) and their own data.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to fine-tune a model using your own data and any prompt you want.
#
# We demonstrate how to deploy a LoRA fine-tuned model for serverless inference using [Fireworks](https://fireworks.ai). Full instructions to deploy LoRA or full fine-tuned models are provided by [Fireworks](https://docs.fireworks.ai/fine-tuning/fine-tuning-models), [Together](https://docs.together.ai/docs/deploying-a-fine-tuned-model), and other inference providers. You can also use [vLLM](https://docs.vllm.ai/en/latest/examples/online_serving/api_client.html) to serve your fine-tuned model locally. The TensorZero client seemlessly integrates inference using your fine-tuned model for any of these approaches.
#
# To get started:
#
# - Set your `TENSORZERO_CLICKHOUSE_URL` enironment variable to point to the database containing the historical inferences you'd like to train on.
# - You'll also need to [install](https://docs.fireworks.ai/tools-sdks/firectl/firectl) the CLI tool `firectl` on your machine and sign in with `firectl signin`. You can test that this all worked with `firectl whoami`.
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
# The name of the model to fine-tune (supported models: https://docs.unsloth.ai/get-started/all-our-models)
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"

SERVERLESS = True  # Whether to use a serverless deployment. Set to False is full model fine tuning or using LoRA for a model without serverless support

MAX_SEQ_LENGTH = 8192  # Choose any! Unsloth supports RoPE Scaling internally!

MODEL_DTYPE = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage. Can be False.

# %% [markdown]
# Choose the appropriate chat template for the selected model

# %%
from unsloth.chat_templates import CHAT_TEMPLATES

print(list(CHAT_TEMPLATES.keys()))

# %%
# Choose the chat template corresponding the the model you're fine-tuning.
# For example, if you're fine-tuning "unsloth/Meta-Llama-3.1-8B-Instruct" you should use "llama-3.1"
CHAT_TEMPLATE = "llama-3.1"

# %% [markdown]
# Set training parameters

# %%
NUM_EPOCHS = 1

LEARNING_RATE = 2e-4

BATCH_SIZE = 4

# %% [markdown]
# Optionally, use Low Rank Adaptation.
#
# Some [Fireworks Models]() support [serverless LoRA deployment](https://docs.fireworks.ai/fine-tuning/fine-tuning-models), but full fine-tuning usually needs some form of reserved capacity.

# %%
# Whether to use LoRA or not. Set to False for full model fine-tuning
# If set to False, SEVERLESS must also be False as you will need to create your own deployment
USE_LORA = True

# LoRA Parameters
LORA_R = 8  # LoRA rank (the bottleneck dimension in the adaptation matrices)
LORA_ALPHA = 16  # LoRA scaling factor (sometimes set to 2x the rank)
LORA_DROPOUT = 0.0  # Dropout rate applied to the LoRA layers (sometimes 0.05 or 0.1)
LORA_TARGETS = [  # Which modules to inject LoRA into (often q_proj, v_proj, or all linear layers in attention)
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_BIAS = "none"  # Whether to add bias in LoRA adapters (rarely needed)

# %%
import os
import sys

tensorzero_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
if tensorzero_path not in sys.path:
    sys.path.append(tensorzero_path)

# %%
import subprocess
import tempfile
from typing import Any, Dict

import toml
from datasets import Dataset
from tensorzero import (
    FloatMetricFilter,
    TensorZeroGateway,
)
from tensorzero.util import uuid7
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

from recipes.util import tensorzero_rendered_samples_to_conversations, train_val_split

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
train_conversations = tensorzero_rendered_samples_to_conversations(train_samples)
eval_conversations = tensorzero_rendered_samples_to_conversations(eval_samples)

# %% [markdown]
# Instantiate the model and tokenizer

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=MODEL_DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# %% [markdown]
# Apply the chat completion template

# %%
tokenizer = get_chat_template(
    tokenizer,
    chat_template=CHAT_TEMPLATE,
)


# %%
def process_conversations(inference: Dict[str, Any]):
    inference.update({"add_generation_prompt": False, "tokenize": False})
    return {
        "text": tokenizer.apply_chat_template(
            **inference,
        )
    }


# %%
train_dataset = Dataset.from_list([process_conversations(sample) for sample in train_conversations])
eval_dataset = Dataset.from_list([process_conversations(sample) for sample in eval_conversations])

# %% [markdown]
# Set LoRA parameters

# %%
if USE_LORA:
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias=LORA_BIAS,
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=SEED,
        use_rslora=False,  # Unsloth supports rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

# %% [markdown]
# Build the trainer

# %%
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        eval_strategy="steps",
        eval_steps=20,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        num_train_epochs=NUM_EPOCHS,  # Set this for 1 full training run.
        lr_scheduler_type="linear",
        warmup_steps=5,
        logging_steps=10,
        save_strategy="no",
        seed=SEED,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        report_to="none",  # Use this for WandB etc
    ),
)

# %% [markdown]
# Train the model

# %%
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
trainer_stats = trainer.train()

# %% [markdown]
# Now that the model is done training, we need to [deploy](https://docs.fireworks.ai/fine-tuning/fine-tuning-models#deploying-and-using-a-model) it to Fireworks serverless inference. If you need high or guaranteed throughput you can also deploy the model to [reserved capacity](https://docs.fireworks.ai/deployments/reservations) or an on-demand [deployment](https://docs.fireworks.ai/guides/ondemand-deployments).

# %%
base_model_id = "llama-v3p1-8b-instruct"
fine_tuned_model_id = f"{MODEL_NAME.lower().replace('/', '-').replace('.', 'p')}-{str(uuid7()).split('-')[-1]}"

with tempfile.TemporaryDirectory() as tmpdirname:
    tmpdirname = "trainer_output"
    print(f"Saving to temp dir: {tmpdirname}")
    model.save_pretrained(tmpdirname)
    tokenizer.save_pretrained(tmpdirname)

    base_model_path = f"accounts/fireworks/models/{base_model_id}"
    command = [
        "firectl",
        "create",
        "model",
        fine_tuned_model_id,
        tmpdirname,
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
# Once the model is deployed, you can add the fine-tuned model and a new variant to your config file.

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
