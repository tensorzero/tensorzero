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
CONFIG_PATH = "../../../../examples/data-extraction-ner/config/tensorzero.toml"

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

MODEL_DTYPE = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)

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
import subprocess
import tempfile
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import toml
from datasets import Dataset
from tensorzero import (
    ContentBlock,
    FloatMetricFilter,
    RawText,
    TensorZeroGateway,
    Text,
    Thought,
    ToolCall,
    ToolResult,
)
from tensorzero.internal import OutputMessage
from tensorzero.util import uuid7
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

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
rendered_inferences = tensorzero_client.experimental_render_inferences(
    stored_inferences=stored_inferences,
    variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME},
)

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


# %% [markdown]
# Reformat the rendered inferences to ChatML and tokenize


# %%
def tensorzero_to_chatml_tools(tools: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Convert TensorZero tools to OpenAI format."""
    chatml_tools: List[Dict[str, Any]] = []
    if tools:
        for tool in tools:
            chatml_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
    return chatml_tools


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
    # Add tools if available
    payload = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if rendered_inference.tool_params:
        tools = tensorzero_to_chatml_tools(
            rendered_inference.tool_params.tools_available
        )
        payload["tools"] = tools
    tokenized_messages = tokenizer.apply_chat_template(**payload)

    # Drop conversations that have unknown content
    if all(msg is not None for msg in messages):
        conversations.append(
            {"text": tokenized_messages, "episode_id": rendered_inference.episode_id}
        )

conversations = pd.DataFrame(conversations)

# %% [markdown]
# Split the data into training and validation sets for fine-tuning.
#

# %%
# Get unique episode_ids
unique_episode_ids = conversations["episode_id"].unique()

# Shuffle the unique episode_ids
np.random.seed(42)
np.random.shuffle(unique_episode_ids)

# Calculate the split index for episode_ids
split_index = int(len(unique_episode_ids) * (1 - VAL_FRACTION))

# Split the episode_ids into training and validation sets
train_episode_ids = unique_episode_ids[:split_index]
val_episode_ids = unique_episode_ids[split_index:]

# Create training and validation DataFrames based on episode_ids
train_df = conversations[conversations["episode_id"].isin(train_episode_ids)]
val_df = conversations[conversations["episode_id"].isin(val_episode_ids)]

# Convert to huggingface dataset
train_dataset = Dataset.from_pandas(train_df.drop("episode_id", axis=1))
eval_dataset = Dataset.from_pandas(val_df.drop("episode_id", axis=1))

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Actual validation fraction: {len(val_df) / len(conversations):.2f}")

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
            "providers": {
                "fireworks": {"type": "fireworks", "model_name": model_identifier}
            },
        }
    }
}

print(toml.dumps(model_config))

# %% [markdown]
# You're all set!
#
# You can change the weight to enable a gradual rollout of the new model.
