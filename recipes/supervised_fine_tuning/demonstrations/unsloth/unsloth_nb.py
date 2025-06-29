# %%
# type: ignore

# %% [markdown]
# # Unsloth Supervised Fine-Tuning
#
# This recipe allows TensorZero users to fine-tune open-source LLMs with your own data and export them for serving with vLLM or Ollama.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to fine-tune a model using your own data and any prompt you want.
# We follow the Unsloth [notebook](<https://github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb>).
#

# %% [markdown]
# To get started:
#
# - Set the `TENSORZERO_CLICKHOUSE_URL` environment variable. For example: `TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero"`
# - Ensure you have access to a compatible [GPU](https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements) with the appropriate drivers installed.
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

# Maximum number of samples to use for fine-tuning
MAX_SAMPLES = 100_000

# The name of the model to fine-tune (supported models: https://docs.unsloth.ai/get-started/all-our-models)
MODEL_NAME = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"

# At the time of writing, Fireworks does not support tool call content blocks in assistant messages. Or the tool role.
# We will drop these invalid messages from the dataset by default.
# You can set this to False to keep the invalid messages in the dataset.
DROP_INVALID_MESSAGES = True

# Sequence length configuration for unsloth fine-tuning
# This is the maximum sequence length that will be used during training
MAX_SEQ_LENGTH = 2048

# Data type for model weights. None for auto-detection
# - Float16: for Tesla T4, V100 GPUs
# - Bfloat16: for Ampere+ GPUs (A100, H100)
DTYPE = None

# Enable 4-bit quantization to reduce VRAM usage
LOAD_IN_4BIT = True

# %%
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
import torch
from clickhouse_connect import get_client
from datasets import Dataset
from minijinja import Environment
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
)  # Use FastModel instead of FastLanguageModel for MoE models
from unsloth.chat_templates import get_chat_template, train_on_responses_only


from trl import SFTConfig, SFTTrainer

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

df.sample(20)[["output", "value"]]

# %% [markdown]
# Render the inputs to HuggingFace's multiturn chat format.
#

# %%
def warning_message(role: str) -> str:
    return (
        f"Multiple content blocks detected in {role} message. "
        f"Since some models have limited support for multiple content blocks, "
        f"all content blocks will be concatenated with newlines. "
        f"Please verify this behavior is appropriate for your use case "
        f"and consider restructuring your messages if needed."
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
                f"Content block type '{content_block['type']}' not supported. Only text input allowed. Dropping example.",
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
        elif content_block["type"] == "tool_call" and role == "assistant":
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
        elif content_block["type"] == "tool_result" and role == "user":
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
                    f"Content block type '{content_block['type']}' not supported. Only text content is allowed. Dropping example.",
                    UserWarning,
                )
                return None
            if content_block["type"] == "text":
                content.append({"type": "text", "text": content_block["text"]})
            elif content_block["type"] == "thought":
                content.append(
                    {"type": "text", "text": f"<think>{content_block['text']}</think>"}
                )
            elif content_block["type"] == "tool_call":
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

    return rendered_messages


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
# Here we'll load the model and configure LoRA adapters
#

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# %% [markdown]
# We now use `get_chat_template` function to get the correct chat template.
#

# %%
def formatting_prompts_func(examples):
    convos = examples["conversational_messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        for convo in convos
    ]
    return {
        "text": texts,
    }


dataset = Dataset.from_pandas(df[["conversational_messages"]])
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

# %% [markdown]
# Train the model
#

# %%
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use this for WandB etc
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
    ),
)

# We also use Unsloth's train_on_completions method to only train on the assistant outputs and
# ignore the loss on the user's inputs.


def get_chat_part(role: str, enable_thinking=False) -> str:
    return (
        tokenizer.apply_chat_template(
            [{"role": role, "content": ""}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        .strip()
        .rstrip(tokenizer.eos_token)
        .split(tokenizer.eos_token)[-1]
    )


instruction_part = get_chat_part("user")

response_part = get_chat_part("assistant")

trainer = train_on_responses_only(
    trainer,
    instruction_part=instruction_part,
    response_part=response_part,
    force_match=False,
)

# %% [markdown]
# Check that the system and instruction prompts are properly masked in the training data.
#

# %%
instruction_part

# %%
response_part

# %%
print(tokenizer.decode(trainer.train_dataset[1]["input_ids"]))

# %%
space = tokenizer(" ", add_special_tokens=True).input_ids[0]
tokenizer.decode(
    [space if x == -100 else x for x in trainer.train_dataset[1]["labels"]]
).strip()

# %% [markdown]
# We can see the System and Instruction prompts are successfully masked.
#
# Now we start the fine-tuning job. Only the assistant's responses are used for loss calculation.
#

# %%
trainer_stats = trainer.train()

# %% [markdown]
# Now that the model is done training, save your fine-tuned model in the desired format (16bit, 4bit, LoRA adapters, or GGUF) for deployment with vLLM, Ollama, or llama.cpp.
#

# %%
# Merge to 16bit
if True:
    model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# Merge to 4bit
if False:
    model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")

# Just LoRA adapters
if False:
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")

# Save to 8bit Q8_0
if False:
    model.save_pretrained_gguf(
        "model",
        tokenizer,
    )

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")


