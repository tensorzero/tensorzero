# %%
# type: ignore

# %% [markdown]
# # Automated Prompt Engineering using MIPRO
#
# This notebook provides an automated approach to optimizing prompt engineering using the [Multi-prompt Instruction PRoposal Optimizer (MIPRO)](https://arxiv.org/abs/2406.11695v1).
# It is designed for TensorZero users who want to optimize their system prompts based on collected inference and feedback data. As such, we currently only support prompt optimization for applications with a single system prompt.
#
# Support for applications with multiple system prompts is in the pipeline. If this use case interests you, please see our our [LLM Gym Example](https://github.com/tensorzero/llmgym/tree/main/examples/mipro) for a full implementation.
#
# By following this guide, you can systematically refine your prompts to improve model performance in specific tasks.
#

# %% [markdown]
# ## Overview
#
# The optimization process involves the following steps:
#
# 1. **Generate candidate instructions and demonstrations**
#    - Candidate instructions are generated using OpenAI's o1 model based on a system template and an optional schema.
#    - Candidate demonstrations are sets of few-shot examples sampled from the training dataset.
# 2. **Evaluate Instruction-Demonstration Pairs**
#    - Sample an instruction and demonstration pair and score it using a Large Language Model (LLM) judge.
#    - The judge (using OpenAI's GPT-4o-mini model) scores the quality of the instruction-demonstration pair.
#    - Scores are aggregated over the evaluation set to produce a final evaluation score.
# 3. **Optimization via Search Algorithms**
#    - Utilize a random search or a Tree-structured Parzen Estimator (TPE) to determine the next instruction and demonstration pair for evaluation.
# 4. **Iterate the Optimization Process**
#    - Repeat the optimization process for a fixed number of iterations.
# 5. **Select the Best Performing Prompts**
#    - The instruction and demonstration pairs corresponding to the highest-performing prompts are formatted to yield optimized system templates.

# %% [markdown]
# ## Step 1: Define Function Configuration Parameters
#
# Specify the TensorZero function you want to optimize. The example below optimizes the system prompt for Named Entity Recognition (NER):
#
# - **Function Configuration Directory:** Location of the function’s configuration files.
#
# - **Function Name:** The TensorZero function being optimized.
#
# - **Model Variant:** The specific function variant to use as an example for the system template.
#

# %%
# Configuation arguments for the function you want to optimize the prompt for
CONFIG_DIR = "../../examples/data-extraction-ner/config/tensorzero.toml"

# The name of the function you want to optimize the prompt for
FUNCTION_NAME = "extract_entities"

# The name of the variant to use
TEMPLATE_VARIANT_NAME = "gpt_4o_mini"

# %% [markdown]
# ## Step 2: Configure the LLM Judge for Metric Optimization
#
# The LLM judge guides the optimization process by evaluating prompt effectiveness. You must define:
#
# - **Task Description:** A summary of the task being optimized.
# - **Optimization Metric:** The metric used for evaluating prompt effectiveness (e.g. Jaccard similarity between predicted and ground truth entities).
#

# %%
# Description of the task you are optimizing the prompt for to be used by the optimizer judge
TASK_DESCRIPTION = "The task is to extract named entities from the input text."

# Metric definition for scoring generated prompts
METRIC_PROPERTIES = "The metric is the Jaccard similarity between the predicted and ground truth entities."

# %% [markdown]
# ## Step 3: Define Optimization Parameters
#
# The following parameters control the optimization process. Experimenting with different values can help refine results:
#
# - **Search Space**
#   - `NUM_CANDIDATE_INSTRUCTIONS`: Number of candidate instructions to generate.
#   - `NUM_CANDIDATE_DEMONSTRATIONS`: Number of candidate demonstrations to sample.
# - **Optimization Control**
#   - `MAX_ITERATIONS`: Number of optimization steps.
#   - `MAX_EXAMPLES_PER_DEMONSTRATION`: Maximum few-shot examples per demonstration.
# - **Evaluation Control**
#   - `EVAL_FRACTION`: Fraction of the dataset used for scoring generated prompts.
#   - `MAX_SAMPLES`: Limit on the number of demonstration samples.
# - **Reproducibility**
#   - `SEED`: Random seed for consistent results.
#

# %%
# Number of candidate instructions to generate and search over
NUM_CANDIDATE_INSTRUCTIONS = 10

# Number of candidate demonstrations to sample and search over
NUM_CANDIDATE_DEMONSTRATIONS = 10

# Maximum number of demonstrations in each candidate demonstration set
MAX_EXAMPLES_PER_DEMONSTRATION = 10

# Maximum number of search steps taken by the optimization algorithm for evaluating instruction-demonstration pairs
MAX_ITERATIONS = 5

# Set optimization direction ('maximize' or 'minimize') based on the metric properties you described above.
OPTIMIZER_DIRECTION = "maximize"

# Fraction of the dataset used by the judge to score the quality of the generated prompt
EVAL_FRACTION = 0.2

# Limit on the number of samples for demonstration selection
MAX_SAMPLES = 100_000

# Random seed for reproducibility
SEED = 0

# %% [markdown]
# ## Import Dependencies
#

# %%
import os
import sys

tensorzero_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if tensorzero_path not in sys.path:
    sys.path.append(tensorzero_path)

# %%
import asyncio
import json
from pathlib import Path
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from minijinja import Environment
from openai import AsyncOpenAI
from optuna.samplers import TPESampler
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatCompletionConfig,
    ChatInferenceOutput,
    ListInferencesRequest,
    RenderedSample,
)
from tqdm.asyncio import tqdm_asyncio
from utils.client_calls import candidate_inference, get_instructions, judge_answer

from recipes.util import train_val_split

# %% [markdown]
# ## Initialize Clients
#
# We use the OpenAI SDK for inference (instruction generation, candidate evaluation, and judging)
# and the TensorZero SDK for non-inference operations (config inspection, data retrieval, sample rendering).

# %%
MAX_CONCURRENT_REQUESTS = 50

# %%
# OpenAI client for all inference calls (instruction generation, judging, and candidate evaluation)
openai_client = AsyncOpenAI()
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Load MIPRO templates for instruction generation and judging
_MIPRO_CONFIG_DIR = Path("config/functions")
mipro_templates = {
    "generate_instruction_system": (
        _MIPRO_CONFIG_DIR / "generate_instruction" / "baseline" / "system_template.minijinja"
    ).read_text(),
    "judge_answer_system": (_MIPRO_CONFIG_DIR / "judge_answer" / "baseline" / "system_template.minijinja").read_text(),
    "judge_answer_user": (_MIPRO_CONFIG_DIR / "judge_answer" / "baseline" / "user_template.minijinja").read_text(),
}
mipro_env = Environment(templates=mipro_templates)

# %% [markdown]
# ## Load Data
#
# Load the TensorZero configuration for the function you want to optimize the prompt for.
#

# %% [markdown]
# Retrieve the configuration for the variant with the templates we'll use for prompt optimization.
#
# The TensorZero SDK is used here for non-inference operations (config inspection, data retrieval, sample rendering).

# %%
# TensorZero client for non-inference operations (config, data retrieval, sample rendering)
original_client = await AsyncTensorZeroGateway.build_embedded(
    config_file=CONFIG_DIR,
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
)

# %%
config = original_client.experimental_get_config()
base_function = config.functions[FUNCTION_NAME]
base_variant = base_function.variants[TEMPLATE_VARIANT_NAME]
if not isinstance(base_variant, ChatCompletionConfig):
    raise ValueError("Only chat completion variants are supported")

# Extract the OpenAI model name from the TensorZero model reference (e.g. "openai::gpt-4o-mini" -> "gpt-4o-mini")
tz_model_name = base_variant.model
if "::" in tz_model_name:
    openai_model_name = tz_model_name.split("::", 1)[1]
else:
    raise ValueError(
        f"Cannot extract OpenAI model name from `{tz_model_name}`. "
        "Expected format: `provider::model_name` (e.g. `openai::gpt-4o-mini`)."
    )

# %% [markdown]
# Query the inferences and demonstration feedback from ClickHouse.
#

# %%
response = await original_client.list_inferences(
    request=ListInferencesRequest(
        function_name="extract_entities",
        output_source="demonstration",  # or "inference"
    )
)
inferences = response.inferences

# %%
rendered_samples = await original_client.experimental_render_samples(
    stored_samples=inferences,
    variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME},
)

# %% [markdown]
# Split the data into training and validation sets for fine-tuning.

# %%
train_samples, val_samples = train_val_split(
    rendered_samples,
    val_size=EVAL_FRACTION,
    last_inference_only=True,
)

# %% [markdown]
# Retrieve the system, user, and assistant templates in the variant (if any), and initialize a minijinja environment with them.
#

# %%
templates = {}
candidate_template = """
{{ instructions }}
{% for demo in demonstrations %}
=== Demonstration {{ loop.index }} ===
{% for msg in demo.messages %}{% if msg.role != 'system' %}
**{{ msg.role | capitalize }}**
{% if msg.content is defined %}{% if msg.content is string %}
{{ msg.content }}
{% else %}{% for block in msg.content %}
{{ block.text }}
{% endfor %}{% endif %}{% endif %}
{% if msg.tool_calls is defined %}{% for call in msg.tool_calls %}
> Tool Call: `{{ call.function.name }}` ({{ call.function.arguments }})
{% endfor %}{% endif %}{% endif %}{% endfor %}{% endfor %}
"""

templates["candidate"] = candidate_template

env = Environment(templates=templates)


# %% [markdown]
# Render the messages in the input and demonstration columns.
#


# %%
def prepare_output(output: ChatInferenceOutput) -> Dict[str, Any]:
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


def sample_to_openai_messages(sample: RenderedSample) -> List[Dict[str, Any]]:
    rendered_messages = []
    # Add the system message to the rendered messages
    # If there is data passed in or a system template there must be a system message
    system = sample.input.system
    if system:
        rendered_messages.append({"role": "system", "content": system})

    # Add the input messages to the rendered messages
    for message in sample.input.messages:
        content = []
        for part in message.content:
            if part.type == "text":
                content.append({"type": "text", "text": part.text})
            elif part.type == "tool_call":
                content.append(
                    {
                        "type": "tool_call",
                        "name": part.raw_name,
                        "arguments": part.raw_arguments,
                    }
                )
            elif part.type == "tool_result":
                content.append({"type": "tool_result", "name": part.name, "result": part.result})
            elif part.type == "thought":
                content.append({"type": "text", "text": f"<think>{part.text}</think>"})
            else:
                raise ValueError(f"Unsupported content type: {part.type}")
        rendered_messages.append({"role": message.role, "content": content})

    # Add the output to the messages
    if sample.output:
        rendered_messages.append({"role": "assistant", "content": prepare_output(sample.output)})

    return rendered_messages


def sample_to_user_messages(sample: RenderedSample) -> List[Dict[str, Any]]:
    """Extract non-system messages from a rendered sample (for use with candidate_inference)."""
    all_messages = sample_to_openai_messages(sample)
    # Filter out system messages — the candidate system prompt will be provided separately
    return [msg for msg in all_messages if msg["role"] != "system"]


# %% [markdown]
# Split the data into training and evaluation sets.
# The training set is used to generate candidate demonstrations.
# The evaluation set is used by the judge to score the quality of the generated prompt.
#

# %%
# Create training and validation DataFrames based on episode_ids
train_examples = [(sample_to_openai_messages(example), example) for example in train_samples]
val_examples = [(sample_to_openai_messages(example), example) for example in val_samples]

# %% [markdown]
# ## Generate Candidate Instructions
#
# Given the function's system template as an example, generate a set of candidate instructions to optimize the prompt over.
#

# %%
if not isinstance(base_variant, ChatCompletionConfig):
    raise ValueError("Only chat completion variants are supported")

example_instructions = base_variant.system_template
if example_instructions is None:
    raise ValueError("System template is required")

if base_function.system_schema is not None:
    example_schema = json.dumps(base_function.system_schema.model_json_schema())
else:
    example_schema = None

instruction_system_prompt = mipro_env.render_template(
    "generate_instruction_system",
    example_instructions=example_instructions,
    **({"example_schema": example_schema} if example_schema else {}),
)

responses = await tqdm_asyncio.gather(
    *[
        get_instructions(
            client=openai_client,
            system_prompt=instruction_system_prompt,
            semaphore=semaphore,
        )
        for _ in range(NUM_CANDIDATE_INSTRUCTIONS)
    ]
)

candidate_instructions = [example_instructions]
for response in responses:
    if response is None:
        continue
    candidate_instructions.append(response["instructions"])


# %% [markdown]
# ## Generate Candidate Demonstrations
#
# Given the training set, generate a set of candidate demonstrations to optimize the prompt over.
#


# %%
def generate_demonstrations(
    train_examples: List[Tuple[List[Dict[str, Any]], RenderedSample]],
    max_examples_per_demonstration: int,
    seed: int = 42,
) -> str:
    shuffle(train_examples)
    demonstrations = []
    for example in train_examples[:max_examples_per_demonstration]:
        demonstrations.append({"messages": example[0]})
    return demonstrations


# %%
candidate_demonstrations = [
    generate_demonstrations(
        train_examples=train_examples,
        max_examples_per_demonstration=MAX_EXAMPLES_PER_DEMONSTRATION,
        seed=seed,
    )
    for seed in range(NUM_CANDIDATE_DEMONSTRATIONS)
]

# %%
candidate_demonstrations[0]

# %%
print(
    env.render_template(
        "candidate",
        demonstrations=candidate_demonstrations[0],
        instructions=candidate_instructions[1],
    )
)

# %% [markdown]
# ## Optimize the Prompt
#
# ### Define the optimization objective
#

# %%
# Initialize online statistics
num_instructions = len(candidate_instructions)
num_demonstrations = len(candidate_demonstrations)


# %%
def format_response(response: Optional[Any]) -> str:
    if response is None:
        return ""
    content = response.choices[0].message.content
    return content if content else ""


async def objective(trial: optuna.Trial):
    # Sample an instruction and a demonstration set
    instruction_index = trial.suggest_categorical("instruction_index", range(num_instructions))
    demonstration_index = trial.suggest_categorical("demonstration_index", range(num_demonstrations))
    # Format the candidate prompt
    candidate_system_prompt = env.render_template(
        "candidate",
        instructions=candidate_instructions[instruction_index],
        demonstrations=candidate_demonstrations[demonstration_index],
    )

    # Asynchronously generate answers for each query in the evaluation set
    responses = await tqdm_asyncio.gather(
        *[
            candidate_inference(
                client=openai_client,
                messages=sample_to_user_messages(example[1]),
                system_prompt=candidate_system_prompt,
                model_name=openai_model_name,
                semaphore=semaphore,
            )
            for example in val_examples
        ]
    )

    # Score the responses using the judge
    judge_system_prompt = mipro_env.render_template(
        "judge_answer_system",
        task_description=TASK_DESCRIPTION,
        metric_properties=METRIC_PROPERTIES,
    )
    judge_responses = await tqdm_asyncio.gather(
        *[
            judge_answer(
                client=openai_client,
                system_prompt=judge_system_prompt,
                user_prompt=mipro_env.render_template(
                    "judge_answer_user",
                    prediction=format_response(response) if response is not None else "",
                    ground_truth=str(example[1].output),
                ),
                semaphore=semaphore,
            )
            for response, example in zip(responses, val_examples)
        ]
    )

    # Aggregate the scores
    scores = []
    for response in judge_responses:
        if response is not None:
            scores.append(response["score"])

    # Return the mean score
    return np.mean(scores)


# %% [markdown]
# ### Random Search
#
# We start by sampling a random instruction and demonstration at each iteration in the optimization loop.
#

# %%
study_random = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=SEED), direction=OPTIMIZER_DIRECTION)

for iteration in range(MAX_ITERATIONS):
    trial = study_random.ask()

    value = await objective(trial)
    print(f"Iteration {iteration + 1}: {value}")

    frozen_trial = study_random.tell(trial, value)
    study_random._log_completed_trial(frozen_trial)

# %% [markdown]
# ### Tree-structured Parzen Estimator
#
# Following the MIPRO paper, we use a tree-structured parzen estimator (TPE) to sample the next instruction and demonstration pair to evaluate.
#

# %%
study_tpe = optuna.create_study(sampler=TPESampler(seed=SEED), direction=OPTIMIZER_DIRECTION)

for iteration in range(MAX_ITERATIONS):
    trial = study_tpe.ask()

    value = await objective(trial)
    print(f"Iteration {iteration + 1}: {value}")

    frozen_trial = study_tpe.tell(trial, value)
    study_tpe._log_completed_trial(frozen_trial)

# %% [markdown]
# ## Save the Optimized Candidate
#
# We now have an estimate of the best instruction and demonstration pair.
# We can now generate an optimized system template.
#

# %%
optimized_system_template = env.render_template(
    "candidate",
    instructions=candidate_instructions[study_tpe.best_params["instruction_index"]],
    demonstrations=candidate_demonstrations[study_tpe.best_params["demonstration_index"]],
)
print(optimized_system_template)

# %% [markdown]
# You can make a new variant with this optimized system template.
#

# %% [markdown]
# ## Conclusion
#
# By following this notebook, you can systematically refine prompts for better performance.
# The optimized prompt can be saved and used in production by updating the function's system template configuration.
#
