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
#     - Candidate instructions are generated using OpenAI's o1 model based on a system template and an optional schema.
#         - This is configurable in the `config/tensorzero.toml` file if you want to use a different model.
#     - Candidate demonstrations are sets of few-shot examples sampled from the training dataset.
# 2. **Evaluate Instruction-Demonstration Pairs**
#     - Sample an instruction and demonstration pair and score it using a Large Language Model (LLM) judge.
#     - The judge (a TensorZero function utilizing OpenAI's GPT-4o-mini model) scores the quality of the instruction-demonstration pair.
#     - Scores are aggregated over the evaluation set to produce a final evaluation score.
# 3. **Optimization via Search Algorithms**
#     - Utilize a random search or a Tree-structured Parzen Estimator (TPE) to determine the next instruction and demonstration pair for evaluation.
# 4. **Iterate the Optimization Process**
#     - Repeat the optimization process for a fixed number of iterations.
# 5. **Select the Best Performing Prompts**
#     - The instruction and demonstration pairs corresponding to the highest-performing prompts are formatted to yield optimized system templates.
#
#

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

# %%
# Configuation arguments for the function you want to optimize the prompt for
CONFIG_DIR = "../../examples/data-extraction-ner/config"

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
#     - `NUM_CANDIDATE_INSTRUCTIONS`: Number of candidate instructions to generate.
#     - `NUM_CANDIDATE_DEMONSTRATIONS`: Number of candidate demonstrations to sample.
# - **Optimization Control**
#     - `MAX_ITERATIONS`: Number of optimization steps.
#     - `MAX_EXAMPLES_PER_DEMONSTRATION`: Maximum few-shot examples per demonstration.
# - **Evaluation Control**
#     - `EVAL_FRACTION`: Fraction of the dataset used for scoring generated prompts.
#     - `MAX_SAMPLES`: Limit on the number of demonstration samples.
# - **Reproducibility**
#     - `SEED`: Random seed for consistent results.

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
#
# ## Import Dependencies

# %%
import asyncio
import json
import os
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
from clickhouse_connect import get_client
from minijinja import Environment
from optuna.samplers import TPESampler
from tensorzero import (
    AsyncTensorZeroGateway,
    InferenceResponse,
    JsonInferenceResponse,
    RawText,
    Text,
)
from tqdm.asyncio import tqdm_asyncio
from utils.client_calls import candidate_inference, get_instructions, judge_answer
from utils.configs.reader import load_config

# %% [markdown]
# ## Initialize the MIPRO TensorZero Client
#
# This client is used to generate candidate instructions and score the quality of responses given the candidate instructions and demonstrations.

# %%
MAX_CONCURRENT_REQUESTS = 50

# %%
mipro_client = await AsyncTensorZeroGateway.build_embedded(
    config_file="config/tensorzero.toml",
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# %% [markdown]
# ## Load Data
#
# Load the TensorZero configuration for the function you want to optimize the prompt for.

# %%
base_config = load_config(CONFIG_DIR)

# %% [markdown]
# Retrieve the configuration for the variant with the templates we'll use for prompt optimization.

# %%
assert FUNCTION_NAME in base_config.functions.keys(), (
    f"No function named `{FUNCTION_NAME}` found in config"
)
assert TEMPLATE_VARIANT_NAME in base_config.functions[FUNCTION_NAME].variants.keys(), (
    f"No variant named `{TEMPLATE_VARIANT_NAME}` found in function `{FUNCTION_NAME}`"
)

base_function = base_config.functions[FUNCTION_NAME]
base_variant = deepcopy(base_function.variants[TEMPLATE_VARIANT_NAME])

# %% [markdown]
# Initialize the ClickHouse client.

# %%
assert "TENSORZERO_CLICKHOUSE_URL" in os.environ, (
    "TENSORZERO_CLICKHOUSE_URL environment variable not set"
)

clickhouse_client = get_client(dsn=os.environ["TENSORZERO_CLICKHOUSE_URL"])

# %% [markdown]
# Determine the inference table name based on the function type.

# %%
inference_table_name = {"chat": "ChatInference", "json": "JsonInference"}.get(
    base_function.type
)

if inference_table_name is None:
    raise ValueError(f"Unsupported function type: {base_function.type}")

# %% [markdown]
# Query the inferences and demonstration feedback from ClickHouse.

# %% [markdown]
# You can use one of the metrics above, or choose `FILTER_METRIC_NAME = "demonstration"` to use ground truth demonstrations.

# %%
print(base_config.metrics.keys())

# %%
FILTER_METRIC_NAME = "demonstration"
FILTER_METRIC_THRESHOLD = 0.9

if (
    FILTER_METRIC_NAME != "demonstration"
):  # If no metric name is provided, use ground truth demonstrations
    filter_metric = base_config.metrics[FILTER_METRIC_NAME]

# %%
if (
    FILTER_METRIC_NAME == "demonstration"
):  # Assume demonstration feedback is available and used.
    query = f"""
    SELECT
        i.input,
        f.value as output,
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
else:
    feedback_table_name = {
        "float": "FloatMetricFeedback",
        "boolean": "BooleanMetricFeedback",
    }.get(filter_metric.type)

    inference_join_key = {
        "episode": "episode_id",
        "inference": "id",
    }.get(filter_metric.level)

    if inference_join_key is None:
        raise ValueError(f"Unsupported metric level: {filter_metric.level}")

    threshold = FILTER_METRIC_THRESHOLD if filter_metric.type == "float" else 0.5
    comparison_operator = ">=" if filter_metric.optimize == "max" else "<="

    query = f"""
    SELECT
        i.input,
        i.output,
        i.episode_id,
        i.function_name,
        f.value
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
        "max_samples": MAX_SAMPLES,
        "metric_name": FILTER_METRIC_NAME,
        "threshold": FILTER_METRIC_THRESHOLD,
    }

df = clickhouse_client.query_df(query, params)

df.head()

# %% [markdown]
# Retrieve the system, user, and assistant templates in the variant (if any), and initialize a minijinja environment with them.
#

# %%
templates = {}

if base_variant.assistant_template is not None:
    templates["assistant"] = base_variant.assistant_template

if base_variant.system_template is not None:
    templates["system"] = base_variant.system_template

if base_variant.user_template is not None:
    templates["user"] = base_variant.user_template

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


# %%
def render_message(message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    role = message["role"]
    assert role in ["user", "assistant"], f"Invalid role: {role}"
    content: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    rendered_messages: List[Dict[str, Any]] = []

    for content_block in message["content"]:
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
            role_message["content"] = content
        if tool_calls:
            role_message["tool_calls"] = tool_calls
        rendered_messages.append(role_message)

    return rendered_messages


def render_output(
    output: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Parses the assistant message from an observation using the provided function configuration.
    """
    content: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    if base_function.type == "json":
        return {"role": "assistant", "content": output["raw"]}
    elif base_function.type == "chat":
        for content_block in output:
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
        raise ValueError(f"Unsupported function type: {base_function.type}")

    # Once we finish collecting all blocks, create one assistant message.
    output_message: Dict[str, Any] = {"role": "assistant"}
    if content:
        output_message["content"] = content
    if tool_calls:
        output_message["tool_calls"] = tool_calls

    return output_message


def sample_to_openai_messages(sample) -> List[Dict[str, Any]]:
    function_input = json.loads(sample["input"])

    rendered_messages = []

    # Add the system message to the rendered messages
    # If there is data passed in or a system template there must be a system message
    system = function_input.get("system", {})
    if len(system) > 0 or base_variant.system_template:
        if base_variant.system_template:
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
        rendered_messages.extend(render_message(message))

    # Add the output to the messages
    output = json.loads(sample["output"])
    rendered_output = render_output(output)
    if rendered_output is None:
        # `render_output` will return None if the output contains an unknown or unsupported content block.
        # The entire example is dropped if this is the case.
        return None
    rendered_messages.append(rendered_output)

    return {"messages": rendered_messages}


df["conversational_messages"] = df.apply(sample_to_openai_messages, axis=1)

# Drop null rows
df = df[df["conversational_messages"].notna()]

df.head()

# %% [markdown]
# Split the data into training and evaluation sets.
# The training set is used to generate candidate demonstrations.
# The evaluation set is used by the judge to score the quality of the generated prompt.

# %%
# Get unique episode_ids
unique_episode_ids = df["episode_id"].unique()

# Shuffle the unique episode_ids
np.random.seed(42)
np.random.shuffle(unique_episode_ids)

# Calculate the split index for episode_ids
split_index = int(len(unique_episode_ids) * (1 - EVAL_FRACTION))

# Split the episode_ids into training and validation sets
train_episode_ids = unique_episode_ids[:split_index]
val_episode_ids = unique_episode_ids[split_index:]

# Create training and validation DataFrames based on episode_ids
train_df = df[df["episode_id"].isin(train_episode_ids)]
eval_df = df[df["episode_id"].isin(val_episode_ids)]

print(f"Training set size: {len(train_df)}")
print(f"Evaluation set size: {len(eval_df)}")
print(f"Actual evaluation fraction: {len(eval_df) / len(df):.2f}")

# %% [markdown]
# ## Generate Candidate Instructions
#
# Given the function's system template as an example, generate a set of candidate instructions to optimize the prompt over.

# %%
example_instructions = base_variant.system_template

if base_function.system_schema is not None:
    example_schema = base_function.system_schema.model_json_schema()
else:
    example_schema = None

responses = await tqdm_asyncio.gather(
    *[
        get_instructions(
            client=mipro_client,
            example_instructions=example_instructions,
            example_schema=example_schema,
            semaphore=semaphore,
        )
        for _ in range(NUM_CANDIDATE_INSTRUCTIONS)
    ]
)

candidate_instructions = [example_instructions]
for response in responses:
    if response is None:
        continue
    candidate_instructions.append(response.output.parsed["instructions"])


# %% [markdown]
# ## Generate Candidate Demonstrations
#
# Given the training set, generate a set of candidate demonstrations to optimize the prompt over.


# %%
def generate_demonstrations(
    df: pd.DataFrame,
    max_examples_per_demonstration: int,
    seed: int = 42,
) -> str:
    sample = df.sample(
        n=max_examples_per_demonstration, replace=False, random_state=seed
    )
    demonstrations = []
    for _, row in sample.iterrows():  # type: ignore
        demonstrations.append(row["conversational_messages"])
    return demonstrations


# %%
candidate_demonstrations = [
    generate_demonstrations(
        df=train_df,
        max_examples_per_demonstration=MAX_EXAMPLES_PER_DEMONSTRATION,
        seed=seed,
    )
    for seed in range(NUM_CANDIDATE_DEMONSTRATIONS)
]

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

# %%
# Initialize online statistics
num_instructions = len(candidate_instructions)
num_demonstrations = len(candidate_demonstrations)


# %%
def format_response(response: Optional[InferenceResponse]) -> str:
    if response is None:
        return ""
    if isinstance(response, JsonInferenceResponse):
        return str(response.output.parsed)
    else:
        content = response.content
        assert len(content) == 1  # TODO: Handle multiple content blocks
        if isinstance(content[0], Text):
            return content[0].text
        elif isinstance(content[0], RawText):
            return content[0].value
        else:
            raise ValueError(f"Unsupported content type: {type(content[0])}")


async def objective(trial: optuna.Trial):
    # Sample an instruction and a demonstration set
    instruction_index = trial.suggest_categorical(
        "instruction_index", range(num_instructions)
    )
    demonstration_index = trial.suggest_categorical(
        "demonstration_index", range(num_demonstrations)
    )
    # Format the candidate prompt
    candidate_prompt = env.render_template(
        "candidate",
        instructions=candidate_instructions[instruction_index],
        demonstrations=candidate_demonstrations[demonstration_index],
    )
    # Create a new variant with the candidate prompt
    candidate_variant_name = f"{instruction_index}_{demonstration_index}"
    candidate_config = deepcopy(base_config)
    candidate_config.functions[FUNCTION_NAME].variants[candidate_variant_name] = (
        deepcopy(base_variant)
    )
    candidate_config.functions[FUNCTION_NAME].variants[
        candidate_variant_name
    ].system_template = candidate_prompt
    candidate_config.functions[FUNCTION_NAME].variants[
        candidate_variant_name
    ].name = candidate_variant_name
    # Write the new config to a temporary directory
    tmp_config_dir = candidate_config.write()
    # Build a new client with the new config
    target_client = await AsyncTensorZeroGateway.build_embedded(
        config_file=str(tmp_config_dir / "tensorzero.toml"),
        clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
    )
    # Asynchronously generate answers for each query in the evaluation set
    responses = await tqdm_asyncio.gather(
        *[
            candidate_inference(
                client=target_client,
                function_name=FUNCTION_NAME,
                input=json.loads(input_args),
                variant_name=candidate_variant_name,
                semaphore=semaphore,
            )
            for input_args in eval_df["input"]
        ]
    )

    # Score the responses using the judge
    judge_responses = await tqdm_asyncio.gather(
        *[
            judge_answer(
                client=mipro_client,
                task_description=TASK_DESCRIPTION,
                metric_properties=METRIC_PROPERTIES,
                prediction=format_response(response) if response is not None else "",
                ground_truth=str(ground_truth),
                semaphore=semaphore,
            )
            for response, ground_truth in zip(responses, eval_df["output"])
        ]
    )

    # Aggregate the scores
    scores = []
    for response in judge_responses:
        if response is not None:
            if response.output.parsed is not None:
                scores.append(response.output.parsed["score"])

    # Return the mean score
    return np.mean(scores)


# %% [markdown]
# ### Random Search
#
# We start by sampling a random instruction and demonstration at each iteration in the optimization loop.

# %%
study_random = optuna.create_study(
    sampler=optuna.samplers.RandomSampler(seed=SEED), direction=OPTIMIZER_DIRECTION
)

for iteration in range(MAX_ITERATIONS):
    trial = study_random.ask()

    value = await objective(trial)
    print(f"Iteration {iteration + 1}: {value}")

    frozen_trial = study_random.tell(trial, value)
    study_random._log_completed_trial(frozen_trial)

# %% [markdown]
# ### Tree-structured Parzen Estimator
# Following the MIPRO paper, we use a tree-structured parzen estimator (TPE) to sample the next instruction and demonstration pair to evaluate.

# %%
study_tpe = optuna.create_study(
    sampler=TPESampler(seed=SEED), direction=OPTIMIZER_DIRECTION
)

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

# %%
optimized_system_template = env.render_template(
    "candidate",
    instructions=candidate_instructions[study_tpe.best_params["instruction_index"]],
    demonstrations=candidate_demonstrations[
        study_tpe.best_params["demonstration_index"]
    ],
)
print(optimized_system_template)

# %% [markdown]
# You can save the optimized configuration file tree.

# %%
OUTPUT_DIR = "tmp"  # Set to a local path to save the optimized config

optimized_variant_name = "mipro_optimized"
optimized_config = deepcopy(base_config)
optimized_config.functions[FUNCTION_NAME].variants[optimized_variant_name] = deepcopy(
    base_variant
)
optimized_config.functions[FUNCTION_NAME].variants[
    optimized_variant_name
].system_template = optimized_system_template
optimized_config.functions[FUNCTION_NAME].variants[
    optimized_variant_name
].name = optimized_variant_name
# write the new config to a temporary directory
optimized_config_dir = optimized_config.write(base_dir=OUTPUT_DIR)

# %% [markdown]
# ## Conclusion
#
# By following this notebook, you can systematically refine prompts for better performance.
# The optimized prompt can be saved and used in production by updating the function's system template configuration.
#
