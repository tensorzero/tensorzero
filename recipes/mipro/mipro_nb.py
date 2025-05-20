# %%
# type: ignore
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

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
    comparison_operator = ">=" if filter_metric.optimize == "maximize" else "<="

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

if FILTER_METRIC_NAME != "demonstration":
    df.value = df.output

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

env = Environment(templates=templates)


# %% [markdown]
# Render the messages in the input and demonstration columns.


# %%
def render_message(content: List[Dict[str, Any]], role: str) -> str:
    assert role in ["user", "assistant"], f"Invalid role: {role}"

    if len(content) != 1:
        raise ValueError(f"Message must have exactly one content block: {content}")

    if role == "user":
        output = "ENVIRONMENT:\n"
    else:
        output = "AGENT:\n"

    if content[0]["type"] == "text":
        value = content[0]["value"]
        if isinstance(value, str):
            output += value
        else:
            value = env.render_template(role, **value)  # type: ignore
            assert isinstance(value, str)
            output += value
    elif content[0]["type"] == "tool_call":
        del content[0]["id"]
        del content[0]["type"]
        output += f"Tool call: {json.dumps(content[0])}"
    elif content[0]["type"] == "tool_result":
        output += f"Tool result: {content[0]['result']}"
    else:
        raise ValueError(
            f"Content block must be of type text, tool_call, or tool_result: {content}"
        )

    return output


def format_input(sample):
    function_input = json.loads(sample["input"])
    rendered_message = ""
    for message in function_input["messages"]:
        rendered_message += render_message(message["content"], message["role"])
        rendered_message += "\n"
    return rendered_message


def format_output(sample):
    output = json.loads(sample["value"])
    if base_function.type == "chat":
        if len(output) != 1:
            raise ValueError(f"Output {output} must have exactly one content block.")
        if output[0]["type"] == "text":
            return output[0]["text"]
        elif output[0]["type"] == "tool_call":
            del output[0]["raw_arguments"]
            del output[0]["raw_name"]
            del output[0]["type"]
            return f"Tool call: {json.dumps(output[0])}"
        elif output[0]["type"] == "tool_result":
            return json.dumps(output[0])
        else:
            raise ValueError(f"Output {output} must be a text block.")
    elif base_function.type == "json":
        return output["raw"]
    else:
        raise ValueError(f"Unsupported function type: {base_function.type}")


def format_system_args(sample):
    function_input = json.loads(sample["input"])
    if "system" in function_input:
        return function_input["system"]
    else:
        return ""


df["input_str"] = df.apply(format_input, axis=1)
df["value_str"] = df.apply(format_output, axis=1)
df["system_args"] = df.apply(format_system_args, axis=1)
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
    input_col: str,
    output_col: str,
    system_col: str,
    seed: int = 42,
) -> str:
    sample = df.sample(
        n=max_examples_per_demonstration, replace=False, random_state=seed
    )
    demonstrations = ""
    demonstration_number = 1
    for _, row in sample.iterrows():  # type: ignore
        demonstrations += f"DEMONSTRATION {demonstration_number}:\n"
        demonstration_number += 1
        if row[system_col] is not None and row[system_col] != "":
            demonstrations += f"SYSTEM:\n{row[system_col]}\n"
        demonstrations += f"{row[input_col]}AGENT:\n{row[output_col]}\n\n"
    return demonstrations


# %%
candidate_demonstrations = [
    generate_demonstrations(
        df=train_df,
        max_examples_per_demonstration=MAX_EXAMPLES_PER_DEMONSTRATION,
        input_col="input_str",
        output_col="value_str",
        system_col="system_args",
        seed=seed,
    )
    for seed in range(NUM_CANDIDATE_DEMONSTRATIONS)
]

# %% [markdown]
# ## Optimize the Prompt
#
# ### Define the optimization objective

# %%
# Initialize online statistics
num_instructions = len(candidate_instructions)
num_demonstrations = len(candidate_demonstrations)


# %%
def format_system_template(instructions: str, demonstrations: str) -> str:
    return f"# Instructions:\n\n{instructions}\n\n# Demonstrations:\n\n{demonstrations}"


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
    candidate_prompt = format_system_template(
        candidate_instructions[instruction_index],
        candidate_demonstrations[demonstration_index],
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
            for response, ground_truth in zip(responses, eval_df["value_str"])
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
optimized_system_template = format_system_template(
    instructions=candidate_instructions[study_tpe.best_params["instruction_index"]],
    demonstrations=candidate_demonstrations[
        study_tpe.best_params["demonstration_index"]
    ],
)
print(optimized_system_template)

# %% [markdown]
# You can save the optimized configuration file tree.

# %%
OUTPUT_DIR = None  # Set to a local path to save the optimized config

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
