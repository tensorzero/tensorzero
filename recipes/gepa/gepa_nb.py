# %%
# type: ignore

# %% [markdown]
# # GEPA Optimization
#
# This recipe allows TensorZero users to optimize prompts using [GEPA (Genetic-Pareto prompt optimization)](https://arxiv.org/abs/2507.19457).
# GEPA evolves prompts through an iterative process of evaluation, analysis, and mutation to improve performance.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to optimize prompts using your own data.
#

# %% [markdown]
# To get started:
#
# - Set the `TENSORZERO_CLICKHOUSE_URL` environment variable. For example: `TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero"`
# - Set the `OPENAI_API_KEY` environment variable (for the analysis and mutation models).
# - Update the following parameters:
#

# %%
CONFIG_PATH = "../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

METRIC_NAME = "jaccard_similarity"

# The name of the variant to use to grab the templates used for rendering samples
TEMPLATE_VARIANT_NAME = "gpt_4o_mini"

# If the metric is a float metric, you can set the threshold to filter the data
FLOAT_METRIC_THRESHOLD = 0.9

# Fraction of the data to use for validation
VAL_FRACTION = 0.2

# Maximum number of samples to use for optimization
MAX_SAMPLES = 100_000

# GEPA-specific configuration
EVALUATION_NAME = "extract_entities_eval"

# Models to use for analyzing inferences and generating prompt mutations
ANALYSIS_MODEL = "openai::gpt-5"
MUTATION_MODEL = "openai::gpt-5"

# Initial variants to start the optimization from
INITIAL_VARIANTS = ["gpt_4o_mini"]

# Number of evolution iterations (each iteration evaluates, analyzes, and mutates variants)
MAX_ITERATIONS = 10

# Maximum number of concurrent inference requests during evaluation
MAX_CONCURRENCY = 50

# %%
import os
import sys

tensorzero_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if tensorzero_path not in sys.path:
    sys.path.append(tensorzero_path)

# %%
import time

from IPython.display import clear_output
from tensorzero import (
    FloatMetricFilter,
    GEPAConfig,
    OptimizationJobStatus,
    TensorZeroGateway,
)

from recipes.util import train_val_split

# %% [markdown]
# Initialize the TensorZero client
#

# %%
tensorzero_client = TensorZeroGateway.build_embedded(
    config_file=CONFIG_PATH,
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
    timeout=3000,
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
stored_inferences = tensorzero_client.experimental_list_inferences(
    function_name=FUNCTION_NAME,
    variant_name=None,
    output_source="inference",  # could also be "demonstration"
    filters=metric_node,
    limit=MAX_SAMPLES,
)

# %% [markdown]
# Render the inputs using the templates.
#

# %%
rendered_samples = tensorzero_client.experimental_render_samples(
    stored_samples=stored_inferences,
    variants={FUNCTION_NAME: TEMPLATE_VARIANT_NAME},
)

# %% [markdown]
# Split the data into training and validation sets.
#
# **Note:** GEPA requires validation samples for Pareto frontier filtering.

# %%
train_samples, val_samples = train_val_split(
    rendered_samples,
    val_size=VAL_FRACTION,
    last_inference_only=True,
)

# %% [markdown]
# Configure and launch the GEPA optimization job
#
# **Note:** The optimization runs synchronously and may take around 15 minutes to complete.

# %%
optimization_config = GEPAConfig(
    function_name=FUNCTION_NAME,
    evaluation_name=EVALUATION_NAME,
    analysis_model=ANALYSIS_MODEL,
    mutation_model=MUTATION_MODEL,
    initial_variants=INITIAL_VARIANTS,
    max_iterations=MAX_ITERATIONS,
    max_concurrency=MAX_CONCURRENCY,
)

job_handle = tensorzero_client.experimental_launch_optimization(
    train_samples=train_samples,
    val_samples=val_samples,
    optimization_config=optimization_config,
)

# %% [markdown]
# Wait for the GEPA optimization job to complete.

# %%
while True:
    clear_output(wait=True)

    try:
        job_info = tensorzero_client.experimental_poll_optimization(job_handle=job_handle)
        print(job_info)
        if job_info.status in (
            OptimizationJobStatus.Completed,
            OptimizationJobStatus.Failed,
        ):
            break
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(10)

# %% [markdown]
# Once the GEPA optimization is complete, you can add the optimized variant(s) to your config file.
#
# GEPA returns a dictionary of variant configurations that achieved Pareto-optimal performance.
# The templates below can be used when creating new variants in your `tensorzero.toml`.

# %%
# GEPA returns variant configurations, not a model name
variant_configs = job_info.output["content"]

for variant_name, variant_config in variant_configs.items():
    print(f"\n# Optimized variant: {variant_name}")
    for template_name, template in variant_config["templates"].items():
        print(f"## '{template_name}' template:")
        print(template["path"]["__data"])

# %% [markdown]
# To use the optimized variant:
#
# 1. Save the template content above to a new file (e.g., `functions/extract_entities/gepa_optimized/system_template.minijinja`)
# 2. Add a new variant to your `tensorzero.toml` pointing to the new template
# 3. Adjust the variant weight to enable a gradual rollout

# %% [markdown]
# ## Tips for Further Optimization
#
# - **Increase iterations**: Set `MAX_ITERATIONS` higher (e.g., 20-50) to allow more template evolution
# - **Try different analysis/mutation models**: Experiment with `anthropic::claude-opus-4-5` or `google::gemini-2.5-pro` for potentially different optimization strategies
