# %% [markdown]
# # Understanding Failure Modes with AI Assisted Root Cause Analysis
#
# This recipe allows TensorZero users to analyze failure modes of their LLM application with help from Root Cause Analysis AI.
# Since TensorZero automatically logs all inferences and feedback, it is straightforward to analyze the failure modes of your LLM application on your data.

# %% [markdown]
# To get started:
#
# - Set the `TENSORZERO_CLICKHOUSE_URL` environment variable. For example: `TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero"`
# - Set the `OPENAI_API_KEY` environment variable.
# - Update the following parameters:

# %%
CONFIG_PATH = "../../examples/data-extraction-ner/config/tensorzero.toml"

FUNCTION_NAME = "extract_entities"

METRIC_NAME = "jaccard_similarity"

# The name of the variant to analyze root cause analysis
SUBJECT_VARIANT_NAME = "gpt_4o_mini"

# Optional list of tools available if your function supports them.
# Each entry is formatted as as a dictionary.
# {"name": "<The tool's identifier.>", "description": "<A brief description of what the tool does.>"}
# These will be passed to the assistant to aid in root cause analysis.
TOOLS_AVAILABLE = []

# If the metric is a float metric, you can set the threshold to define a failure and filter the data
FLOAT_METRIC_THRESHOLD = 0.5

# Maximum number of samples to use for root cause analysis
MAX_SAMPLES = 100_000

# The name of the variant to use for root cause and failure mode analysis
ANALYSIS_VARIANT_NAME = "gpt-5"

# Embedding model to use for root cause and failure mode analysis
EMBEDDING_MODEL = "text-embedding-3-small"

# Number of root cause clusters to use
N_CLUSTERS = 3

# %%
import os
import sys

tensorzero_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if tensorzero_path not in sys.path:
    sys.path.append(tensorzero_path)

# %%
import asyncio
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from tensorzero import (
    AsyncTensorZeroGateway,
    FloatMetricFilter,
    TensorZeroGateway,
)
from tqdm.asyncio import tqdm_asyncio
from utils import generate_root_causes

# %%
data_client = TensorZeroGateway.build_embedded(
    config_file=CONFIG_PATH,
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
    timeout=15,
)

# %%
comparison_operator = "<="
metric_node = FloatMetricFilter(
    metric_name=METRIC_NAME,
    value=FLOAT_METRIC_THRESHOLD,
    comparison_operator=comparison_operator,
)

# from tensorzero import BooleanMetricFilter

# metric_node = BooleanMetricFilter(
#     metric_name=METRIC_NAME,
#     value=False  #
# )

metric_node

# %%
stored_inferences = data_client.experimental_list_inferences(
    function_name=FUNCTION_NAME,
    variant_name=SUBJECT_VARIANT_NAME,
    output_source="inference",  # could also be "demonstration"
    filters=metric_node,
    limit=MAX_SAMPLES,
)

# %%
rendered_samples = data_client.experimental_render_inferences(
    stored_inferences=stored_inferences,
    variants={FUNCTION_NAME: SUBJECT_VARIANT_NAME},
)

# %%
len(rendered_samples)

# %% [markdown]
# ## Root Cause Analysis

# %% [markdown]
# Generate a list of root causes for each failure.

# %%
root_cause_client = await AsyncTensorZeroGateway.build_embedded(
    config_file="config/tensorzero.toml",
    clickhouse_url=os.environ["TENSORZERO_CLICKHOUSE_URL"],
)
semaphore = asyncio.Semaphore(40)

# %%
tasks = [
    generate_root_causes(
        gateway=root_cause_client,
        rendered_sample=rendered_sample,
        variant_name=ANALYSIS_VARIANT_NAME,
        semaphore=semaphore,
        dryrun=True,
    )
    for rendered_sample in rendered_samples
]

root_causes = await tqdm_asyncio.gather(*tasks)

# %%
root_causes_concat = [
    "\n".join(root_cause) for root_cause in root_causes if root_cause is not None
]

# %% [markdown]
# ## Failure Mode Analysis

# %% [markdown]
# **Step 1**: Get a vector representation of each root cause.

# %%
openai_client = AsyncOpenAI()


# %%
async def get_embedding(text: str) -> Optional[list[float]]:
    try:
        async with semaphore:
            response = await openai_client.embeddings.create(
                input=text, model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


# %%
tasks = [get_embedding(root_cause) for root_cause in root_causes_concat]

embeddings = await tqdm_asyncio.gather(*tasks, desc="Embedding inputs")
embeddings = np.array(embeddings)

# %% [markdown]
# **Step 2**: Find failure modes by clustering the root cause embeddings

# %%
# Use Bayesian GMM instead of KMeans
bgmm = BayesianGaussianMixture(
    n_components=N_CLUSTERS,
    covariance_type="full",
    weight_concentration_prior_type="dirichlet_process",
    random_state=42,
)
bgmm.fit(embeddings)
labels = bgmm.predict(embeddings)

# Assign root cause labels to DataFrame
df = pd.DataFrame(data={"root_cause": root_causes_concat, "cluster": labels})

# %%
df = pd.DataFrame(data={"root_cause": root_causes_concat, "cluster": labels})

# %%
# Ensure embeddings is a NumPy array
embeddings = np.array(embeddings)

# Fit PCA and transform
pca = PCA(n_components=2)
vis_dims2 = pca.fit_transform(embeddings)

# Create DataFrame for plotting
pca_df = pd.DataFrame(
    {"PC1": vis_dims2[:, 0], "PC2": vis_dims2[:, 1], "failure_mode": df["cluster"]}
)

# Compute cluster centroids
centroids_df = pca_df.groupby("failure_mode")[["PC1", "PC2"]].mean().reset_index()
centroids_df["label"] = centroids_df["failure_mode"].apply(lambda c: f"Cluster {c}")

# Scatter plot of points
points_chart = (
    alt.Chart(pca_df)
    .mark_circle(opacity=0.3, size=60)
    .encode(
        x=alt.X("PC1", title="Principal Component 1"),
        y=alt.Y("PC2", title="Principal Component 2"),
        color=alt.Color("failure_mode:N", title="Failure Mode"),
        tooltip=["failure_mode"],
    )
)

# Cross markers for centroids
centroids_chart = (
    alt.Chart(centroids_df)
    .mark_point(filled=True, size=100, shape="cross")
    .encode(x="PC1", y="PC2", color=alt.Color("failure_mode:N"), tooltip=["label"])
)

# Combine
(points_chart + centroids_chart).properties(
    title="Failure Modes visualized using Principal Component Analysis (PCA)"
)

# %% [markdown]
# **Step 3**: Summarize the failure modes in natural language.

# %%
# summaries = []
# for i in range(N_CLUSTERS):
#     print(f"Cluster {i}:", end=" ")

#     # Sample for summarization
#     root_causes_sample = df[df.cluster == i].root_cause.to_list()
#     # break

#     gateway_input = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "arguments": {
#                             "root_causes": root_causes_sample,
#                             "system_template": system_template,
#                         },
#                     }
#                 ],
#             }
#         ]
#     }
#     response = await gateway.inference(
#         input=gateway_input,
#         function_name="summarize_failure_modes",
#         variant_name=ANALYSIS_VARIANT_NAME,
#     )
#     summary = response.output.parsed["summary"]

#     # Sample for representative examples (different random seed to avoid duplication)
#     examples = df[df.cluster == i]["rendered_input"].to_list()
#     examples = [example["messages"][1:] for example in examples]
#     summaries.append(summary)
#     pprint(f"\nSummary: {summary}")

#     # Show example root causes
#     print("\nRepresentative examples:")
#     for ex in examples[:10]:
#         print(f" - {ex}")
#     print("-" * 100)

# %% [markdown]
# You're all set!
#
# We encourage you to experiment with other parameters (e.g. `N_CLUSTERS`, `EMBEDDING_MODEL`, or the clustering algorithm).
#
# We use OpenAI o4-mini for the root cause and failure mode analysis.
# You can try using other models by adding variants to `config/tensorzero.toml` and updating `ANALYSIS_VARIANT_NAME`.
