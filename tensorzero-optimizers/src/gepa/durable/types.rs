//! Serializable types for the durable GEPA task tool.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_core::optimization::gepa::GepaEvaluatorStats;
use tensorzero_core::variant::chat_completion::UninitializedChatCompletionConfig;

/// Parameters for the durable GEPA tool (visible to LLM / used as spawn params).
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct GepaToolParams {
    /// Name of the function being optimized.
    pub function_name: String,
    /// Single dataset name (auto-split 50/50).
    /// Mutually exclusive with `train_dataset_name`/`val_dataset_name`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_name: Option<String>,
    /// Training dataset name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub train_dataset_name: Option<String>,
    /// Validation dataset name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub val_dataset_name: Option<String>,
    /// Named evaluation. Mutually exclusive with `evaluators`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_name: Option<String>,
    /// Inline list of evaluator names. Mutually exclusive with `evaluation_name`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluators: Option<Vec<String>>,
    /// Model for analysis (e.g., "anthropic::claude-sonnet-4-5").
    pub analysis_model: String,
    /// Model for mutation.
    pub mutation_model: String,
    /// Optional list of variant names to initialize GEPA with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_variants: Option<Vec<String>>,
    /// Maximum number of training iterations.
    pub max_iterations: u32,
    /// Prefix for the name of the new optimized variants.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_prefix: Option<String>,
    /// Number of training samples to analyze per iteration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
    /// Optional random seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    /// Whether to include inference input and output in Analysis for mutation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_inference_for_mutation: Option<bool>,
    /// Maximum number of tokens for analysis and mutation model calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Maximum number of concurrent inference calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_concurrency: Option<u32>,
}

/// Output of the durable GEPA tool.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct GepaToolOutput {
    /// Map of variant_name to its configuration (the Pareto frontier).
    pub variants: HashMap<String, UninitializedChatCompletionConfig>,
    /// Map of variant_name to { evaluator_name to stats }.
    pub statistics: HashMap<String, HashMap<String, GepaEvaluatorStats>>,
}
