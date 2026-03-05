//! Serializable types for the durable GEPA task tool.
//!
//! Checkpoint types must not store full datapoints, inference results, or score maps.
//! That data already lives in ClickHouse/Postgres and can be reloaded by ID.
//! Configs, datapoint IDs, and other lightweight state are fine.

use std::collections::{BTreeMap, HashMap};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_core::config::MetricConfigOptimize;
use tensorzero_core::evaluations::EvaluatorConfig;
use tensorzero_core::optimization::gepa::GepaEvaluatorStats;
use tensorzero_core::variant::chat_completion::UninitializedChatCompletionConfig;
use uuid::Uuid;

use crate::gepa::evaluate::{EvaluatorName, VariantName, VariantScores};
use crate::gepa::pareto::VariantScoresMap;
use crate::gepa::validate::SerializableFunctionContext;

// ── Tool params & output (visible to LLM / used as spawn params) ────────

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

// ── Setup step types ────────────────────────────────────────────────────

/// Result of the "setup" step — stored once as step output.
/// Subsequent steps use lightweight params instead of cloning this.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetupResult {
    pub function_context: SerializableFunctionContext,
    pub original_variants: HashMap<String, UninitializedChatCompletionConfig>,
    pub val_dataset_name: String,
    pub val_datapoint_ids: Vec<Uuid>,
    pub evaluator_configs: HashMap<String, EvaluatorConfig>,
    pub run_id: Uuid,
    pub gepa_config: ResolvedGEPAConfig,
}

/// Resolved GEPA config fields (extracted from GepaToolParams with defaults applied).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedGEPAConfig {
    pub function_name: String,
    pub evaluation_name: String,
    pub initial_variants: Option<Vec<String>>,
    pub variant_prefix: Option<String>,
    pub batch_size: usize,
    pub max_iterations: u32,
    pub max_concurrency: u32,
    pub analysis_model: String,
    pub mutation_model: String,
    pub seed: Option<u32>,
    pub include_inference_for_mutation: bool,
    pub max_tokens: Option<u32>,
}

// ── Lightweight step params (avoid cloning full SetupResult) ────────────

/// Params for evaluating a single variant on a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalStepParams {
    pub evaluation_name: String,
    pub dataset_name: String,
    pub variant_name: VariantName,
    pub variant_config: UninitializedChatCompletionConfig,
    pub max_concurrency: u32,
}

/// Params for the initial evaluation step (multiple variants).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitEvalStepParams {
    pub evaluation_name: String,
    pub val_dataset_name: String,
    pub variants: HashMap<String, UninitializedChatCompletionConfig>,
    pub max_concurrency: u32,
}

/// Params for the mutation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutateStepParams {
    pub function_context: SerializableFunctionContext,
    pub gepa_config: ResolvedGEPAConfig,
    pub parent_name: VariantName,
    pub parent_config: UninitializedChatCompletionConfig,
    pub iteration: u32,
}

// ── Step result types ───────────────────────────────────────────────────

/// Result of evaluating a variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub variant_name: VariantName,
    pub scores: VariantScores,
    pub stats: HashMap<EvaluatorName, evaluations::EvaluatorStats>,
}

/// Result of a mutation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationResult {
    pub child_name: VariantName,
    pub child_config: UninitializedChatCompletionConfig,
}

// ── Pareto frontier checkpoint ──────────────────────────────────────────

/// Serializable snapshot of ParetoFrontier state.
///
/// Contains variant configs, per-datapoint scores, frequencies, layout info, and RNG seed.
/// The `objective_vector_cache` is recomputed on restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoCheckpoint {
    pub variant_configs: HashMap<VariantName, UninitializedChatCompletionConfig>,
    pub variant_scores_map: VariantScoresMap,
    pub variant_frequencies: HashMap<VariantName, usize>,
    pub datapoint_ids: Vec<Uuid>,
    pub optimize_directions: BTreeMap<EvaluatorName, MetricConfigOptimize>,
    /// RNG seed to reconstruct deterministic sampling state.
    pub rng_seed: u64,
}
