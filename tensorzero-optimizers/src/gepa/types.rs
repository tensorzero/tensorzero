//! Serializable types for durable GEPA execution.
//!
//! These types support per-step checkpointing so that the durable task framework
//! can resume GEPA optimization from the last completed step.

use std::collections::{BTreeMap, HashMap};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use tensorzero_core::{
    config::MetricConfigOptimize, optimization::gepa::UninitializedGEPAConfig,
    stored_inference::RenderedSample, variant::chat_completion::UninitializedChatCompletionConfig,
};

use evaluations::stats::{EvaluationInfo, EvaluatorStats};

use crate::gepa::analyze::Analysis;
use crate::gepa::evaluate::{DatapointId, EvaluatorName, VariantName, VariantScores};

/// Parameters for a durable GEPA optimization task.
///
/// Serialized as JSON and passed to `spawn_tool_by_name("gepa_optimization", ...)`.
/// Only contains LLM-level configuration; infrastructure data (examples) is in `GepaSideInfo`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GepaToolParams {
    pub gepa_config: UninitializedGEPAConfig,
}

/// Side info for durable GEPA tasks.
///
/// Contains data that is passed at spawn time but is not LLM-generated.
/// Uses its own type (not `AutopilotSideInfo`) because GEPA is a standalone
/// tool that bypasses the autopilot result-publishing wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaSideInfo {
    pub train_examples: Vec<RenderedSample>,
    pub val_examples: Vec<RenderedSample>,
}

/// Output of a completed GEPA optimization task.
///
/// Stored as the `completed_payload` in the durable task result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaToolOutput {
    pub variants: HashMap<String, UninitializedChatCompletionConfig>,
}

/// Serializable snapshot of Pareto frontier state for checkpointing between durable steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoCheckpoint {
    pub variant_configs: HashMap<VariantName, UninitializedChatCompletionConfig>,
    pub variant_scores_map: HashMap<VariantName, VariantScores>,
    pub variant_frequencies: HashMap<VariantName, usize>,
    pub datapoint_ids: Vec<DatapointId>,
    pub optimize_directions: BTreeMap<EvaluatorName, MetricConfigOptimize>,
    pub seed: Option<u64>,
}

/// Result of the GEPA setup step.
///
/// Contains the initial Pareto frontier checkpoint and all context needed
/// for subsequent iteration and cleanup steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaSetupResult {
    pub checkpoint: ParetoCheckpoint,
    pub train_examples: Vec<RenderedSample>,
    pub max_iterations: u32,
    pub val_dataset_name: String,
    pub temporary_datasets: Vec<String>,
    pub original_variant_names: Vec<String>,
    pub gepa_config: UninitializedGEPAConfig,
    pub per_variant_concurrency: usize,
    pub run_id: Uuid,
}

/// Parameters for a single GEPA iteration step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaIterationParams {
    pub checkpoint: ParetoCheckpoint,
    pub train_examples: Vec<RenderedSample>,
    pub iteration: usize,
    pub gepa_config: UninitializedGEPAConfig,
    pub val_dataset_name: String,
    pub temporary_datasets: Vec<String>,
    pub per_variant_concurrency: usize,
    pub run_id: Uuid,
}

/// Result of a single GEPA iteration step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaIterationResult {
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
}

/// Parameters for the GEPA cleanup step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaCleanupParams {
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
    pub original_variant_names: Vec<String>,
}

/// Serializable variant reference for passing between sub-steps.
///
/// Equivalent to `GEPAVariant` but with `Serialize`/`Deserialize` for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedVariant {
    pub name: VariantName,
    pub config: UninitializedChatCompletionConfig,
}

/// Parameters for the GEPA setup step (replaces individual args).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaSetupParams {
    pub gepa_config: UninitializedGEPAConfig,
    pub train_examples: Vec<RenderedSample>,
    pub val_examples: Vec<RenderedSample>,
}

/// Shared data for `SkipIteration` variants across all sub-step results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaSkipIteration {
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
}

// ============================================================================
// Sub-step types for breaking GEPA iteration into checkpointable pieces
// ============================================================================

/// Sub-step 1: Sample parent and create minibatch dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaSampleParams {
    pub checkpoint: ParetoCheckpoint,
    pub train_examples: Vec<RenderedSample>,
    pub iteration: usize,
    pub gepa_config: UninitializedGEPAConfig,
    pub val_dataset_name: String,
    pub temporary_datasets: Vec<String>,
    pub per_variant_concurrency: usize,
    pub run_id: Uuid,
}

/// Data for the `Continue` variant of [`GepaSampleResult`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaSampleContinue {
    pub parent: SelectedVariant,
    pub mutation_dataset_name: String,
    pub temporary_datasets: Vec<String>,
    pub checkpoint: ParetoCheckpoint,
}

/// Result of the sample sub-step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GepaSampleResult {
    Continue(Box<GepaSampleContinue>),
    SkipIteration(Box<GepaSkipIteration>),
}

/// Sub-step 2: Evaluate parent variant on minibatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaEvalParentParams {
    pub parent: SelectedVariant,
    pub mutation_dataset_name: String,
    pub gepa_config: UninitializedGEPAConfig,
    pub iteration: usize,
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
    pub val_dataset_name: String,
    pub per_variant_concurrency: usize,
}

/// Data for the `Continue` variant of [`GepaEvalParentResult`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaEvalParentContinue {
    pub parent: SelectedVariant,
    pub parent_evaluation_infos: Vec<EvaluationInfo>,
    pub parent_evaluation_stats: HashMap<EvaluatorName, EvaluatorStats>,
    pub mutation_dataset_name: String,
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
    pub val_dataset_name: String,
    pub per_variant_concurrency: usize,
}

/// Result of the eval-parent sub-step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GepaEvalParentResult {
    Continue(Box<GepaEvalParentContinue>),
    SkipIteration(Box<GepaSkipIteration>),
}

/// Sub-step 3: Analyze parent inferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaAnalyzeParams {
    pub parent: SelectedVariant,
    pub parent_evaluation_infos: Vec<EvaluationInfo>,
    pub parent_evaluation_stats: HashMap<EvaluatorName, EvaluatorStats>,
    pub mutation_dataset_name: String,
    pub gepa_config: UninitializedGEPAConfig,
    pub iteration: usize,
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
    pub val_dataset_name: String,
    pub per_variant_concurrency: usize,
}

/// Data for the `Continue` variant of [`GepaAnalyzeResult`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaAnalyzeContinue {
    pub parent: SelectedVariant,
    pub parent_analyses: Vec<Analysis>,
    pub parent_evaluation_stats: HashMap<EvaluatorName, EvaluatorStats>,
    pub mutation_dataset_name: String,
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
    pub val_dataset_name: String,
    pub per_variant_concurrency: usize,
}

/// Result of the analyze sub-step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GepaAnalyzeResult {
    Continue(Box<GepaAnalyzeContinue>),
    SkipIteration(Box<GepaSkipIteration>),
}

/// Sub-step 4: Mutate parent to produce child variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaMutateParams {
    pub parent: SelectedVariant,
    pub parent_analyses: Vec<Analysis>,
    pub parent_evaluation_stats: HashMap<EvaluatorName, EvaluatorStats>,
    pub mutation_dataset_name: String,
    pub gepa_config: UninitializedGEPAConfig,
    pub iteration: usize,
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
    pub val_dataset_name: String,
    pub per_variant_concurrency: usize,
}

/// Data for the `Continue` variant of [`GepaMutateResult`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaMutateContinue {
    pub child: SelectedVariant,
    pub parent_evaluation_stats: HashMap<EvaluatorName, EvaluatorStats>,
    pub mutation_dataset_name: String,
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
    pub val_dataset_name: String,
    pub per_variant_concurrency: usize,
}

/// Result of the mutate sub-step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GepaMutateResult {
    Continue(Box<GepaMutateContinue>),
    SkipIteration(Box<GepaSkipIteration>),
}

/// Sub-step 5: Evaluate child, compare, conditionally eval on val set, update frontier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaEvalAndUpdateParams {
    pub child: SelectedVariant,
    pub parent_evaluation_stats: HashMap<EvaluatorName, EvaluatorStats>,
    pub mutation_dataset_name: String,
    pub gepa_config: UninitializedGEPAConfig,
    pub iteration: usize,
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
    pub val_dataset_name: String,
    pub per_variant_concurrency: usize,
}

/// Result of the eval-and-update sub-step (always returned, no enum).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaIterUpdateResult {
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
}
