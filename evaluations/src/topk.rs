//! Durable Top-K Variant Evaluation Task
//!
//! This module implements an adaptive evaluation algorithm that evaluates multiple variants
//! against a dataset using durable execution for fault tolerance. The algorithm supports:
//!
//! - Multi-variant evaluation with per-variant stopping conditions
//! - Batch processing for efficiency
//! - Checkpointed execution for crash recovery
//! - Configurable minimum/maximum datapoints and precision targets
//!
//! See `topk_explanation.md` for detailed documentation.

use tensorzero_core::db::clickhouse::clickhouse_client::ProductionClickHouseClient;
use uuid::Uuid;

use crate::EvaluationVariant;
use crate::betting_confidence_sequences::MeanBettingConfidenceSequence;

const EVALUATOR_FAILURE_THRESHOLD: f32 = 0.05;
const VARIANT_FAILURE_THRESHOLD: f32 = 0.05;

enum VariantStatus {
    // Still running evals on this variant
    Active,
    // Not running evals; variant is confidently within top k_min
    Include,
    // Not running evals; variant is confidently outside the top k_max
    Exclude,
    // Not running evals; variant failure rate is confidently >= VARIANT_FAILURE_THRESHOLD
    Failed,
}

// Enum for global stopping condition.
// In case multiple stopping conditions are satisfied simultaneously,
// the highest ranked condition takes precedence. The order of the
// last three is fairly arbitrary.
pub enum GlobalStoppingReason {
    // If top-k found, return the k that caused stopping (largest k found in k_max..k_min)
    TopKFound(u32),
    // Hit datapoint limit
    MaxDatapointsReached,
    // If evaluator(s) failed, return name(s) of failed evaluator(s).
    // An evaluator fails if the lower bound of the confidence sequence for its
    // failure rate exceeds EVALUATOR_FAILURE_THRESHOLD.
    EvaluatorsFailed(Vec<String>),
    // If too many variants failes, return name(s) of failed variant(s).
    // A variant fails if the lower bound of the confidence sequences for its
    // failure rate exceed VARIANT_FAILURE_THRESHOLD.
    // If more than num_variants - k_min variants have failed,
    // we can no longer identify the top-k variants for any k
    // in k_min..k_max.
    TooManyVariantsFailed(Vec<String>),
}

// Arguments to run() function below
pub struct TopKVariantArgs {
    evaluation_name: String,
    variant_list: Vec<EvaluationVariant>,
    dataset_name: String,
    k_min: u32,
    k_max: u32,
    max_datapoints: u64,
    epsilon: f32,
    alpha_performance: f32,
    alpha_failure: f32,
    batch_size: u32,
}

// Struct for the output of the run() function below
pub struct AdaptiveEvalStoppingResults {
    pub variant_performance: Vec<MeanBettingConfidenceSequence>,
    pub variant_failure_rates: Vec<MeanBettingConfidenceSequence>,
    pub evaluator_failure_rates: Vec<MeanBettingConfidenceSequence>,
    pub stopping_reason: GlobalStoppingReason,
}
