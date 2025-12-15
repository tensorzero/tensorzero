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
//! NOTE: This module is work in progress.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use durable::{Task, TaskContext, TaskResult, async_trait};
use serde::{Deserialize, Serialize};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::endpoints::datasets::v1::get_datapoints;
use tensorzero_core::endpoints::datasets::v1::list_datapoints;
use tensorzero_core::endpoints::datasets::v1::types::{
    GetDatapointsRequest, ListDatapointsRequest,
};
use tensorzero_core::evaluations::EvaluationConfig;
use tracing::{debug, info};
use uuid::Uuid;

use crate::betting_confidence_sequences::{
    MeanBettingConfidenceSequence, WealthProcessGridPoints, WealthProcesses, update_betting_cs,
};
use crate::evaluators::EvaluationResult;
use crate::stopping::CancellationTokens;
use crate::types::EvaluationVariant;
use crate::{
    BatchItemResult, Clients, EvaluationFunctionConfigTable, ProcessBatchParams,
    collect_batch_result, process_batch,
};

// ============================================================================
// Constants
// ============================================================================

/// Default batch size for top-k evaluation
const DEFAULT_BATCH_SIZE: usize = 100;

/// Default confidence sequence resolution (grid points for mean estimation)
const DEFAULT_CS_RESOLUTION: usize = 101;

/// Default alpha (significance level) for confidence sequences
const DEFAULT_ALPHA: f32 = 0.05;

// ============================================================================
// Core Types
// ============================================================================

/// Status of a variant in the top-k evaluation process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariantStatus {
    /// Still running evals on this variant
    Active,
    /// Not running evals; variant is confidently within top k_min
    Include,
    /// Not running evals; variant is confidently outside the top k_max
    Exclude,
    /// Not running evals; variant failure rate is confidently >= failure threshold
    Failed,
}

/// Reason why the top-k evaluation stopped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GlobalStoppingReason {
    /// Successfully identified a top-k set of variants
    TopKFound { k: u32, top_variants: Vec<String> },
    /// Reached the maximum number of datapoints without identifying top-k
    MaxDatapointsReached,
    /// At least one evaluator has a failure rate above the threshold
    EvaluatorFailed { evaluator_name: String },
    /// Too many variants failed (>= num_variants - k_min)
    TooManyVariantsFailed { num_failed: usize },
}

/// Result of checking the top-k stopping condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopKStoppingResult {
    /// Whether a top-k set was identified
    pub stopped: bool,
    /// The k value for which stopping occurred (if stopped)
    pub k: Option<u32>,
    /// Names of variants confidently in the top-k (if stopped)
    pub top_variants: Vec<String>,
}

/// Results from running the adaptive top-k evaluation.
#[derive(Debug)]
pub struct AdaptiveEvalStoppingResults {
    /// Unique ID for this evaluation run
    pub evaluation_run_id: Uuid,
    /// Performance confidence sequences for each variant
    pub variant_performance: HashMap<String, MeanBettingConfidenceSequence>,
    /// Failure rate confidence sequences for each variant
    pub variant_failures: HashMap<String, MeanBettingConfidenceSequence>,
    /// Failure rate confidence sequences for each evaluator
    pub evaluator_failures: HashMap<String, MeanBettingConfidenceSequence>,
    /// Final status of each variant
    pub variant_status: HashMap<String, VariantStatus>,
    /// Why the evaluation stopped
    pub stopping_reason: GlobalStoppingReason,
    /// Number of datapoints processed
    pub num_datapoints_processed: usize,
}

// ============================================================================
// Scoring
// ============================================================================

/// Type alias for batch results grouped by datapoint, then by variant.
///
/// This structure is optimized for the scoring function, which needs to compare
/// variants on the same datapoint. The outer map is keyed by datapoint ID,
/// and the inner map is keyed by variant name.
pub type BatchResultsByDatapoint = HashMap<Uuid, HashMap<String, BatchItemResult>>;

/// Trait for scoring functions that compare variants on a single datapoint.
///
/// A scoring function takes an (implicit) (m × r) matrix of evaluator results for a single datapoint
/// (where m is the number of variants and r is the number of evaluators) and produces an
/// m-vector of scores, one per variant. This enables cross-variant comparisons such as:
/// - Ranking variants by how many evaluators they perform best on
/// - Computing relative performance vs. the best variant on each evaluator
/// - Applying tournament-style scoring across evaluators
///
/// The matrix is represented as references to avoid copying:
/// - Rows (variants) are keyed by variant name in the outer `HashMap`
/// - Columns (evaluators) are keyed by evaluator name in each `EvaluationResult`
///
/// Implementations must gracefully handle:
/// - **Missing variants**: Some variants may not have results for this datapoint
/// - **Evaluator failures**: Individual evaluators may fail (`Err(...)` in `EvaluationResult`)
/// - **Missing evaluator results**: Some (variant, evaluator) cells may be `Ok(None)`
///
/// Common implementations might:
/// - Extract a single evaluator's score directly (no cross-variant comparison)
/// - Rank variants by performance on a single evaluator, then normalize ranks
/// - Count how many evaluators each variant "wins" on, normalized to [0, 1]
pub trait ScoringFunction: Send + Sync {
    /// Compute scores for each variant from a single datapoint's evaluation results.
    ///
    /// # Arguments
    /// * `evaluations` - An (m × r) matrix for one datapoint: variant name → &EvaluationResult.
    ///   Each `EvaluationResult` is a `HashMap<String, Result<Option<Value>>>` mapping
    ///   evaluator name to result, providing the column alignment.
    ///
    /// # Returns
    /// A map from variant name to score in [0, 1]. Variants for which a score cannot be
    /// computed (e.g., all evaluators failed) should be omitted from the result.
    fn score(&self, evaluations: &HashMap<String, &EvaluationResult>) -> HashMap<String, f64>;
}

/// Update variant statistics and confidence sequences based on evaluation results.
///
/// This function updates:
/// - `variant_performance`: Mean performance confidence sequences for each variant
/// - `variant_failures`: Failure rate confidence sequences for each variant
/// - `evaluator_failures`: Per-evaluator failure rate confidence sequences
///
/// For variant performance, scores are computed using the provided scoring function,
/// which compares variants across evaluators for each datapoint.
/// For variant failures, each `Error` result is counted as a failure (1.0) and each
/// `Success` is counted as a non-failure (0.0).
/// For evaluator failures, each evaluator error within a `Success` result is counted
/// as a failure (1.0) for that specific evaluator.
///
/// # Arguments
/// * `results_by_datapoint` - Batch results grouped by datapoint ID, then by variant name
/// * `scoring_fn` - Function to convert per-datapoint evaluation matrices to scores
/// * `variant_performance` - Map to update with performance statistics
/// * `variant_failures` - Map to update with variant failure rates
/// * `evaluator_failures` - Map to update with evaluator failure rates
///
/// # Returns
/// `Ok(())` on success, or an error if confidence sequence updates fail.
pub fn compute_updates(
    results_by_datapoint: &BatchResultsByDatapoint,
    scoring_fn: &dyn ScoringFunction,
    variant_performance: &mut HashMap<String, MeanBettingConfidenceSequence>,
    variant_failures: &mut HashMap<String, MeanBettingConfidenceSequence>,
    evaluator_failures: &mut HashMap<String, MeanBettingConfidenceSequence>,
) -> Result<()> {
    // Collect observations per variant for performance scores
    let mut performance_observations: HashMap<String, Vec<f64>> = HashMap::new();

    // Collect observations per variant for failure rates (1.0 = failure, 0.0 = success)
    let mut failure_observations: HashMap<String, Vec<f64>> = HashMap::new();

    // Collect observations for each evaluator's failure rate
    let mut evaluator_failure_observations: HashMap<String, Vec<f64>> = HashMap::new();

    // Process each datapoint's results
    for results_by_variant in results_by_datapoint.values() {
        // Build the scoring function input (variant_name -> &EvaluationResult) for this datapoint
        let mut evaluations_for_scoring: HashMap<String, &EvaluationResult> = HashMap::new();

        for (variant_name, result) in results_by_variant {
            match result {
                BatchItemResult::Success(success) => {
                    // Variant didn't fail for this datapoint
                    failure_observations
                        .entry(variant_name.clone())
                        .or_default()
                        .push(0.0);

                    // Check each evaluator for errors
                    for (evaluator_name, eval_result) in &success.evaluation_result {
                        let failure_value = match eval_result {
                            Ok(_) => 0.0,  // Evaluator succeeded
                            Err(_) => 1.0, // Evaluator failed
                        };
                        evaluator_failure_observations
                            .entry(evaluator_name.clone())
                            .or_default()
                            .push(failure_value);
                    }

                    // Add to scoring input
                    evaluations_for_scoring
                        .insert(variant_name.clone(), &success.evaluation_result);
                }
                BatchItemResult::Error(_) => {
                    // Variant failed for this datapoint
                    failure_observations
                        .entry(variant_name.clone())
                        .or_default()
                        .push(1.0);
                    // Note: We don't update evaluator failures here since we don't have
                    // evaluation results - the failure occurred before evaluation
                }
                BatchItemResult::Cancelled => {
                    // Skip cancelled tasks - they don't count toward any statistics
                    continue;
                }
            }
        }

        // Compute performance scores for this datapoint if we have any successful evaluations
        if !evaluations_for_scoring.is_empty() {
            let scores = scoring_fn.score(&evaluations_for_scoring);

            for (variant_name, score) in scores {
                debug_assert!(
                    (0.0..=1.0).contains(&score),
                    "ScoringFunction returned score {score} for variant {variant_name}, but scores must be in [0, 1]"
                );
                performance_observations
                    .entry(variant_name)
                    .or_default()
                    .push(score);
            }
        }
    }

    // Update variant performance confidence sequences
    for (variant_name, observations) in performance_observations {
        if observations.is_empty() {
            continue;
        }
        if let Some(cs) = variant_performance.remove(&variant_name) {
            let updated = update_betting_cs(cs, observations, None)?;
            variant_performance.insert(variant_name, updated);
        }
    }

    // Update variant failures confidence sequences
    for (variant_name, observations) in failure_observations {
        if observations.is_empty() {
            continue;
        }
        if let Some(cs) = variant_failures.remove(&variant_name) {
            let updated = update_betting_cs(cs, observations, None)?;
            variant_failures.insert(variant_name, updated);
        }
    }

    // Update evaluator failures confidence sequences
    for (evaluator_name, observations) in evaluator_failure_observations {
        if observations.is_empty() {
            continue;
        }
        if let Some(cs) = evaluator_failures.remove(&evaluator_name) {
            let updated = update_betting_cs(cs, observations, None)?;
            evaluator_failures.insert(evaluator_name, updated);
        }
    }

    Ok(())
}

// ============================================================================
// Stopping Conditions
// ============================================================================

/// Check if we can confidently identify a top-k set of variants.
///
/// A variant is "confidently in the top k" if its confidence sequence lower bound
/// exceeds the upper bounds of at least (num_variants - k) other variants, with a
/// tolerance `epsilon`. This means we can be confident it's better than at least
/// (num_variants - k) variants, or at least not more than epsilon worse than those
/// variants.
///
/// We check for each k in the range [k_min, k_max] (inclusive), starting from k_max
/// and working down. We return the largest k for which we can identify a top-k set.
///
/// # Arguments
/// * `variant_performance` - Map from variant name to its performance confidence sequence
/// * `k_min` - Minimum acceptable k value
/// * `k_max` - Maximum k value to check
/// * `epsilon` (optional) - A tolerance for performance equivalence. If None, set to 0.0.
///
/// # Returns
/// A `TopKStoppingResult` indicating whether stopping occurred and which variants are in the top-k
/// if stopping occurred.
pub fn check_topk_stopping(
    variant_performance: &HashMap<String, MeanBettingConfidenceSequence>,
    k_min: u32,
    k_max: u32,
    epsilon: Option<f64>,
) -> anyhow::Result<TopKStoppingResult> {
    let epsilon = epsilon.unwrap_or(0.0);
    let num_variants = variant_performance.len();

    // Invalid parameters: return error
    if k_min == 0 {
        anyhow::bail!("k_min must be > 0");
    }
    if k_max < k_min {
        anyhow::bail!("k_max ({k_max}) < k_min ({k_min})");
    }
    if epsilon < 0.0 {
        anyhow::bail!("epsilon ({epsilon}) must be >= 0");
    }

    // Runtime edge cases: log a warning, return empty result
    if num_variants == 0 {
        tracing::warn!("check_topk_stopping: no variants provided");
        return Ok(TopKStoppingResult {
            stopped: false,
            k: None,
            top_variants: vec![],
        });
    }

    if k_min as usize > num_variants {
        tracing::warn!(
            k_min,
            num_variants,
            "check_topk_stopping: k_min > num_variants, can't identify top-k variants"
        );
        return Ok(TopKStoppingResult {
            stopped: false,
            k: None,
            top_variants: vec![],
        });
    }

    // Collect upper bounds into a sorted vec for binary search
    let mut upper_bounds: Vec<f64> = variant_performance.values().map(|cs| cs.cs_upper).collect();
    upper_bounds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // For each variant, count how many other variants' upper bounds its lower bound exceedsm
    // with tolerance epsilon. This tells us how many variants it "confidently beats".
    // We subtract 1 if the variant would count itself as beaten (when cs_upper - epsilon < cs_lower).
    let mut variants_with_n_beaten: Vec<(&String, usize)> = variant_performance
        .iter()
        .map(|(name, cs)| {
            // Binary search to find how many upper bounds satisfy: ub - epsilon < cs_lower
            let num_beaten_including_self =
                upper_bounds.partition_point(|&ub| ub - epsilon < cs.cs_lower);
            // Subtract 1 if this variant's own upper bound was counted
            let beats_self = cs.cs_upper - epsilon < cs.cs_lower;
            let num_beaten = if beats_self {
                num_beaten_including_self.saturating_sub(1)
            } else {
                num_beaten_including_self
            };
            (name, num_beaten)
        })
        .collect();

    // Sort by num_beaten descending (best variants first)
    variants_with_n_beaten.sort_by(|a, b| b.1.cmp(&a.1));

    // Check each k from k_max down to k_min
    // We want the largest k for which we can identify a top-k set
    for k in (k_min..=k_max).rev() {
        let k_usize = k as usize;
        let threshold = num_variants.saturating_sub(k_usize);

        // Check if the top k variants each beat at least (num_variants - k) others
        if k_usize <= variants_with_n_beaten.len()
            && variants_with_n_beaten[..k_usize]
                .iter()
                .all(|(_, num_beaten)| *num_beaten >= threshold)
        {
            let top_variants: Vec<String> = variants_with_n_beaten[..k_usize]
                .iter()
                .map(|(name, _)| (*name).clone())
                .collect();

            return Ok(TopKStoppingResult {
                stopped: true,
                k: Some(k),
                top_variants,
            });
        }
    }

    Ok(TopKStoppingResult {
        stopped: false,
        k: None,
        top_variants: vec![],
    })
}

/// Convenience wrapper to check if a specific k can be identified.
///
/// This is equivalent to calling `check_topk_stopping` with k_min = k_max = k.
pub fn check_topk(
    variant_performance: &HashMap<String, MeanBettingConfidenceSequence>,
    k: u32,
    epsilon: Option<f64>,
) -> anyhow::Result<TopKStoppingResult> {
    check_topk_stopping(variant_performance, k, k, epsilon)
}

/// Check if a variant should be marked as failed based on its failure rate confidence sequence.
fn check_variant_failed(cs: &MeanBettingConfidenceSequence, threshold: f64) -> bool {
    // Variant is failed if the lower bound of its failure rate CI exceeds the threshold
    cs.cs_lower > threshold
}

/// Check if an evaluator has failed based on its failure rate confidence sequence.
fn check_evaluator_failed(cs: &MeanBettingConfidenceSequence, threshold: f64) -> bool {
    // Evaluator is failed if the lower bound of its failure rate CI exceeds the threshold
    cs.cs_lower > threshold
}

// ============================================================================
// Durable Task
// ============================================================================

/// Serializable parameters for the top-k durable task.
/// Non-serializable resources (clients, scoring_fn) are passed via State.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopKTaskParams {
    /// Name of the evaluation
    pub evaluation_name: String,
    /// Dataset name to evaluate on
    pub dataset_name: String,
    /// List of variant names to evaluate
    pub variant_names: Vec<String>,
    /// Minimum k for top-k identification
    pub k_min: u32,
    /// Maximum k for top-k identification
    pub k_max: u32,
    /// Tolerance for performance equivalence (epsilon)
    pub epsilon: Option<f64>,
    /// Maximum number of datapoints to process (None = unlimited)
    pub max_datapoints: Option<usize>,
    /// Batch size for processing
    pub batch_size: Option<usize>,
    /// Failure rate threshold for variants
    pub variant_failure_threshold: Option<f64>,
    /// Failure rate threshold for evaluators
    pub evaluator_failure_threshold: Option<f64>,
    /// Number of concurrent requests
    pub concurrency: usize,
    /// Cache mode for inference (serialized)
    pub inference_cache: CacheEnabledMode,
}

/// Application state for the top-k task (non-serializable resources).
#[derive(Clone)]
pub struct TopKTaskState {
    /// Clients for inference and database access
    pub clients: Arc<Clients>,
    /// Evaluation configuration
    pub evaluation_config: Arc<EvaluationConfig>,
    /// Function configs table
    pub function_configs: Arc<EvaluationFunctionConfigTable>,
    /// Scoring function for converting evaluation results to performance scores
    pub scoring_fn: Arc<dyn ScoringFunction>,
}

/// Serializable loop state for checkpointing between batches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopKLoopState {
    /// Variant statuses
    pub variant_status: HashMap<String, VariantStatus>,
    /// Performance confidence sequences
    pub variant_performance: HashMap<String, MeanBettingConfidenceSequence>,
    /// Variant failure rate confidence sequences
    pub variant_failures: HashMap<String, MeanBettingConfidenceSequence>,
    /// Evaluator failure rate confidence sequences
    pub evaluator_failures: HashMap<String, MeanBettingConfidenceSequence>,
    /// Number of datapoints processed
    pub num_datapoints_processed: usize,
    /// Current batch index
    pub batch_index: usize,
}

/// Parameters for the fetch_datapoint_ids step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FetchDatapointIdsParams {
    function_name: String,
    dataset_name: String,
    max_datapoints: Option<usize>,
}

/// Parameters for the process_batch step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessBatchStepParams {
    batch_ids: Vec<Uuid>,
    loop_state: TopKLoopState,
    k_min: u32,
    k_max: u32,
    epsilon: Option<f64>,
    variant_failure_threshold: Option<f64>,
    evaluator_failure_threshold: Option<f64>,
    batch_idx: usize,
    // TopKContext fields that need to be passed through
    evaluation_name: String,
    evaluation_run_id: Uuid,
    dataset_name: String,
    inference_cache: CacheEnabledMode,
    concurrency: usize,
}

/// Serializable output for the top-k task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopKTaskOutput {
    /// Unique ID for this evaluation run
    pub evaluation_run_id: Uuid,
    /// Final variant statuses
    pub variant_status: HashMap<String, VariantStatus>,
    /// Final performance confidence sequences
    pub variant_performance: HashMap<String, MeanBettingConfidenceSequence>,
    /// Final variant failure rate confidence sequences
    pub variant_failures: HashMap<String, MeanBettingConfidenceSequence>,
    /// Final evaluator failure rate confidence sequences
    pub evaluator_failures: HashMap<String, MeanBettingConfidenceSequence>,
    /// Why the evaluation stopped
    pub stopping_reason: GlobalStoppingReason,
    /// Number of datapoints processed
    pub num_datapoints_processed: usize,
}

impl From<AdaptiveEvalStoppingResults> for TopKTaskOutput {
    fn from(results: AdaptiveEvalStoppingResults) -> Self {
        Self {
            evaluation_run_id: results.evaluation_run_id,
            variant_status: results.variant_status,
            variant_performance: results.variant_performance,
            variant_failures: results.variant_failures,
            evaluator_failures: results.evaluator_failures,
            stopping_reason: results.stopping_reason,
            num_datapoints_processed: results.num_datapoints_processed,
        }
    }
}

/// Step function to fetch and shuffle datapoint IDs.
async fn fetch_datapoint_ids_step(
    params: FetchDatapointIdsParams,
    state: TopKTaskState,
) -> anyhow::Result<Vec<Uuid>> {
    let request = ListDatapointsRequest {
        function_name: Some(params.function_name),
        limit: params.max_datapoints.map(|n| n as u32),
        offset: Some(0),
        ..Default::default()
    };
    let datapoints = list_datapoints(
        &state.clients.clickhouse_client,
        params.dataset_name,
        request,
    )
    .await?
    .datapoints;

    let mut ids: Vec<Uuid> = datapoints.iter().map(|d| d.id()).collect();

    // Shuffle for randomization
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    ids.shuffle(&mut rng);

    Ok(ids)
}

/// Extract variant name from EvaluationVariant.
fn get_variant_name(variant: &EvaluationVariant) -> String {
    match variant {
        EvaluationVariant::Name(name) => name.clone(),
        // Dynamic variants don't have names, use a placeholder
        EvaluationVariant::Info(_) => "dynamic_variant".to_string(),
    }
}

/// Step function to process a batch and update loop state.
async fn process_batch_step(
    params: ProcessBatchStepParams,
    state: TopKTaskState,
) -> anyhow::Result<TopKLoopState> {
    let mut current_state = params.loop_state;

    // Retrieve datapoints from ClickHouse
    let request = GetDatapointsRequest {
        ids: params.batch_ids.clone(),
    };
    let datapoints = get_datapoints(&state.clients.clickhouse_client, None, request)
        .await?
        .datapoints;

    // Process the batch if we have datapoints and active variants
    let results: BatchResultsByDatapoint = if datapoints.is_empty() {
        tracing::warn!(
            batch_size = params.batch_ids.len(),
            "No datapoints found for batch"
        );
        HashMap::new()
    } else {
        // Get active variants
        let active_variants: Vec<Arc<EvaluationVariant>> = current_state
            .variant_status
            .iter()
            .filter(|(_, status)| **status == VariantStatus::Active)
            .map(|(name, _)| Arc::new(EvaluationVariant::Name(name.clone())))
            .collect();

        if active_variants.is_empty() {
            tracing::debug!("No active variants to process");
            HashMap::new()
        } else {
            // Use empty cancellation tokens - top-k has its own stopping logic
            let cancellation_tokens = Arc::new(CancellationTokens::default());

            // Build params for shared process_batch
            let semaphore = Arc::new(tokio::sync::Semaphore::new(params.concurrency));
            let batch_params = ProcessBatchParams {
                clients: state.clients.clone(),
                function_configs: state.function_configs.clone(),
                evaluation_config: state.evaluation_config.clone(),
                evaluation_name: Arc::new(params.evaluation_name.clone()),
                evaluation_run_id: params.evaluation_run_id,
                dataset_name: Arc::new(params.dataset_name.clone()),
                inference_cache: params.inference_cache,
                semaphore,
                cancellation_tokens,
            };

            // Call the shared process_batch from lib.rs
            let (mut join_set, task_id_map) =
                process_batch(&batch_params, datapoints, &active_variants);

            // Collect results and group by datapoint ID, then by variant
            let mut results_by_datapoint: BatchResultsByDatapoint = HashMap::new();

            while let Some(result) = join_set.join_next_with_id().await {
                let batch_result = collect_batch_result(result, &task_id_map);

                // Get datapoint ID and variant name for grouping
                let (datapoint_id, variant_name) = match &batch_result {
                    BatchItemResult::Success(success) => {
                        (success.datapoint.id(), get_variant_name(&success.variant))
                    }
                    BatchItemResult::Error(error) => {
                        let variant_name = error
                            .variant
                            .as_ref()
                            .map(|v| get_variant_name(v))
                            .unwrap_or_else(|| "unknown".to_string());
                        (error.datapoint_id, variant_name)
                    }
                    BatchItemResult::Cancelled => continue, // Skip cancelled tasks
                };

                results_by_datapoint
                    .entry(datapoint_id)
                    .or_default()
                    .insert(variant_name, batch_result);
            }

            results_by_datapoint
        }
    };

    // Update confidence sequences
    compute_updates(
        &results,
        state.scoring_fn.as_ref(),
        &mut current_state.variant_performance,
        &mut current_state.variant_failures,
        &mut current_state.evaluator_failures,
    )?;

    current_state.num_datapoints_processed += params.batch_ids.len();
    current_state.batch_index = params.batch_idx + 1;

    // Check top-k stopping condition
    let stopping_result = check_topk_stopping(
        &current_state.variant_performance,
        params.k_min,
        params.k_max,
        params.epsilon,
    )?;

    // Update variant statuses
    update_variant_statuses(
        &mut current_state.variant_status,
        &current_state.variant_performance,
        &current_state.variant_failures,
        params.variant_failure_threshold,
        &stopping_result,
        params.k_max,
    );

    // Check for evaluator failure
    if let Some(threshold) = params.evaluator_failure_threshold {
        for (evaluator_name, cs) in &current_state.evaluator_failures {
            if check_evaluator_failed(cs, threshold) {
                info!(
                    evaluator_name = %evaluator_name,
                    "Evaluator failure threshold exceeded"
                );
                // Mark all active variants as failed due to evaluator
                // (The stopping reason will be set in the output)
            }
        }
    }

    Ok(current_state)
}

/// Create an initial confidence sequence for a named entity.
fn create_initial_cs(name: &str, resolution: usize, alpha: f32) -> MeanBettingConfidenceSequence {
    MeanBettingConfidenceSequence {
        name: name.to_string(),
        mean_regularized: 0.5,
        variance_regularized: 0.25,
        count: 0,
        mean_est: 0.5,
        cs_lower: 0.0,
        cs_upper: 1.0,
        alpha,
        wealth: WealthProcesses {
            grid: WealthProcessGridPoints::Resolution(resolution),
            wealth_upper: vec![1.0; resolution],
            wealth_lower: vec![1.0; resolution],
        },
    }
}

/// Update variant statuses based on current confidence sequences and top-k stopping result.
fn update_variant_statuses(
    variant_status: &mut HashMap<String, VariantStatus>,
    variant_performance: &HashMap<String, MeanBettingConfidenceSequence>,
    variant_failures: &HashMap<String, MeanBettingConfidenceSequence>,
    variant_failure_threshold: Option<f64>,
    stopping_result: &TopKStoppingResult,
    k_max: u32,
) {
    let num_variants = variant_status.len();

    for (name, status) in variant_status.iter_mut() {
        // Skip already-stopped variants
        if *status != VariantStatus::Active {
            continue;
        }

        // Check for failure based on failure rate
        if let Some(threshold) = variant_failure_threshold
            && let Some(failure_cs) = variant_failures.get(name)
            && check_variant_failed(failure_cs, threshold)
        {
            *status = VariantStatus::Failed;
            continue;
        }

        // If top-k stopping occurred, update based on inclusion in top set
        if stopping_result.stopped {
            if stopping_result.top_variants.contains(name) {
                *status = VariantStatus::Include;
            } else {
                *status = VariantStatus::Exclude;
            }
            continue;
        }

        // Check if variant can be excluded early (its upper bound is below k_max others' lower bounds)
        if let Some(perf_cs) = variant_performance.get(name) {
            let num_definitely_better = variant_performance
                .iter()
                .filter(|(other_name, other_cs)| {
                    *other_name != name && other_cs.cs_lower > perf_cs.cs_upper
                })
                .count();

            // If at least (num_variants - k_max) variants are definitely better,
            // this variant cannot be in the top k_max
            if num_definitely_better >= num_variants.saturating_sub(k_max as usize) {
                *status = VariantStatus::Exclude;
            }
        }
    }
}

/// Determine the stopping reason based on final loop state.
fn determine_stopping_reason(
    loop_state: &TopKLoopState,
    params: &TopKTaskParams,
    _total_datapoints: usize,
) -> GlobalStoppingReason {
    // Check if we identified a top-k set
    let stopping_result = check_topk_stopping(
        &loop_state.variant_performance,
        params.k_min,
        params.k_max,
        params.epsilon,
    );

    if let Ok(result) = stopping_result
        && result.stopped
    {
        return GlobalStoppingReason::TopKFound {
            k: result.k.unwrap_or(0),
            top_variants: result.top_variants,
        };
    }

    // Check for evaluator failure
    if let Some(threshold) = params.evaluator_failure_threshold {
        for (evaluator_name, cs) in &loop_state.evaluator_failures {
            if check_evaluator_failed(cs, threshold) {
                return GlobalStoppingReason::EvaluatorFailed {
                    evaluator_name: evaluator_name.clone(),
                };
            }
        }
    }

    // Check for too many variant failures
    let num_failed = loop_state
        .variant_status
        .values()
        .filter(|s| **s == VariantStatus::Failed)
        .count();
    let num_variants = loop_state.variant_status.len();
    if num_failed >= num_variants.saturating_sub(params.k_min as usize) {
        return GlobalStoppingReason::TooManyVariantsFailed { num_failed };
    }

    // Default: max datapoints reached
    GlobalStoppingReason::MaxDatapointsReached
}

/// The durable top-k evaluation task.
pub struct TopKTask;

#[async_trait]
impl Task<TopKTaskState> for TopKTask {
    const NAME: &'static str = "topk-evaluation";
    type Params = TopKTaskParams;
    type Output = TopKTaskOutput;

    async fn run(
        params: Self::Params,
        mut ctx: TaskContext<TopKTaskState>,
        state: TopKTaskState,
    ) -> TaskResult<Self::Output> {
        // Validate arguments
        if params.variant_names.is_empty() {
            return Err(durable::TaskError::Failed(anyhow::anyhow!(
                "At least one variant must be provided"
            )));
        }
        if params.k_min == 0 {
            return Err(durable::TaskError::Failed(anyhow::anyhow!(
                "k_min must be > 0"
            )));
        }
        if params.k_max < params.k_min {
            return Err(durable::TaskError::Failed(anyhow::anyhow!(
                "k_max ({}) must be >= k_min ({})",
                params.k_max,
                params.k_min
            )));
        }
        if params.k_max as usize > params.variant_names.len() {
            return Err(durable::TaskError::Failed(anyhow::anyhow!(
                "k_max ({}) must be <= number of variants ({})",
                params.k_max,
                params.variant_names.len()
            )));
        }

        let batch_size = params.batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

        // Get evaluator names from the evaluation config
        let EvaluationConfig::Inference(inference_config) = &*state.evaluation_config;
        let evaluator_names: Vec<String> = inference_config.evaluators.keys().cloned().collect();

        // Generate evaluation run ID durably
        let evaluation_run_id = ctx.uuid7().await?;

        info!(
            evaluation_run_id = %evaluation_run_id,
            num_variants = params.variant_names.len(),
            k_min = params.k_min,
            k_max = params.k_max,
            batch_size = batch_size,
            "Starting top-k evaluation"
        );

        // CHECKPOINT 1: Fetch and shuffle datapoint IDs
        let fetch_params = FetchDatapointIdsParams {
            function_name: inference_config.function_name.clone(),
            dataset_name: params.dataset_name.clone(),
            max_datapoints: params.max_datapoints,
        };
        let datapoint_ids: Vec<Uuid> = ctx
            .step(
                "fetch_datapoint_ids",
                fetch_params,
                fetch_datapoint_ids_step,
            )
            .await?;

        let total_datapoints = datapoint_ids.len();
        info!(
            total_datapoints = total_datapoints,
            "Loaded and shuffled datapoint IDs"
        );

        if total_datapoints == 0 {
            return Err(durable::TaskError::Failed(anyhow::anyhow!(
                "Dataset is empty"
            )));
        }

        // Initialize loop state
        let mut loop_state = TopKLoopState {
            variant_status: params
                .variant_names
                .iter()
                .map(|name| (name.clone(), VariantStatus::Active))
                .collect(),
            variant_performance: params
                .variant_names
                .iter()
                .map(|name| {
                    (
                        name.clone(),
                        create_initial_cs(name, DEFAULT_CS_RESOLUTION, DEFAULT_ALPHA),
                    )
                })
                .collect(),
            variant_failures: params
                .variant_names
                .iter()
                .map(|name| {
                    (
                        name.clone(),
                        create_initial_cs(name, DEFAULT_CS_RESOLUTION, DEFAULT_ALPHA),
                    )
                })
                .collect(),
            evaluator_failures: evaluator_names
                .iter()
                .map(|name| {
                    (
                        name.clone(),
                        create_initial_cs(name, DEFAULT_CS_RESOLUTION, DEFAULT_ALPHA),
                    )
                })
                .collect(),
            num_datapoints_processed: 0,
            batch_index: 0,
        };

        // Process batches
        let batches: Vec<Vec<Uuid>> = datapoint_ids
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        for (batch_idx, batch_ids) in batches.into_iter().enumerate() {
            // Check if all active variants are gone
            let has_active = loop_state
                .variant_status
                .values()
                .any(|s| *s == VariantStatus::Active);
            if !has_active {
                break;
            }

            debug!(
                batch_number = batch_idx + 1,
                batch_size = batch_ids.len(),
                "Processing batch"
            );

            // CHECKPOINT 2: Process batch and update state
            let batch_step_params = ProcessBatchStepParams {
                batch_ids,
                loop_state: loop_state.clone(),
                k_min: params.k_min,
                k_max: params.k_max,
                epsilon: params.epsilon,
                variant_failure_threshold: params.variant_failure_threshold,
                evaluator_failure_threshold: params.evaluator_failure_threshold,
                batch_idx,
                evaluation_name: params.evaluation_name.clone(),
                evaluation_run_id,
                dataset_name: params.dataset_name.clone(),
                inference_cache: params.inference_cache,
                concurrency: params.concurrency,
            };
            loop_state = ctx
                .step(
                    &format!("batch_{batch_idx}"),
                    batch_step_params,
                    process_batch_step,
                )
                .await?;
        }

        // Determine final stopping reason
        let stopping_reason = determine_stopping_reason(&loop_state, &params, total_datapoints);

        info!(
            stopping_reason = ?stopping_reason,
            num_datapoints_processed = loop_state.num_datapoints_processed,
            "Top-k evaluation complete"
        );

        Ok(TopKTaskOutput {
            evaluation_run_id,
            variant_status: loop_state.variant_status,
            variant_performance: loop_state.variant_performance,
            variant_failures: loop_state.variant_failures,
            evaluator_failures: loop_state.evaluator_failures,
            stopping_reason,
            num_datapoints_processed: loop_state.num_datapoints_processed,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DatapointVariantError;
    use crate::DatapointVariantResult;
    use crate::betting_confidence_sequences::{WealthProcessGridPoints, WealthProcesses};
    use crate::types::EvaluationVariant;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use tensorzero_core::client::Input;
    use tensorzero_core::endpoints::datasets::{ChatInferenceDatapoint, Datapoint};
    use tensorzero_core::endpoints::inference::{ChatInferenceResponse, InferenceResponse};
    use tensorzero_core::inference::types::{ContentBlockChatOutput, Text, Usage};
    use tensorzero_core::tool::DynamicToolParams;

    // ============================================================================
    // Test Helpers
    // ============================================================================

    /// Helper to create a confidence sequence with specified bounds (for testing stopping conditions)
    fn mock_cs_with_bounds(
        name: &str,
        cs_lower: f64,
        cs_upper: f64,
    ) -> (String, MeanBettingConfidenceSequence) {
        (
            name.to_string(),
            MeanBettingConfidenceSequence {
                name: name.to_string(),
                mean_regularized: (cs_lower + cs_upper) / 2.0,
                variance_regularized: 0.1,
                count: 100,
                mean_est: (cs_lower + cs_upper) / 2.0,
                cs_lower,
                cs_upper,
                alpha: 0.05,
                wealth: WealthProcesses {
                    grid: WealthProcessGridPoints::Resolution(101),
                    wealth_upper: vec![1.0; 101],
                    wealth_lower: vec![1.0; 101],
                },
            },
        )
    }

    /// Helper to create a fresh confidence sequence with no observations (for testing updates)
    fn mock_fresh_cs(name: &str) -> MeanBettingConfidenceSequence {
        MeanBettingConfidenceSequence {
            name: name.to_string(),
            mean_regularized: 0.5,
            variance_regularized: 0.25,
            count: 0,
            mean_est: 0.5,
            cs_lower: 0.0,
            cs_upper: 1.0,
            alpha: 0.05,
            wealth: WealthProcesses {
                grid: WealthProcessGridPoints::Resolution(101),
                wealth_upper: vec![1.0; 101],
                wealth_lower: vec![1.0; 101],
            },
        }
    }

    /// Helper to create a mock DatapointVariantResult for testing.
    /// Takes a variant name and evaluation results (evaluator_name -> value).
    fn mock_success(
        datapoint_id: Uuid,
        variant_name: &str,
        eval_results: HashMap<String, Result<Option<Value>>>,
    ) -> BatchItemResult {
        // Create a minimal ChatInferenceDatapoint
        let datapoint = Datapoint::Chat(ChatInferenceDatapoint {
            dataset_name: "test_dataset".to_string(),
            function_name: "test_function".to_string(),
            id: datapoint_id,
            episode_id: None,
            input: Input::default(),
            output: None,
            tool_params: DynamicToolParams::default(),
            tags: None,
            auxiliary: String::new(),
            source_inference_id: None,
            staled_at: None,
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            is_deleted: false,
            is_custom: false,
            name: None,
        });

        // Create a minimal InferenceResponse
        let inference_response = InferenceResponse::Chat(ChatInferenceResponse {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: variant_name.to_string(),
            content: vec![ContentBlockChatOutput::Text(Text {
                text: "test output".to_string(),
            })],
            usage: Usage {
                input_tokens: Some(0),
                output_tokens: Some(0),
            },
            original_response: None,
            finish_reason: None,
        });

        BatchItemResult::Success(Box::new(DatapointVariantResult {
            datapoint,
            variant: Arc::new(EvaluationVariant::Name(variant_name.to_string())),
            inference_response,
            evaluation_result: eval_results,
        }))
    }

    /// A simple scoring function that extracts each variant's first evaluator value as its score
    struct FirstEvaluatorScore;

    impl ScoringFunction for FirstEvaluatorScore {
        fn score(&self, evaluations: &HashMap<String, &EvaluationResult>) -> HashMap<String, f64> {
            let mut scores = HashMap::new();
            for (variant_name, eval_result) in evaluations {
                // Return the first Ok(Some(value)) we find, converted to f64
                for result in eval_result.values() {
                    if let Ok(Some(value)) = result
                        && let Some(num) = value.as_f64()
                    {
                        scores.insert(variant_name.clone(), num);
                        break;
                    }
                }
            }
            scores
        }
    }

    // ============================================================================
    // Tests for check_topk_stopping
    // ============================================================================

    /// Test that empty input returns no stopping (graceful handling).
    #[test]
    fn test_check_topk_stopping_empty() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();
        let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
        assert!(!result.stopped);
        assert!(result.k.is_none());
        assert!(result.top_variants.is_empty());
    }

    /// Test that k_min=0 returns an error.
    #[test]
    fn test_check_topk_stopping_k_min_zero_errors() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs_with_bounds("a", 0.5, 0.7)].into_iter().collect();

        let result = check_topk_stopping(&variant_performance, 0, 1, None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("k_min must be > 0")
        );
    }

    /// Test that k_max < k_min returns an error.
    #[test]
    fn test_check_topk_stopping_k_max_less_than_k_min_errors() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs_with_bounds("a", 0.5, 0.7)].into_iter().collect();

        let result = check_topk_stopping(&variant_performance, 2, 1, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("k_max"));
    }

    /// Test that k_min > num_variants returns no stopping (graceful handling).
    #[test]
    fn test_check_topk_stopping_k_min_exceeds_num_variants() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs_with_bounds("a", 0.5, 0.7)].into_iter().collect();

        let result = check_topk_stopping(&variant_performance, 5, 5, None).unwrap();
        assert!(!result.stopped);
        assert!(result.k.is_none());
        assert!(result.top_variants.is_empty());
    }

    /// Test that negative epsilon returns an error.
    #[test]
    fn test_check_topk_stopping_negative_epsilon_errors() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.7, 0.9),
            mock_cs_with_bounds("b", 0.3, 0.5),
        ]
        .into_iter()
        .collect();

        let result = check_topk_stopping(&variant_performance, 1, 1, Some(-0.1));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("epsilon"));
    }

    /// Test top-1 identification when one variant clearly dominates all others.
    #[test]
    fn test_check_topk_stopping_clear_winner() {
        // Variant A: [0.7, 0.9] - clearly better
        // Variant B: [0.3, 0.5] - clearly worse
        // Variant C: [0.2, 0.4] - clearly worse
        // A's lower bound (0.7) exceeds B's upper (0.5) and C's upper (0.4)
        // So A beats 2 variants, which is >= (3 - 1) = 2, so top-1 is identified
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.7, 0.9),
            mock_cs_with_bounds("b", 0.3, 0.5),
            mock_cs_with_bounds("c", 0.2, 0.4),
        ]
        .into_iter()
        .collect();

        let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert_eq!(result.top_variants.len(), 1);
        assert!(result.top_variants.contains(&"a".to_string()));
    }

    /// Test top-2 identification when two variants clearly beat a third.
    #[test]
    fn test_check_topk_stopping_top2() {
        // Variant A: [0.7, 0.9] - in top 2
        // Variant B: [0.6, 0.8] - in top 2
        // Variant C: [0.2, 0.4] - clearly worse
        // A's lower (0.7) > C's upper (0.4), so A beats 1
        // B's lower (0.6) > C's upper (0.4), so B beats 1
        // For top-2, each needs to beat >= (3 - 2) = 1 variant
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.7, 0.9),
            mock_cs_with_bounds("b", 0.6, 0.8),
            mock_cs_with_bounds("c", 0.2, 0.4),
        ]
        .into_iter()
        .collect();

        let result = check_topk_stopping(&variant_performance, 2, 2, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(2));
        assert_eq!(result.top_variants.len(), 2);
        assert!(result.top_variants.contains(&"a".to_string()));
        assert!(result.top_variants.contains(&"b".to_string()));
    }

    /// Test that overlapping confidence intervals prevent top-1 identification.
    #[test]
    fn test_check_topk_stopping_overlapping_intervals_no_stop() {
        // All intervals overlap significantly - can't distinguish
        // Variant A: [0.4, 0.7]
        // Variant B: [0.45, 0.65]
        // Variant C: [0.5, 0.6]
        // No lower bound exceeds any upper bound, so no one "beats" anyone
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.4, 0.7),
            mock_cs_with_bounds("b", 0.45, 0.65),
            mock_cs_with_bounds("c", 0.5, 0.6),
        ]
        .into_iter()
        .collect();

        let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
        assert!(!result.stopped);
        assert!(result.k.is_none());
    }

    /// Test that the largest viable k is returned when checking a range.
    #[test]
    fn test_check_topk_stopping_k_range() {
        // Variant A: [0.8, 0.9] - beats B and C
        // Variant B: [0.5, 0.7] - beats C only
        // Variant C: [0.2, 0.4] - beats no one
        // A beats 2 (both B and C's uppers are below 0.8)
        // B beats 1 (C's upper 0.4 < B's lower 0.5)
        // C beats 0
        //
        // For k=1: need to beat >= 2. Only A qualifies. Top-1 found.
        // For k=2: need to beat >= 1. A and B qualify. Top-2 found.
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.8, 0.9),
            mock_cs_with_bounds("b", 0.5, 0.7),
            mock_cs_with_bounds("c", 0.2, 0.4),
        ]
        .into_iter()
        .collect();

        // When k_min=1, k_max=2, should return k=2 (largest k that works)
        let result = check_topk_stopping(&variant_performance, 1, 2, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(2));
        assert_eq!(result.top_variants.len(), 2);

        // When k_min=1, k_max=1, should return k=1
        let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert_eq!(result.top_variants.len(), 1);
        assert!(result.top_variants.contains(&"a".to_string()));
    }

    /// Test that a single variant is trivially identified as top-1.
    #[test]
    fn test_check_topk_single_variant() {
        // Single variant - should be identified as top-1
        // It needs to beat >= (1 - 1) = 0 variants, which it does trivially
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs_with_bounds("a", 0.5, 0.7)].into_iter().collect();

        let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert_eq!(result.top_variants, vec!["a".to_string()]);
    }

    /// Test that check_topk wrapper correctly calls check_topk_stopping with k_min = k_max.
    #[test]
    fn test_check_topk_wrapper() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.7, 0.9),
            mock_cs_with_bounds("b", 0.3, 0.5),
            mock_cs_with_bounds("c", 0.2, 0.4),
        ]
        .into_iter()
        .collect();

        let result = check_topk(&variant_performance, 1, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
    }

    /// Test that epsilon tolerance enables top-1 stopping for nearly-separated intervals.
    #[test]
    fn test_check_topk_stopping_epsilon_enables_stopping() {
        // Variant A: [0.48, 0.7] - lower bound just below B's upper
        // Variant B: [0.3, 0.5] - upper bound at 0.5
        // Variant C: [0.2, 0.4] - clearly worse
        //
        // Without epsilon: A's lower (0.48) < B's upper (0.5), so A doesn't beat B
        // A only beats C (0.48 > 0.4), so num_beaten = 1
        // For top-1, need to beat >= 2, so no stopping
        //
        // With epsilon = 0.05: A beats B if 0.5 - 0.05 < 0.48, i.e., 0.45 < 0.48 ✓
        // A now beats both B and C, so top-1 is identified
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.48, 0.7),
            mock_cs_with_bounds("b", 0.3, 0.5),
            mock_cs_with_bounds("c", 0.2, 0.4),
        ]
        .into_iter()
        .collect();

        // Without epsilon, can't identify top-1
        let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
        assert!(!result.stopped);

        // With epsilon = 0.05, top-1 is identified
        let result = check_topk_stopping(&variant_performance, 1, 1, Some(0.05)).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert!(result.top_variants.contains(&"a".to_string()));
    }

    /// Test that epsilon tolerance enables identifying a larger top-k set.
    #[test]
    fn test_check_topk_stopping_epsilon_enables_larger_k() {
        // Variant A: [0.7, 0.9] - clearly best
        // Variant B: [0.48, 0.65] - B's lower (0.48) just below C's upper (0.5)
        // Variant C: [0.3, 0.5] - middle
        // Variant D: [0.1, 0.3] - clearly worst
        //
        // Without epsilon:
        // A beats 3 (all uppers < 0.7)
        // B beats 1 (only D's upper 0.3 < 0.48)
        // C beats 1 (only D's upper 0.3 < 0.3? No, 0.3 is not < 0.3)
        // Actually C's lower is 0.3, D's upper is 0.3, so C beats 0 (0.3 is not < 0.3)
        // For top-2, need to beat >= 2. Only A qualifies.
        //
        // With epsilon = 0.05:
        // B beats C if 0.5 - 0.05 < 0.48, i.e., 0.45 < 0.48 ✓
        // B beats D if 0.3 - 0.05 < 0.48, i.e., 0.25 < 0.48 ✓
        // So B now beats 2, qualifying for top-2
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.7, 0.9),
            mock_cs_with_bounds("b", 0.48, 0.65),
            mock_cs_with_bounds("c", 0.3, 0.5),
            mock_cs_with_bounds("d", 0.1, 0.3),
        ]
        .into_iter()
        .collect();

        // Without epsilon, can identify top-1 but not top-2
        let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));

        let result = check_topk_stopping(&variant_performance, 2, 2, None).unwrap();
        assert!(!result.stopped);

        // With epsilon = 0.05, can identify top-2
        let result = check_topk_stopping(&variant_performance, 2, 2, Some(0.05)).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(2));
        assert!(result.top_variants.contains(&"a".to_string()));
        assert!(result.top_variants.contains(&"b".to_string()));
    }

    /// Test that epsilon=0.0 behaves identically to epsilon=None.
    #[test]
    fn test_check_topk_stopping_epsilon_zero_same_as_none() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.7, 0.9),
            mock_cs_with_bounds("b", 0.3, 0.5),
            mock_cs_with_bounds("c", 0.2, 0.4),
        ]
        .into_iter()
        .collect();

        let result_none = check_topk_stopping(&variant_performance, 1, 2, None).unwrap();
        let result_zero = check_topk_stopping(&variant_performance, 1, 2, Some(0.0)).unwrap();

        assert_eq!(result_none.stopped, result_zero.stopped);
        assert_eq!(result_none.k, result_zero.k);
        // Note: top_variants order might differ, so just check same elements
        assert_eq!(
            result_none.top_variants.len(),
            result_zero.top_variants.len()
        );
    }

    /// Test that a very large epsilon makes all variants beat all others.
    #[test]
    fn test_check_topk_stopping_large_epsilon_all_beat_all() {
        // With a very large epsilon, every variant "beats" every other variant
        // (since ub - epsilon will be negative for any reasonable upper bound)
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.1, 0.2),
            mock_cs_with_bounds("b", 0.3, 0.4),
            mock_cs_with_bounds("c", 0.5, 0.6),
        ]
        .into_iter()
        .collect();

        // With epsilon = 1.0, all variants beat the other 2 variants
        // (a variant never counts itself as beaten)
        // For top-1, need >= 2. All qualify, so we get a top-3 (largest k checked first)
        let result = check_topk_stopping(&variant_performance, 1, 3, Some(1.0)).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(3));
        assert_eq!(result.top_variants.len(), 3);
    }

    /// Test that a variant never counts itself as beaten.
    #[test]
    fn test_check_topk_stopping_variant_does_not_beat_itself() {
        // With 2 variants and large epsilon, each variant beats the other but not itself.
        // So each variant beats exactly 1 other variant.
        // For top-1: need to beat >= 1. Both qualify.
        // For top-2: need to beat >= 0. Both qualify.
        // The key point: with epsilon=1.0, if variants counted themselves, they'd each
        // beat 2 variants and top-1 would be identified. But they shouldn't count themselves.
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.1, 0.2),
            mock_cs_with_bounds("b", 0.3, 0.4),
        ]
        .into_iter()
        .collect();

        // With epsilon = 1.0, each variant beats exactly 1 other (not itself)
        // For top-1, need to beat >= 1. Both qualify, so we get top-2.
        let result = check_topk_stopping(&variant_performance, 1, 2, Some(1.0)).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(2));
        assert_eq!(result.top_variants.len(), 2);

        // If variants incorrectly counted themselves, each would beat 2 variants,
        // and top-1 would be viable (need >= 1 beaten). Verify top-1 does NOT
        // incorrectly select just one variant.
        let result = check_topk_stopping(&variant_performance, 1, 1, Some(1.0)).unwrap();
        // Both variants beat exactly 1 other, so top-1 requires beating >= 1, which both do.
        // Since both qualify but we can only pick 1, this should still stop with k=1.
        // The important thing is the count is 1 (not 2).
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
    }

    /// Test that k_max caps the returned k, even when larger k values are viable.
    #[test]
    fn test_check_topk_stopping_returns_largest_viable_k() {
        // Variant A: [0.8, 0.95] - beats all 4 others
        // Variant B: [0.7, 0.85] - beats C, D, E (3 others)
        // Variant C: [0.5, 0.65] - beats D, E (2 others)
        // Variant D: [0.3, 0.45] - beats E (1 other)
        // Variant E: [0.1, 0.25] - beats none
        //
        // For top-1: need to beat >= 4. Only A qualifies.
        // For top-2: need to beat >= 3. A and B qualify.
        // For top-3: need to beat >= 2. A, B, C qualify.
        // For top-4: need to beat >= 1. A, B, C, D qualify.
        // For top-5: need to beat >= 0. All qualify.
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.8, 0.95),
            mock_cs_with_bounds("b", 0.7, 0.85),
            mock_cs_with_bounds("c", 0.5, 0.65),
            mock_cs_with_bounds("d", 0.3, 0.45),
            mock_cs_with_bounds("e", 0.1, 0.25),
        ]
        .into_iter()
        .collect();

        // k_min=1, k_max=5: should return k=5 (largest viable)
        let result = check_topk_stopping(&variant_performance, 1, 5, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(5));
        assert_eq!(result.top_variants.len(), 5);

        // k_min=1, k_max=3: should return k=3 (largest viable within range)
        let result = check_topk_stopping(&variant_performance, 1, 3, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(3));
        assert_eq!(result.top_variants.len(), 3);
        assert!(result.top_variants.contains(&"a".to_string()));
        assert!(result.top_variants.contains(&"b".to_string()));
        assert!(result.top_variants.contains(&"c".to_string()));

        // k_min=1, k_max=2: should return k=2
        let result = check_topk_stopping(&variant_performance, 1, 2, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(2));
        assert_eq!(result.top_variants.len(), 2);
        assert!(result.top_variants.contains(&"a".to_string()));
        assert!(result.top_variants.contains(&"b".to_string()));

        // k_min=2, k_max=4: should return k=4 (largest viable within range)
        let result = check_topk_stopping(&variant_performance, 2, 4, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(4));
        assert_eq!(result.top_variants.len(), 4);
    }

    /// Test that no stopping occurs when no k in the range is viable.
    #[test]
    fn test_check_topk_stopping_k_range_no_viable_k() {
        // Variant A: [0.4, 0.7]
        // Variant B: [0.35, 0.65]
        // Variant C: [0.3, 0.6]
        // All intervals overlap significantly - no one beats anyone
        //
        // For any k < 3, we need variants to beat others, but none do.
        // Only k=3 works (need to beat >= 0).
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.4, 0.7),
            mock_cs_with_bounds("b", 0.35, 0.65),
            mock_cs_with_bounds("c", 0.3, 0.6),
        ]
        .into_iter()
        .collect();

        // k_min=1, k_max=2: neither k=1 nor k=2 is viable, so no stopping
        let result = check_topk_stopping(&variant_performance, 1, 2, None).unwrap();
        assert!(!result.stopped);
        assert!(result.k.is_none());

        // k_min=1, k_max=3: k=3 is viable (all variants beat >= 0 others)
        let result = check_topk_stopping(&variant_performance, 1, 3, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(3));

        // k_min=3, k_max=3: k=3 is viable
        let result = check_topk_stopping(&variant_performance, 3, 3, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(3));
    }

    /// Test a "gap" scenario where k=1 and k=3 are viable but k=2 is not.
    #[test]
    fn test_check_topk_stopping_k_range_partial_viability() {
        // Variant A: [0.7, 0.9] - beats B and C
        // Variant B: [0.4, 0.6] - beats no one (C's upper 0.55 > B's lower 0.4)
        // Variant C: [0.35, 0.55] - beats no one
        //
        // A beats 2 (both uppers < 0.7)
        // B beats 0
        // C beats 0
        //
        // For top-1: need >= 2. Only A qualifies. ✓
        // For top-2: need >= 1. Only A qualifies (1 variant). ✗
        // For top-3: need >= 0. All qualify. ✓
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.7, 0.9),
            mock_cs_with_bounds("b", 0.4, 0.6),
            mock_cs_with_bounds("c", 0.35, 0.55),
        ]
        .into_iter()
        .collect();

        // k_min=1, k_max=3: k=3 is largest viable
        let result = check_topk_stopping(&variant_performance, 1, 3, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(3));

        // k_min=1, k_max=2: only k=1 is viable (k=2 fails)
        let result = check_topk_stopping(&variant_performance, 1, 2, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert_eq!(result.top_variants, vec!["a".to_string()]);

        // k_min=2, k_max=2: k=2 is not viable, no stopping
        let result = check_topk_stopping(&variant_performance, 2, 2, None).unwrap();
        assert!(!result.stopped);

        // k_min=2, k_max=3: k=3 is viable
        let result = check_topk_stopping(&variant_performance, 2, 3, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(3));
    }

    // ============================================================================
    // Tests for compute_updates
    // ============================================================================

    /// Test that compute_updates handles empty results gracefully (no changes to CS maps).
    #[test]
    fn test_compute_updates_empty_results() {
        let scoring_fn = FirstEvaluatorScore;
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
                .into_iter()
                .collect();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
                .into_iter()
                .collect();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [("evaluator1".to_string(), mock_fresh_cs("evaluator1"))]
                .into_iter()
                .collect();

        // Empty results should not change any confidence sequences
        let results_by_datapoint: BatchResultsByDatapoint = HashMap::new();
        let result = compute_updates(
            &results_by_datapoint,
            &scoring_fn,
            &mut variant_performance,
            &mut variant_failures,
            &mut evaluator_failures,
        );

        assert!(result.is_ok());
        // Count should still be 0 since no observations were processed
        assert_eq!(variant_performance["test_variant"].count, 0);
        assert_eq!(variant_failures["test_variant"].count, 0);
        assert_eq!(evaluator_failures["evaluator1"].count, 0);
    }

    /// Test that cancelled tasks are skipped and don't affect any confidence sequences.
    #[test]
    fn test_compute_updates_only_cancelled() {
        let scoring_fn = FirstEvaluatorScore;
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
                .into_iter()
                .collect();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
                .into_iter()
                .collect();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [("evaluator1".to_string(), mock_fresh_cs("evaluator1"))]
                .into_iter()
                .collect();

        // Cancelled results should not affect any statistics
        // In the new structure, we group by datapoint, then by variant
        // Cancelled tasks are skipped entirely in process_topk_batch,
        // so an empty map represents the case of only cancelled tasks
        let results_by_datapoint: BatchResultsByDatapoint = HashMap::new();

        let result = compute_updates(
            &results_by_datapoint,
            &scoring_fn,
            &mut variant_performance,
            &mut variant_failures,
            &mut evaluator_failures,
        );

        assert!(result.is_ok());
        // Count should still be 0 since cancelled tasks don't count
        assert_eq!(variant_performance["test_variant"].count, 0);
        assert_eq!(variant_failures["test_variant"].count, 0);
        assert_eq!(evaluator_failures["evaluator1"].count, 0);
    }

    /// Test that variant-level errors (BatchItemResult::Error) update failure CS but not performance/evaluator CS.
    #[test]
    fn test_compute_updates_variant_failures() {
        let scoring_fn = FirstEvaluatorScore;
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
                .into_iter()
                .collect();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
                .into_iter()
                .collect();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();

        // Create two errors for two different datapoints
        let datapoint_id_1 = Uuid::now_v7();
        let datapoint_id_2 = Uuid::now_v7();

        // Structure: datapoint_id -> variant_name -> result
        let results_by_datapoint: BatchResultsByDatapoint = [
            (
                datapoint_id_1,
                [(
                    "test_variant".to_string(),
                    BatchItemResult::Error(DatapointVariantError {
                        datapoint_id: datapoint_id_1,
                        variant: None,
                        message: "test error 1".to_string(),
                    }),
                )]
                .into_iter()
                .collect(),
            ),
            (
                datapoint_id_2,
                [(
                    "test_variant".to_string(),
                    BatchItemResult::Error(DatapointVariantError {
                        datapoint_id: datapoint_id_2,
                        variant: None,
                        message: "test error 2".to_string(),
                    }),
                )]
                .into_iter()
                .collect(),
            ),
        ]
        .into_iter()
        .collect();

        let result = compute_updates(
            &results_by_datapoint,
            &scoring_fn,
            &mut variant_performance,
            &mut variant_failures,
            &mut evaluator_failures,
        );

        assert!(result.is_ok());
        // variant_failures should have 2 observations (both 1.0 = failure)
        assert_eq!(variant_failures["test_variant"].count, 2);
        // Performance should not be updated since there were no successes
        assert_eq!(variant_performance["test_variant"].count, 0);
    }

    /// Test that results for variants not in the CS maps are silently ignored.
    #[test]
    fn test_compute_updates_missing_variant_in_map() {
        let scoring_fn = FirstEvaluatorScore;
        // Don't include "test_variant" in confidence sequence maps - should handle gracefully
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            HashMap::new();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();

        let datapoint_id = Uuid::now_v7();
        let results_by_datapoint: BatchResultsByDatapoint = [(
            datapoint_id,
            [(
                "test_variant".to_string(),
                BatchItemResult::Error(DatapointVariantError {
                    datapoint_id,
                    variant: None,
                    message: "test error".to_string(),
                }),
            )]
            .into_iter()
            .collect(),
        )]
        .into_iter()
        .collect();

        // Should not panic, just skip updates for missing variants
        let result = compute_updates(
            &results_by_datapoint,
            &scoring_fn,
            &mut variant_performance,
            &mut variant_failures,
            &mut evaluator_failures,
        );

        assert!(result.is_ok());
        // No maps should be modified since variant wasn't in any of them
        assert!(variant_performance.is_empty());
        assert!(variant_failures.is_empty());
        assert!(evaluator_failures.is_empty());
    }

    /// Test that successful evaluations update performance and evaluator CS correctly.
    #[test]
    fn test_compute_updates_successful_evaluations() {
        let scoring_fn = FirstEvaluatorScore;
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            ("variant_a".to_string(), mock_fresh_cs("variant_a")),
            ("variant_b".to_string(), mock_fresh_cs("variant_b")),
        ]
        .into_iter()
        .collect();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> = [
            ("variant_a".to_string(), mock_fresh_cs("variant_a")),
            ("variant_b".to_string(), mock_fresh_cs("variant_b")),
        ]
        .into_iter()
        .collect();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [("evaluator1".to_string(), mock_fresh_cs("evaluator1"))]
                .into_iter()
                .collect();

        let datapoint_id = Uuid::now_v7();

        // Both variants succeed with different scores
        let results_by_datapoint: BatchResultsByDatapoint = [(
            datapoint_id,
            [
                (
                    "variant_a".to_string(),
                    mock_success(
                        datapoint_id,
                        "variant_a",
                        [("evaluator1".to_string(), Ok(Some(json!(0.8))))]
                            .into_iter()
                            .collect(),
                    ),
                ),
                (
                    "variant_b".to_string(),
                    mock_success(
                        datapoint_id,
                        "variant_b",
                        [("evaluator1".to_string(), Ok(Some(json!(0.6))))]
                            .into_iter()
                            .collect(),
                    ),
                ),
            ]
            .into_iter()
            .collect(),
        )]
        .into_iter()
        .collect();

        let result = compute_updates(
            &results_by_datapoint,
            &scoring_fn,
            &mut variant_performance,
            &mut variant_failures,
            &mut evaluator_failures,
        );

        assert!(result.is_ok());
        // Both variants should have 1 performance observation
        assert_eq!(variant_performance["variant_a"].count, 1);
        assert_eq!(variant_performance["variant_b"].count, 1);
        // Both variants should have 1 failure observation (0.0 = success)
        assert_eq!(variant_failures["variant_a"].count, 1);
        assert_eq!(variant_failures["variant_b"].count, 1);
        // Evaluator should have 2 observations (one per variant)
        assert_eq!(evaluator_failures["evaluator1"].count, 2);
    }

    /// Test that evaluator errors within successful inferences update evaluator failure CS.
    #[test]
    fn test_compute_updates_evaluator_failures() {
        let scoring_fn = FirstEvaluatorScore;
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
                .into_iter()
                .collect();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
                .into_iter()
                .collect();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> = [
            ("evaluator1".to_string(), mock_fresh_cs("evaluator1")),
            ("evaluator2".to_string(), mock_fresh_cs("evaluator2")),
        ]
        .into_iter()
        .collect();

        let datapoint_id = Uuid::now_v7();

        // Variant succeeds but one evaluator fails
        let results_by_datapoint: BatchResultsByDatapoint = [(
            datapoint_id,
            [(
                "test_variant".to_string(),
                mock_success(
                    datapoint_id,
                    "test_variant",
                    [
                        ("evaluator1".to_string(), Ok(Some(json!(0.7)))),
                        (
                            "evaluator2".to_string(),
                            Err(anyhow::anyhow!("evaluator error")),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                ),
            )]
            .into_iter()
            .collect(),
        )]
        .into_iter()
        .collect();

        let result = compute_updates(
            &results_by_datapoint,
            &scoring_fn,
            &mut variant_performance,
            &mut variant_failures,
            &mut evaluator_failures,
        );

        assert!(result.is_ok());
        // Variant succeeded so variant_failures should have 1 observation (0.0)
        assert_eq!(variant_failures["test_variant"].count, 1);
        // Performance should be updated (scoring function uses first successful evaluator)
        assert_eq!(variant_performance["test_variant"].count, 1);
        // evaluator1 succeeded, evaluator2 failed
        assert_eq!(evaluator_failures["evaluator1"].count, 1);
        assert_eq!(evaluator_failures["evaluator2"].count, 1);
    }

    /// Test mixed scenario: one variant succeeds, one fails - verify correct CS updates for each.
    #[test]
    fn test_compute_updates_mixed_success_and_failure() {
        let scoring_fn = FirstEvaluatorScore;
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            ("variant_a".to_string(), mock_fresh_cs("variant_a")),
            ("variant_b".to_string(), mock_fresh_cs("variant_b")),
        ]
        .into_iter()
        .collect();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> = [
            ("variant_a".to_string(), mock_fresh_cs("variant_a")),
            ("variant_b".to_string(), mock_fresh_cs("variant_b")),
        ]
        .into_iter()
        .collect();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [("evaluator1".to_string(), mock_fresh_cs("evaluator1"))]
                .into_iter()
                .collect();

        let datapoint_id = Uuid::now_v7();

        // variant_a succeeds, variant_b fails
        let results_by_datapoint: BatchResultsByDatapoint = [(
            datapoint_id,
            [
                (
                    "variant_a".to_string(),
                    mock_success(
                        datapoint_id,
                        "variant_a",
                        [("evaluator1".to_string(), Ok(Some(json!(0.9))))]
                            .into_iter()
                            .collect(),
                    ),
                ),
                (
                    "variant_b".to_string(),
                    BatchItemResult::Error(DatapointVariantError {
                        datapoint_id,
                        variant: None,
                        message: "variant error".to_string(),
                    }),
                ),
            ]
            .into_iter()
            .collect(),
        )]
        .into_iter()
        .collect();

        let result = compute_updates(
            &results_by_datapoint,
            &scoring_fn,
            &mut variant_performance,
            &mut variant_failures,
            &mut evaluator_failures,
        );

        assert!(result.is_ok());
        // variant_a succeeded: performance and failures updated
        assert_eq!(variant_performance["variant_a"].count, 1);
        assert_eq!(variant_failures["variant_a"].count, 1);
        // variant_b failed: only failures updated (count as failure), no performance
        assert_eq!(variant_performance["variant_b"].count, 0);
        assert_eq!(variant_failures["variant_b"].count, 1);
        // evaluator only ran for variant_a
        assert_eq!(evaluator_failures["evaluator1"].count, 1);
    }
}
