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
use durable::{Durable, Task, TaskContext, TaskResult, async_trait};
use serde::{Deserialize, Serialize};
use sqlx_alpha::PgPool;
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::endpoints::datasets::v1::get_datapoints;
use tensorzero_core::endpoints::datasets::v1::list_datapoints;
use tensorzero_core::endpoints::datasets::v1::types::{
    GetDatapointsRequest, ListDatapointsRequest,
};
use tensorzero_core::evaluations::EvaluationConfig;
use tracing::{debug, info};
use uuid::Uuid;

use crate::betting_confidence_sequences::{MeanBettingConfidenceSequence, update_betting_cs};
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
const DEFAULT_BATCH_SIZE: usize = 20;

/// Default confidence sequence resolution (grid points for mean estimation)
const DEFAULT_CS_RESOLUTION: usize = 1001;

/// Default alpha (significance level) for confidence sequences
const DEFAULT_ALPHA: f32 = 0.05;

/// Queue name for top-k evaluation tasks
const QUEUE_NAME: &str = "evaluations_topk";

// ============================================================================
// Durable Client
// ============================================================================

/// Creates a Durable client configured for top-k evaluation tasks.
///
/// The client is configured to use the `evaluations_topk` queue, which must
/// have been created by running the database migrations.
///
/// # Arguments
/// * `pool` - A PostgreSQL connection pool
/// * `state` - Application state containing clients, configs, and scoring function
///
/// # Returns
/// A configured `Durable` client for spawning and processing `TopKTask` tasks.
pub async fn create_client(pool: PgPool, state: TopKTaskState) -> Result<Durable<TopKTaskState>> {
    let client = Durable::builder()
        .pool(pool)
        .queue_name(QUEUE_NAME)
        .build_with_state(state)
        .await?;

    client.register::<TopKTask>().await;

    Ok(client)
}

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
    /// Exhausted all available datapoints (limited by dataset size or max_datapoints config)
    DatasetExhausted,
    /// One or more evaluators have a failure rate above the threshold
    EvaluatorsFailed { evaluator_names: Vec<String> },
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

/// Output from a completed top-k evaluation task.
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
                    // Cancellation should never occur in top-k evaluation since we use empty
                    // cancellation tokens (cancellation is only used for adaptive stopping
                    // with precision_targets in lib.rs)
                    anyhow::bail!(
                        "Unexpected cancelled task in top-k evaluation. This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports"
                    );
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
/// and working down. We return the largest k for which we can identify a top-k set,
/// if such a k exists.
///
/// When epsilon > 0, it's possible for more than k variants to meet the stopping
/// threshold. In this case, we select the top k by:
/// 1. Highest `mean_est` (point estimate)
/// 2. Highest `cs_lower` (lower confidence bound) as tiebreaker
/// 3. Highest `cs_upper` (upper confidence bound) as final tiebreaker
///
/// If there are still ties at the k boundary after all tiebreakers (i.e., variants
/// with identical point estimates and confidence bounds), we return the enlarged set
/// containing all tied variants and log an info message.
///
/// # Arguments
/// * `variant_performance` - Map from variant name to its performance confidence sequence
/// * `k_min` - Minimum acceptable k value
/// * `k_max` - Maximum k value to check
/// * `epsilon` (optional) - A tolerance for performance equivalence. If None, set to 0.0.
///
/// # Returns
/// A `TopKStoppingResult` indicating whether stopping occurred and, if it did, which variants are
/// in the top-k. Note that `top_variants.len()` may exceed `k` if there are ties at the boundary.
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

    // Intermediate struct for sorting variants by their statistics
    struct VariantStats<'a> {
        name: &'a String,
        /// Number of other variants this variant "confidently beats"
        num_beaten: usize,
        mean_est: f64,
        cs_lower: f64,
        cs_upper: f64,
    }

    // For each variant, count how many other variants' upper bounds its lower bound exceeds
    // with tolerance epsilon. This tells us how many variants it "confidently beats".
    // We subtract 1 if the variant would count itself as beaten (when cs_upper - epsilon < cs_lower).
    let mut variants_with_stats: Vec<VariantStats<'_>> = variant_performance
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
            VariantStats {
                name,
                num_beaten,
                mean_est: cs.mean_est,
                cs_lower: cs.cs_lower,
                cs_upper: cs.cs_upper,
            }
        })
        .collect();

    // Sort by:
    // 1. num_beaten descending (most important - determines if variant qualifies)
    // 2. mean_est descending (primary tiebreaker)
    // 3. cs_lower descending (secondary tiebreaker)
    // 4. cs_upper descending (tertiary tiebreaker)
    variants_with_stats.sort_by(|a, b| {
        b.num_beaten
            .cmp(&a.num_beaten)
            .then_with(|| {
                b.mean_est
                    .partial_cmp(&a.mean_est)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                b.cs_lower
                    .partial_cmp(&a.cs_lower)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                b.cs_upper
                    .partial_cmp(&a.cs_upper)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Check each k from k_max down to k_min
    // We want the largest k for which we can identify a top-k set
    for k in (k_min..=k_max).rev() {
        let k_usize = k as usize;
        let threshold = num_variants.saturating_sub(k_usize);

        // Check if at least k variants beat the threshold
        let num_qualifying = variants_with_stats
            .iter()
            .filter(|v| v.num_beaten >= threshold)
            .count();

        if num_qualifying >= k_usize {
            // We have at least k qualifying variants
            // Take the top k, but check for ties at the boundary
            let kth_variant = &variants_with_stats[k_usize - 1];

            // Find all variants that are tied with the kth variant
            // (same num_beaten, mean_est, cs_lower, cs_upper)
            let mut top_variants: Vec<String> = Vec::new();
            let mut num_tied_at_boundary = 0;

            for v in &variants_with_stats {
                if v.num_beaten < threshold {
                    // This variant doesn't qualify
                    break;
                }

                let is_tied_with_kth = v.num_beaten == kth_variant.num_beaten
                    && (v.mean_est - kth_variant.mean_est).abs() < f64::EPSILON
                    && (v.cs_lower - kth_variant.cs_lower).abs() < f64::EPSILON
                    && (v.cs_upper - kth_variant.cs_upper).abs() < f64::EPSILON;

                if top_variants.len() < k_usize {
                    // Still filling the top k
                    top_variants.push(v.name.clone());
                } else if is_tied_with_kth {
                    // This variant is tied with the kth variant, include it
                    top_variants.push(v.name.clone());
                    num_tied_at_boundary += 1;
                } else {
                    // Not tied, stop here
                    break;
                }
            }

            if num_tied_at_boundary > 0 {
                info!(
                    k = k,
                    num_returned = top_variants.len(),
                    num_tied_at_boundary = num_tied_at_boundary,
                    "Top-k stopping condition met with ties at boundary; returning enlarged set"
                );
            }

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

// ============================================================================
// Durable Top-K Variant Selection Task
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

/// Durable step function to fetch and shuffle datapoint IDs.
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

/// Parameters for updating variant statuses during top-k evaluation.
struct VariantStatusParams {
    k_min: u32,
    k_max: u32,
    epsilon: f64,
    variant_failure_threshold: Option<f64>,
}

/// Updates variant statuses based on confidence sequences and top-k stopping results.
///
/// This function transitions variants from `Active` to one of the terminal states:
/// - `Failed`: Variant's failure rate confidence interval lower bound exceeds the threshold
/// - `Include`: Variant is confidently in the top k_min set
/// - `Exclude`: Variant is confidently outside the top k_max set
///
/// The checks are applied in priority order (failure > global stopping > early exclusion > early inclusion),
/// so a variant that would otherwise be included can still be marked as `Failed` if its failure rate
/// is too high.
///
/// # Status Transition Logic
///
/// 1. **Skip non-active**: Variants already in a terminal state are not modified.
///
/// 2. **Failure check**: If `variant_failure_threshold` is set and the variant's failure rate
///    CS lower bound exceeds it, mark as `Failed`.
///
/// 3. **Global stopping**: If `stopping_result.stopped` is true, mark variants in `top_variants`
///    as `Include` and all others as `Exclude`.
///
/// 4. **Early exclusion**: If at least `k_max` other variants have lower bounds above this
///    variant's upper bound (adjusted by epsilon), this variant cannot be in the top k_max,
///    so mark as `Exclude`.
///
/// 5. **Early inclusion**: If this variant's lower bound exceeds the upper bounds of at least
///    `(num_variants - k_min)` others (adjusted by epsilon), it's confidently in the top k_min,
///    so mark as `Include`.
fn update_variant_statuses(
    variant_status: &mut HashMap<String, VariantStatus>,
    variant_performance: &HashMap<String, MeanBettingConfidenceSequence>,
    variant_failures: &HashMap<String, MeanBettingConfidenceSequence>,
    stopping_result: &TopKStoppingResult,
    params: &VariantStatusParams,
) {
    let num_variants = variant_status.len();
    for (name, status) in variant_status.iter_mut() {
        // Skip already-stopped variants
        if *status != VariantStatus::Active {
            continue;
        }

        // Check for failure based on failure rate
        if let Some(threshold) = params.variant_failure_threshold
            && let Some(failure_cs) = variant_failures.get(name)
            && failure_cs.cs_lower > threshold
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

        // Check if variant can be excluded early because it's confidently outside the top k_max
        // (its upper bound is below (lower_bound_j + epsilon) for k_max other variants j)
        if let Some(perf_cs) = variant_performance.get(name) {
            let num_definitely_worse_than = variant_performance
                .iter()
                .filter(|(other_name, other_cs)| {
                    *other_name != name && perf_cs.cs_upper < other_cs.cs_lower + params.epsilon
                })
                .count();

            // If at least k_max variants are definitely better,
            // this variant cannot be in the top k_max
            if num_definitely_worse_than >= params.k_max as usize {
                *status = VariantStatus::Exclude;
                continue;
            }
        }

        // Check if variant can be included early because it's confidently within the top k_min
        // (its lower bound is above (upper_bound_j - epsilon) for all but k_min other variants j)
        if let Some(perf_cs) = variant_performance.get(name) {
            let num_definitely_better_than = variant_performance
                .iter()
                .filter(|(other_name, other_cs)| {
                    *other_name != name && perf_cs.cs_lower > other_cs.cs_upper - params.epsilon
                })
                .count();

            // If this variant beats at least (num_variants - k_min) others,
            // it's confidently in the top k_min
            if num_definitely_better_than >= num_variants - params.k_min as usize {
                *status = VariantStatus::Include;
            }
        }
    }
}

/// Durable step function to process a batch and update loop state.
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

            // Build params for process_batch() call
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

            // Call process_batch() from lib.rs
            let (mut join_set, task_id_map) =
                process_batch(&batch_params, datapoints, &active_variants).await?;

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
                        tracing::warn!(
                            datapoint_id = %error.datapoint_id,
                            variant = %variant_name,
                            error = %error.message,
                            "Batch item error in top-k evaluation"
                        );
                        (error.datapoint_id, variant_name)
                    }
                    BatchItemResult::Cancelled => {
                        // Cancellation should never occur in top-k evaluation since we use empty
                        // cancellation tokens (cancellation is only used for adaptive stopping
                        // with precision_targets in lib.rs)
                        anyhow::bail!(
                            "Unexpected cancelled task in top-k evaluation. This should never happen, please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports"
                        );
                    }
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
    let status_params = VariantStatusParams {
        k_min: params.k_min,
        k_max: params.k_max,
        epsilon: params.epsilon.unwrap_or(0.0),
        variant_failure_threshold: params.variant_failure_threshold,
    };
    update_variant_statuses(
        &mut current_state.variant_status,
        &current_state.variant_performance,
        &current_state.variant_failures,
        &stopping_result,
        &status_params,
    );

    Ok(current_state)
}

/// Check global stopping conditions in order of precedence
fn check_global_stopping(
    loop_state: &TopKLoopState,
    params: &TopKTaskParams,
) -> Option<GlobalStoppingReason> {
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
        return Some(GlobalStoppingReason::TopKFound {
            k: result.k.unwrap_or(0),
            top_variants: result.top_variants,
        });
    }

    // Check for evaluator failures
    if let Some(threshold) = params.evaluator_failure_threshold {
        let failed_evaluators: Vec<String> = loop_state
            .evaluator_failures
            .iter()
            .filter(|(_, cs)| cs.cs_lower > threshold)
            .map(|(name, _)| name.clone())
            .collect();
        if !failed_evaluators.is_empty() {
            return Some(GlobalStoppingReason::EvaluatorsFailed {
                evaluator_names: failed_evaluators,
            });
        }
    }

    // Check for too many variant failures
    let num_failed = loop_state
        .variant_status
        .values()
        .filter(|s| **s == VariantStatus::Failed)
        .count();
    let num_variants = loop_state.variant_status.len();
    if num_failed > num_variants.saturating_sub(params.k_min as usize) {
        return Some(GlobalStoppingReason::TooManyVariantsFailed { num_failed });
    }

    // If none of the three reasons above apply
    None
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

        // Log if fewer datapoints are available than the configured max
        if let Some(max) = params.max_datapoints
            && total_datapoints < max
        {
            info!(
                "Dataset contains {total_datapoints} datapoints, fewer than max_datapoints ({max})"
            );
        }

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
                        MeanBettingConfidenceSequence::new(
                            name.clone(),
                            DEFAULT_CS_RESOLUTION,
                            DEFAULT_ALPHA,
                        ),
                    )
                })
                .collect(),
            variant_failures: params
                .variant_names
                .iter()
                .map(|name| {
                    (
                        name.clone(),
                        MeanBettingConfidenceSequence::new(
                            name.clone(),
                            DEFAULT_CS_RESOLUTION,
                            DEFAULT_ALPHA,
                        ),
                    )
                })
                .collect(),
            evaluator_failures: evaluator_names
                .iter()
                .map(|name| {
                    (
                        name.clone(),
                        MeanBettingConfidenceSequence::new(
                            name.clone(),
                            DEFAULT_CS_RESOLUTION,
                            DEFAULT_ALPHA,
                        ),
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

        let mut stopping_reason: Option<GlobalStoppingReason> = None;

        for (batch_idx, batch_ids) in batches.into_iter().enumerate() {
            let current_batch_size = batch_ids.len();
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

            debug!(
                batch_number = batch_idx + 1,
                batch_size = current_batch_size,
                "Processing batch"
            );

            if let Some(reason) = check_global_stopping(&loop_state, &params) {
                stopping_reason = Some(reason);
                break;
            }
        }

        // If we exited without an explicit stopping reason then assign reason DatasetExhausted.
        let stopping_reason = stopping_reason.unwrap_or(GlobalStoppingReason::DatasetExhausted);

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

#[cfg(test)]
mod tests;
