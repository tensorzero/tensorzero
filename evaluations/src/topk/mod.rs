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
use serde::{Deserialize, Serialize};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::evaluations::EvaluationConfig;
use tracing::info;
use uuid::Uuid;

use crate::betting_confidence_sequences::{MeanBettingConfidenceSequence, update_betting_cs};
use crate::evaluators::EvaluationResult;
use crate::{BatchItemResult, Clients, EvaluationFunctionConfigTable};

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
const _QUEUE_NAME: &str = "evaluations_topk"; // TODO: Remove underscore once used

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

/// Application state for the top-k task (non-serializable global resources).
///
/// This struct contains only global resources that don't change between task runs.
/// All task-specific configuration is stored in `TopKTaskParams` for durable execution.
#[derive(Clone)]
pub struct TopKTaskState {
    /// Clients for inference and database access
    pub clients: Arc<Clients>,
}

/// Serializable parameters for the top-k durable task.
///
/// All task-specific configuration is stored here to enable durable execution.
/// The exact config is captured at task creation time and doesn't change on resumption.
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
    /// Failure rate threshold for variants.
    /// If a variant's failure rate confidence sequence lower bound exceeds this threshold,
    /// the variant is marked as Failed and excluded from further evaluation.
    /// Failed variants are not candidates for the returned top-k set.
    pub variant_failure_threshold: Option<f64>,
    /// Failure rate threshold for evaluators.
    /// If any evaluator's failure rate confidence sequence lower bound exceeds this threshold,
    /// the top-k identification run terminates.
    pub evaluator_failure_threshold: Option<f64>,
    /// Number of concurrent requests
    pub concurrency: usize,
    /// Cache mode for inference
    pub inference_cache: CacheEnabledMode,
    /// Evaluation configuration (captured at task creation time)
    pub evaluation_config: EvaluationConfig,
    /// Function configs table (captured at task creation time)
    pub function_configs: EvaluationFunctionConfigTable,
    /// Scoring function type (converted to trait object at runtime)
    pub scoring_function: ScoringFunctionType,
}

/// Serializable progress state for checkpointing between batches of evaluations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopKProgress {
    /// Variant statuses, tracking early stopping
    pub variant_status: HashMap<String, VariantStatus>,
    /// Variant performance confidence sequences
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

/// Serializable enum representing available scoring function types.
///
/// This enum allows scoring functions to be specified in task parameters (which must be
/// serializable for durable execution) and then converted to trait objects at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoringFunctionType {
    /// Average all non-failed evaluator scores for each variant.
    AverageEvaluatorScore,
}

impl ScoringFunctionType {
    /// Convert this enum variant to a trait object for use in scoring.
    pub fn into_scoring_fn(self) -> Arc<dyn ScoringFunction> {
        match self {
            ScoringFunctionType::AverageEvaluatorScore => Arc::new(AverageEvaluatorScore),
        }
    }
}

/// A scoring function that averages all non-failed evaluator scores for each variant.
///
/// For each variant, this function:
/// 1. Iterates through all evaluator results
/// 2. For successful results, extracts numeric values (or converts booleans to 0/1)
/// 3. Returns the mean of all successful evaluator scores
///
/// Variants with no successful evaluator results are omitted from the output.
pub struct AverageEvaluatorScore;

impl ScoringFunction for AverageEvaluatorScore {
    fn score(&self, evaluations: &HashMap<String, &EvaluationResult>) -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        for (variant_name, eval_result) in evaluations {
            let mut sum = 0.0;
            let mut count = 0;
            for result in eval_result.values() {
                if let Ok(Some(value)) = result {
                    // Handle both numeric and boolean values
                    if let Some(num) = value.as_f64() {
                        sum += num;
                        count += 1;
                    } else if let Some(b) = value.as_bool() {
                        sum += if b { 1.0 } else { 0.0 };
                        count += 1;
                    }
                }
            }
            if count > 0 {
                scores.insert(variant_name.clone(), sum / count as f64);
            }
        }
        scores
    }
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
// Stopping Conditions: Variant-Specific Stopping and Global Stopping
// ============================================================================

/// Parameters for updating variant statuses during top-k evaluation.
///
/// If `variant_failure_threshold` is set and a variant's failure rate CS lower bound
/// exceeds it, the variant is marked as Failed. Failed variants are not candidates
/// for the returned top-k set. Additionally, early exclusion decisions (marking
/// variants as Exclude) are made relative to the current set of active variants.
/// If variants fail after others have been excluded, the evaluation may end with fewer
/// than k_min identified variants, since excluded variants are not reconsidered.
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
#[expect(dead_code)]
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

/// Check global stopping conditions in order of precedence.
///
/// Returns `Some(reason)` if the evaluation should stop, `None` otherwise.
/// Checks in order:
/// 1. TopKFound - if a top-k set can be confidently identified
/// 2. EvaluatorsFailed - if any evaluator's failure rate exceeds the threshold
/// 3. TooManyVariantsFailed - if too many variants have failed to identify top-k
///
/// The fourth available reason, DatasetExhausted, is checked elsewhere, because
/// check_global_stopping() doesn't have access to the number of batches processed
/// and remaining.
fn check_global_stopping(
    progress: &TopKProgress,
    params: &TopKTaskParams,
) -> Option<GlobalStoppingReason> {
    // Check if we identified a top-k set
    let stopping_result = check_topk_stopping(
        &progress.variant_performance,
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
        let failed_evaluators: Vec<String> = progress
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
    let num_failed = progress
        .variant_status
        .values()
        .filter(|s| **s == VariantStatus::Failed)
        .count();
    let num_variants = progress.variant_status.len();
    if num_failed > num_variants.saturating_sub(params.k_min as usize) {
        return Some(GlobalStoppingReason::TooManyVariantsFailed { num_failed });
    }

    // If none of the three reasons above apply
    None
}

// ============================================================================
// Durable Task (skeleton - implementation in separate PR)
// ============================================================================

// The durable task implementation will use the types and functions defined above:
//
// - `TopKTaskState`: Passed to `Durable::build_with_state()` to provide clients
// - `TopKTaskParams`: Serialized as task input, contains all config for reproducible runs
// - `TopKProgress`: Checkpointed between batches via `ProcessBatchStepParams`
// - `TopKTaskOutput`: Returned when task completes
//
// Key functions used in the task loop:
// - `compute_updates()`: Called after each batch to update confidence sequences
// - `update_variant_statuses()`: Called to transition variants between states
//                                and check variant-specific stopping conditions
// - `check_global_stopping()`: Called after each batch to check if we should stop
// - `check_topk_stopping()`: Called within `check_global_stopping()` for TopKFound check
//
// The task will:
// 1. Fetch and shuffle datapoint IDs (checkpointed step)
// 2. For each batch:
//    a. Fetch datapoints and run inference+evaluation on active variants
//    b. Call `compute_updates()` to update confidence sequences
//    c. Call `update_variant_statuses()` to transition variant states
//    d. Call `check_global_stopping()` to determine if we should stop
//    e. Checkpoint progress via `ProcessBatchStepParams`
// 3. Return `TopKTaskOutput` with final state and stopping reason

/// Durable step function to fetch and shuffle datapoint IDs.
///
/// Called via `ctx.step("fetch_datapoint_ids", params, fetch_datapoint_ids_step)`.
#[expect(dead_code)]
async fn fetch_datapoint_ids_step(
    _function_name: String,
    _dataset_name: String,
    _max_datapoints: Option<usize>,
    _state: TopKTaskState,
) {
    // Implementation will:
    // 1. Call list_datapoints() to get datapoint IDs from ClickHouse
    // 2. Shuffle the IDs for randomized evaluation order
    // 3. Return the shuffled IDs (checkpointed by durable framework)
}

/// Durable step function to process a batch and update progress state.
///
/// Called via `ctx.step(&format!("batch_{batch_idx}"), params, process_batch_step)`.
#[expect(dead_code)]
async fn process_batch_step(
    _batch_ids: Vec<Uuid>,
    _progress: TopKProgress,
    _params: &TopKTaskParams,
    _state: TopKTaskState,
    // ) -> Result<TopKProgress> {
) {
    // Implementation will:
    // 1. Fetch datapoints from ClickHouse via get_datapoints()
    // 2. Run inference + evaluation on active variants via process_batch()
    // 3. Call compute_updates() to update confidence sequences
    // 4. Call update_variant_statuses() to transition variant states
    // 5. Return updated TopKProgress (checkpointed by durable framework)
}

/// The durable top-k evaluation task.
#[expect(dead_code)]
struct TopKTask;

// #[async_trait]
// impl Task<TopKTaskState> for TopKTask {
//     const NAME: &'static str = "topk-evaluation";
//     type Params = TopKTaskParams;
//     type Output = TopKTaskOutput;

//     async fn run(
//         params: Self::Params,
//         mut ctx: TaskContext<TopKTaskState>,
//         _state: TopKTaskState,
//     ) -> TaskResult<Self::Output> {
//         The implementation will:
//           1. Validate input parameters (k_min > 0, k_max >= k_min, etc.)
//           2. Generate a durable evaluation_run_id via ctx.uuid7()
//           3. Checkpoint 1: Fetch and shuffle datapoint IDs via fetch_datapoint_ids_step
//           4. Initialize TopKProgress with all variants Active
//           5. For each batch:
//             - Checkpoint 2: Process batch via process_batch_step
//             - Check global stopping conditions via check_global_stopping()
//             - Break if stopping condition met
//           6. Return TopKTaskOutput with final state and stopping reason
//     }
// }

#[cfg(test)]
mod tests;
