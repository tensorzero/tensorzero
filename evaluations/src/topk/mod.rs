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

use anyhow::Result;
use tracing::info;
use uuid::Uuid;

use crate::BatchItemResult;
use crate::betting_confidence_sequences::{MeanBettingConfidenceSequence, update_betting_cs};
use crate::evaluators::EvaluationResult;

// ============================================================================
// Core Types
// ============================================================================

/// Status of a variant in the top-k evaluation process.
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone)]
pub struct TopKStoppingResult {
    /// Whether a top-k set was identified
    pub stopped: bool,
    /// The k value for which stopping occurred (if stopped)
    pub k: Option<u32>,
    /// Names of variants confidently in the top-k (if stopped)
    pub top_variants: Vec<String>,
}

/// Output from a completed top-k evaluation task.
#[derive(Debug)]
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

/// Parameters for updating variant statuses during top-k evaluation.
/// TODO: remove #[cfg(test)] once other functions that use this are implemented
#[cfg(test)]
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
/// TODO: remove #[cfg(test)] once other functions that use this are implemented
#[cfg(test)]
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

#[cfg(test)]
mod tests;
