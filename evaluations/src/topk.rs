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
    // Tests for compute_updates
    // ============================================================================

    /// Test that compute_updates handles empty results gracefully (no changes to CS maps).
    #[test]
    fn test_compute_updates_empty_results() {
        let scoring_fn = FirstEvaluatorScore;
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs_with_bounds("test_variant", 0.3, 0.7)]
                .into_iter()
                .collect();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs_with_bounds("test_variant", 0.1, 0.4)]
                .into_iter()
                .collect();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs_with_bounds("evaluator1", 0.05, 0.2)]
                .into_iter()
                .collect();

        let results_by_datapoint: BatchResultsByDatapoint = HashMap::new();
        let result = compute_updates(
            &results_by_datapoint,
            &scoring_fn,
            &mut variant_performance,
            &mut variant_failures,
            &mut evaluator_failures,
        );

        assert!(result.is_ok());

        // Verify variant_performance is completely unchanged
        let vp = &variant_performance["test_variant"];
        assert_eq!(vp.name, "test_variant");
        assert_eq!(vp.mean_regularized, 0.5); // (0.3 + 0.7) / 2
        assert_eq!(vp.variance_regularized, 0.1);
        assert_eq!(vp.count, 100);
        assert_eq!(vp.mean_est, 0.5);
        assert_eq!(vp.cs_lower, 0.3);
        assert_eq!(vp.cs_upper, 0.7);
        assert_eq!(vp.alpha, 0.05);
        assert_eq!(vp.wealth.wealth_upper, vec![1.0; 101]);
        assert_eq!(vp.wealth.wealth_lower, vec![1.0; 101]);

        // Verify variant_failures is completely unchanged
        let vf = &variant_failures["test_variant"];
        assert_eq!(vf.name, "test_variant");
        assert_eq!(vf.mean_regularized, 0.25); // (0.1 + 0.4) / 2
        assert_eq!(vf.variance_regularized, 0.1);
        assert_eq!(vf.count, 100);
        assert_eq!(vf.mean_est, 0.25);
        assert_eq!(vf.cs_lower, 0.1);
        assert_eq!(vf.cs_upper, 0.4);
        assert_eq!(vf.alpha, 0.05);
        assert_eq!(vf.wealth.wealth_upper, vec![1.0; 101]);
        assert_eq!(vf.wealth.wealth_lower, vec![1.0; 101]);

        // Verify evaluator_failures is completely unchanged
        let ef = &evaluator_failures["evaluator1"];
        assert_eq!(ef.name, "evaluator1");
        assert_eq!(ef.mean_regularized, 0.125); // (0.05 + 0.2) / 2
        assert_eq!(ef.variance_regularized, 0.1);
        assert_eq!(ef.count, 100);
        assert_eq!(ef.mean_est, 0.125);
        assert_eq!(ef.cs_lower, 0.05);
        assert_eq!(ef.cs_upper, 0.2);
        assert_eq!(ef.alpha, 0.05);
        assert_eq!(ef.wealth.wealth_upper, vec![1.0; 101]);
        assert_eq!(ef.wealth.wealth_lower, vec![1.0; 101]);
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
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [("evaluator1".to_string(), mock_fresh_cs("evaluator1"))]
                .into_iter()
                .collect();

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
        // evaluator_failures should not be updated (variant failed before evaluation ran)
        assert_eq!(evaluator_failures["evaluator1"].count, 0);
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

        // Check variant_a performance (observation = 0.8)
        // mean_regularized = (0.5 * 1 + 0.8) / 2 = 0.65
        // variance_regularized = (0.25 * 1 + (0.8 - 0.65)^2) / 2 = 0.13625
        // mean_est = 0.8
        let vp_a = &variant_performance["variant_a"];
        assert_eq!(vp_a.count, 1);
        assert!((vp_a.mean_regularized - 0.65).abs() < 1e-10);
        assert!((vp_a.variance_regularized - 0.13625).abs() < 1e-10);
        assert!((vp_a.mean_est - 0.8).abs() < 1e-10);

        // Check variant_b performance (observation = 0.6)
        // mean_regularized = (0.5 * 1 + 0.6) / 2 = 0.55
        // variance_regularized = (0.25 * 1 + (0.6 - 0.55)^2) / 2 = 0.12625
        // mean_est = 0.6
        let vp_b = &variant_performance["variant_b"];
        assert_eq!(vp_b.count, 1);
        assert!((vp_b.mean_regularized - 0.55).abs() < 1e-10);
        assert!((vp_b.variance_regularized - 0.12625).abs() < 1e-10);
        assert!((vp_b.mean_est - 0.6).abs() < 1e-10);

        // Check variant_a failures (observation = 0.0, success)
        // mean_regularized = (0.5 * 1 + 0.0) / 2 = 0.25
        // variance_regularized = (0.25 * 1 + (0.0 - 0.25)^2) / 2 = 0.15625
        // mean_est = 0.0
        let vf_a = &variant_failures["variant_a"];
        assert_eq!(vf_a.count, 1);
        assert!((vf_a.mean_regularized - 0.25).abs() < 1e-10);
        assert!((vf_a.variance_regularized - 0.15625).abs() < 1e-10);
        assert!((vf_a.mean_est - 0.0).abs() < 1e-10);

        // Check variant_b failures (observation = 0.0, success)
        let vf_b = &variant_failures["variant_b"];
        assert_eq!(vf_b.count, 1);
        assert!((vf_b.mean_regularized - 0.25).abs() < 1e-10);
        assert!((vf_b.variance_regularized - 0.15625).abs() < 1e-10);
        assert!((vf_b.mean_est - 0.0).abs() < 1e-10);

        // Check evaluator1 failures (2 observations, both = 0.0, success)
        // After first observation (0.0):
        //   mean_regularized = (0.5 * 1 + 0.0) / 2 = 0.25
        //   variance_regularized = (0.25 * 1 + (0.0 - 0.25)^2) / 2 = 0.15625
        // After second observation (0.0):
        //   mean_regularized = (0.25 * 2 + 0.0) / 3 = 0.16666...
        //   variance_regularized = (0.15625 * 2 + (0.0 - 0.16666...)^2) / 3 = 0.11319...
        // mean_est = (0.0 + 0.0) / 2 = 0.0
        let ef = &evaluator_failures["evaluator1"];
        assert_eq!(ef.count, 2);
        assert!((ef.mean_regularized - 1.0 / 6.0).abs() < 1e-10);
        assert!(
            (ef.variance_regularized - (0.15625 * 2.0 + (1.0_f64 / 6.0).powi(2)) / 3.0).abs()
                < 1e-10
        );
        assert!((ef.mean_est - 0.0).abs() < 1e-10);
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

    /// Test tiebreaking by mean_est when num_beaten is equal.
    #[test]
    fn test_check_topk_stopping_tiebreak_by_mean_est() {
        // With large epsilon, all variants beat all others (num_beaten = 2 for each)
        // Tiebreaker should be mean_est descending
        // Note: mock_cs_with_bounds sets mean_est = (cs_lower + cs_upper) / 2
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.1, 0.3), // mean_est = 0.2
            mock_cs_with_bounds("b", 0.4, 0.6), // mean_est = 0.5
            mock_cs_with_bounds("c", 0.7, 0.9), // mean_est = 0.8
        ]
        .into_iter()
        .collect();

        // With epsilon = 1.0, all beat all others
        // For top-1, should return "c" (highest mean_est)
        let result = check_topk_stopping(&variant_performance, 1, 1, Some(1.0)).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert_eq!(result.top_variants.len(), 1);
        assert!(result.top_variants.contains(&"c".to_string()));

        // For top-2, should return "c" and "b" (top two by mean_est)
        let result = check_topk_stopping(&variant_performance, 2, 2, Some(1.0)).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(2));
        assert_eq!(result.top_variants.len(), 2);
        assert!(result.top_variants.contains(&"c".to_string()));
        assert!(result.top_variants.contains(&"b".to_string()));
    }

    /// Test tiebreaking by cs_lower when mean_est is equal.
    #[test]
    fn test_check_topk_stopping_tiebreak_by_cs_lower() {
        // Create variants with same mean_est but different cs_lower
        // mean_est = (cs_lower + cs_upper) / 2, so we need to adjust both bounds
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            HashMap::new();

        // All have mean_est = 0.5, but different cs_lower
        let (name_a, mut cs_a) = mock_cs_with_bounds("a", 0.3, 0.7); // mean_est = 0.5, cs_lower = 0.3
        cs_a.mean_est = 0.5;
        variant_performance.insert(name_a, cs_a);

        let (name_b, mut cs_b) = mock_cs_with_bounds("b", 0.4, 0.6); // mean_est = 0.5, cs_lower = 0.4
        cs_b.mean_est = 0.5;
        variant_performance.insert(name_b, cs_b);

        let (name_c, mut cs_c) = mock_cs_with_bounds("c", 0.2, 0.8); // mean_est = 0.5, cs_lower = 0.2
        cs_c.mean_est = 0.5;
        variant_performance.insert(name_c, cs_c);

        // With epsilon = 1.0, all beat all others
        // For top-1, should return "b" (highest cs_lower since mean_est is tied)
        let result = check_topk_stopping(&variant_performance, 1, 1, Some(1.0)).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert_eq!(result.top_variants.len(), 1);
        assert!(result.top_variants.contains(&"b".to_string()));
    }

    /// Test that identical variants at the k boundary cause an enlarged result set.
    #[test]
    fn test_check_topk_stopping_ties_at_boundary_enlarges_set() {
        // Create 4 variants: one clear winner, then 3 identical variants
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            HashMap::new();

        // Clear winner
        let (name_a, cs_a) = mock_cs_with_bounds("a", 0.8, 0.9);
        variant_performance.insert(name_a, cs_a);

        // Three identical variants (same bounds, same mean_est)
        let (name_b, cs_b) = mock_cs_with_bounds("b", 0.4, 0.6);
        variant_performance.insert(name_b, cs_b);

        let (name_c, cs_c) = mock_cs_with_bounds("c", 0.4, 0.6);
        variant_performance.insert(name_c, cs_c);

        let (name_d, cs_d) = mock_cs_with_bounds("d", 0.4, 0.6);
        variant_performance.insert(name_d, cs_d);

        // With epsilon = 1.0, all beat all others (num_beaten = 3 each)
        // For top-2: "a" is clearly first, but b/c/d are tied for second
        // Should return all 4 variants (enlarged set)
        let result = check_topk_stopping(&variant_performance, 2, 2, Some(1.0)).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(2));
        assert_eq!(result.top_variants.len(), 4); // Enlarged due to ties
        assert!(result.top_variants.contains(&"a".to_string()));
        assert!(result.top_variants.contains(&"b".to_string()));
        assert!(result.top_variants.contains(&"c".to_string()));
        assert!(result.top_variants.contains(&"d".to_string()));
    }

    /// Test that non-identical variants at the k boundary do NOT enlarge the set.
    #[test]
    fn test_check_topk_stopping_no_ties_at_boundary() {
        // Create 4 variants with distinct mean_est values
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.7, 0.9), // mean_est = 0.8
            mock_cs_with_bounds("b", 0.5, 0.7), // mean_est = 0.6
            mock_cs_with_bounds("c", 0.3, 0.5), // mean_est = 0.4
            mock_cs_with_bounds("d", 0.1, 0.3), // mean_est = 0.2
        ]
        .into_iter()
        .collect();

        // With epsilon = 1.0, all beat all others
        // For top-2, should return exactly 2 variants (a and b by mean_est)
        let result = check_topk_stopping(&variant_performance, 2, 2, Some(1.0)).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(2));
        assert_eq!(result.top_variants.len(), 2); // No enlargement
        assert!(result.top_variants.contains(&"a".to_string()));
        assert!(result.top_variants.contains(&"b".to_string()));
    }

    /// Test identical variants (all have same bounds).
    #[test]
    fn test_check_topk_stopping_all_identical_variants() {
        // All variants have identical confidence sequences
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs_with_bounds("a", 0.4, 0.6),
            mock_cs_with_bounds("b", 0.4, 0.6),
            mock_cs_with_bounds("c", 0.4, 0.6),
        ]
        .into_iter()
        .collect();

        // With epsilon = 0, no one beats anyone (intervals are identical)
        // Only k=3 is viable (need to beat >= 0)
        let result = check_topk_stopping(&variant_performance, 1, 3, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(3));
        assert_eq!(result.top_variants.len(), 3);

        // With epsilon = 1.0, all beat all others, but for top-1, all are tied
        // Should return all 3 (enlarged set)
        let result = check_topk_stopping(&variant_performance, 1, 1, Some(1.0)).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert_eq!(result.top_variants.len(), 3); // All tied
    }
}
