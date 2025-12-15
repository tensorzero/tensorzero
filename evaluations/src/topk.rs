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
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::endpoints::datasets::v1::get_datapoints;
use tensorzero_core::endpoints::datasets::v1::types::GetDatapointsRequest;
use tensorzero_core::evaluations::EvaluationConfig;
use uuid::Uuid;

use crate::betting_confidence_sequences::MeanBettingConfidenceSequence;
use crate::stopping::CancellationTokens;
use crate::types::EvaluationVariant;
use crate::{Clients, EvaluationFunctionConfigTable};

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

// ============================================================================
// Top-K Orchestrator Types and Functions
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

/// Trait for scoring functions that convert evaluator results to a single score.
///
/// Implementations of this trait define how to aggregate results from multiple
/// evaluators into a single performance score in [0, 1] for each variant.
pub trait ScoringFunction: Send + Sync {
    /// Convert evaluation results to a single score in [0, 1].
    ///
    /// # Arguments
    /// * `evaluations` - Results from all evaluators for a single (datapoint, variant) pair
    ///
    /// # Returns
    /// A score in [0, 1], or None if the score cannot be computed (e.g., missing evaluator results)
    fn score(&self, evaluations: &crate::evaluators::EvaluationResult) -> Option<f64>;
}

/// Update variant statistics and confidence sequences based on evaluation results.
///
/// This function updates:
/// - `variant_performance`: Mean performance confidence sequences for each variant
/// - `variant_failures`: Failure rate confidence sequences for each variant
/// - `evaluator_failures`: Per-evaluator failure rate confidence sequences
///
/// For variant performance, scores are computed using the provided scoring function.
/// For variant failures, each `Error` result is counted as a failure (1.0) and each
/// `Success` is counted as a non-failure (0.0).
/// For evaluator failures, each evaluator error within a `Success` result is counted
/// as a failure (1.0) for that specific evaluator.
///
/// # Arguments
/// * `variant_name` - Name of the variant being updated
/// * `results` - Batch results (successes and errors) for the datapoints processed
/// * `scoring_fn` - Function to convert evaluation results to a single score in [0, 1]
/// * `variant_performance` - Map to update with performance statistics
/// * `variant_failures` - Map to update with variant failure rates
/// * `evaluator_failures` - Map to update with evaluator failure rates
///
/// # Returns
/// `Ok(())` on success, or an error if confidence sequence updates fail.
pub fn compute_updates(
    variant_name: &str,
    results: &[crate::BatchItemResult],
    scoring_fn: &dyn ScoringFunction,
    variant_performance: &mut HashMap<String, MeanBettingConfidenceSequence>,
    variant_failures: &mut HashMap<String, MeanBettingConfidenceSequence>,
    evaluator_failures: &mut HashMap<String, MeanBettingConfidenceSequence>,
) -> Result<()> {
    use crate::betting_confidence_sequences::update_betting_cs;

    // Collect observations for variant performance (scores from scoring function)
    let mut performance_observations: Vec<f64> = Vec::new();

    // Collect observations for variant failures (1.0 = failure, 0.0 = success)
    let mut failure_observations: Vec<f64> = Vec::new();

    // Collect observations for each evaluator's failure rate
    let mut evaluator_failure_observations: HashMap<String, Vec<f64>> = HashMap::new();

    for result in results {
        match result {
            crate::BatchItemResult::Success(success) => {
                // Variant didn't fail for this datapoint
                failure_observations.push(0.0);

                // Compute performance score using the scoring function
                if let Some(score) = scoring_fn.score(&success.evaluation_result) {
                    performance_observations.push(score);
                }

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
            }
            crate::BatchItemResult::Error(_) => {
                // Variant failed for this datapoint
                failure_observations.push(1.0);
                // Note: We don't update evaluator failures here since we don't have
                // evaluation results - the failure occurred before evaluation
            }
            crate::BatchItemResult::Cancelled => {
                // Skip cancelled tasks - they don't count toward any statistics
                continue;
            }
        }
    }

    // Update variant performance confidence sequence
    if !performance_observations.is_empty()
        && let Some(cs) = variant_performance.remove(variant_name)
    {
        let updated = update_betting_cs(cs, performance_observations, None)?;
        variant_performance.insert(variant_name.to_string(), updated);
    }

    // Update variant failures confidence sequence
    if !failure_observations.is_empty()
        && let Some(cs) = variant_failures.remove(variant_name)
    {
        let updated = update_betting_cs(cs, failure_observations, None)?;
        variant_failures.insert(variant_name.to_string(), updated);
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

/// Parameters for `process_topk_batch`.
pub struct ProcessTopKBatchParams<'a> {
    /// Clients for inference and database access
    pub clients: Arc<Clients>,
    /// Batch of datapoint IDs to process
    pub batch_ids: Vec<Uuid>,
    /// Map of variant names to their current status (only Active variants are processed)
    pub variant_status: &'a HashMap<String, VariantStatus>,
    /// Evaluation configuration
    pub evaluation_config: Arc<EvaluationConfig>,
    /// Function configs table
    pub function_configs: Arc<EvaluationFunctionConfigTable>,
    /// Name of the evaluation
    pub evaluation_name: Arc<String>,
    /// Evaluation run ID for tagging
    pub evaluation_run_id: Uuid,
    /// Dataset name for tagging
    pub dataset_name: Arc<String>,
    /// Cache mode for inference
    pub inference_cache: CacheEnabledMode,
    /// Semaphore for concurrency control
    pub semaphore: Arc<tokio::sync::Semaphore>,
}

/// Process a batch of datapoints for the top-k evaluation.
///
/// This function:
/// 1. Retrieves datapoints from ClickHouse by their IDs
/// 2. For each active variant, runs inference and evaluation using shared `process_batch`
/// 3. Returns results grouped by variant for subsequent `compute_updates` calls
///
/// Uses the same semaphore-based concurrency control as `run_evaluation_core_streaming`.
///
/// # Arguments
/// * `params` - Parameters for batch processing
///
/// # Returns
/// A map from variant name to batch results (successes and errors) for all datapoints in the batch.
/// Each `BatchItemResult` is either a `Success` with full evaluation data or an `Error` with failure info.
pub async fn process_topk_batch(
    params: ProcessTopKBatchParams<'_>,
) -> Result<HashMap<String, Vec<crate::BatchItemResult>>> {
    // Retrieve datapoints from ClickHouse
    let request = GetDatapointsRequest {
        ids: params.batch_ids.clone(),
    };
    let datapoints = get_datapoints(&params.clients.clickhouse_client, None, request)
        .await?
        .datapoints;

    if datapoints.is_empty() {
        tracing::warn!(
            batch_size = params.batch_ids.len(),
            "No datapoints found for batch"
        );
        return Ok(HashMap::new());
    }

    // Get active variants
    let active_variants: Vec<Arc<EvaluationVariant>> = params
        .variant_status
        .iter()
        .filter(|(_, status)| **status == VariantStatus::Active)
        .map(|(name, _)| Arc::new(EvaluationVariant::Name(name.clone())))
        .collect();

    if active_variants.is_empty() {
        tracing::debug!("No active variants to process");
        return Ok(HashMap::new());
    }

    // Use empty cancellation tokens - top-k has its own stopping logic
    let cancellation_tokens = Arc::new(CancellationTokens::default());

    // Build params for shared process_batch
    let batch_params = crate::ProcessBatchParams {
        clients: params.clients,
        function_configs: params.function_configs,
        evaluation_config: params.evaluation_config,
        evaluation_name: params.evaluation_name,
        evaluation_run_id: params.evaluation_run_id,
        dataset_name: params.dataset_name,
        inference_cache: params.inference_cache,
        semaphore: params.semaphore,
        cancellation_tokens,
    };

    // Call the shared process_batch from lib.rs
    let (mut join_set, task_id_map) =
        crate::process_batch(&batch_params, datapoints, &active_variants);

    // Collect results and group by variant
    let mut results_by_variant: HashMap<String, Vec<crate::BatchItemResult>> = HashMap::new();

    while let Some(result) = join_set.join_next_with_id().await {
        let batch_result = crate::collect_batch_result(result, &task_id_map);

        // Get variant name for grouping
        let variant_name = match &batch_result {
            crate::BatchItemResult::Success(success) => get_variant_name(&success.variant),
            crate::BatchItemResult::Error(error) => error
                .variant
                .as_ref()
                .map(|v| get_variant_name(v))
                .unwrap_or_else(|| "unknown".to_string()),
            crate::BatchItemResult::Cancelled => continue, // Skip cancelled tasks
        };

        results_by_variant
            .entry(variant_name)
            .or_default()
            .push(batch_result);
    }

    Ok(results_by_variant)
}

/// Extract variant name from EvaluationVariant.
fn get_variant_name(variant: &EvaluationVariant) -> String {
    match variant {
        EvaluationVariant::Name(name) => name.clone(),
        // Dynamic variants don't have names, use a placeholder
        EvaluationVariant::Info(_) => "dynamic_variant".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::betting_confidence_sequences::{WealthProcessGridPoints, WealthProcesses};

    /// Helper to create a mock confidence sequence with specified bounds
    fn mock_cs(
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
            [mock_cs("a", 0.5, 0.7)].into_iter().collect();

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
            [mock_cs("a", 0.5, 0.7)].into_iter().collect();

        let result = check_topk_stopping(&variant_performance, 2, 1, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("k_max"));
    }

    /// Test that k_min > num_variants returns no stopping (graceful handling).
    #[test]
    fn test_check_topk_stopping_k_min_exceeds_num_variants() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs("a", 0.5, 0.7)].into_iter().collect();

        let result = check_topk_stopping(&variant_performance, 5, 5, None).unwrap();
        assert!(!result.stopped);
        assert!(result.k.is_none());
        assert!(result.top_variants.is_empty());
    }

    /// Test that negative epsilon returns an error.
    #[test]
    fn test_check_topk_stopping_negative_epsilon_errors() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs("a", 0.7, 0.9), mock_cs("b", 0.3, 0.5)]
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
            mock_cs("a", 0.7, 0.9),
            mock_cs("b", 0.3, 0.5),
            mock_cs("c", 0.2, 0.4),
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
            mock_cs("a", 0.7, 0.9),
            mock_cs("b", 0.6, 0.8),
            mock_cs("c", 0.2, 0.4),
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
            mock_cs("a", 0.4, 0.7),
            mock_cs("b", 0.45, 0.65),
            mock_cs("c", 0.5, 0.6),
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
            mock_cs("a", 0.8, 0.9),
            mock_cs("b", 0.5, 0.7),
            mock_cs("c", 0.2, 0.4),
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
            [mock_cs("a", 0.5, 0.7)].into_iter().collect();

        let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert_eq!(result.top_variants, vec!["a".to_string()]);
    }

    /// Test that check_topk wrapper correctly calls check_topk_stopping with k_min = k_max.
    #[test]
    fn test_check_topk_wrapper() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs("a", 0.7, 0.9),
            mock_cs("b", 0.3, 0.5),
            mock_cs("c", 0.2, 0.4),
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
            mock_cs("a", 0.48, 0.7),
            mock_cs("b", 0.3, 0.5),
            mock_cs("c", 0.2, 0.4),
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
            mock_cs("a", 0.7, 0.9),
            mock_cs("b", 0.48, 0.65),
            mock_cs("c", 0.3, 0.5),
            mock_cs("d", 0.1, 0.3),
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
            mock_cs("a", 0.7, 0.9),
            mock_cs("b", 0.3, 0.5),
            mock_cs("c", 0.2, 0.4),
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
            mock_cs("a", 0.1, 0.2),
            mock_cs("b", 0.3, 0.4),
            mock_cs("c", 0.5, 0.6),
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
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs("a", 0.1, 0.2), mock_cs("b", 0.3, 0.4)]
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
            mock_cs("a", 0.8, 0.95),
            mock_cs("b", 0.7, 0.85),
            mock_cs("c", 0.5, 0.65),
            mock_cs("d", 0.3, 0.45),
            mock_cs("e", 0.1, 0.25),
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
            mock_cs("a", 0.4, 0.7),
            mock_cs("b", 0.35, 0.65),
            mock_cs("c", 0.3, 0.6),
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
            mock_cs("a", 0.7, 0.9),
            mock_cs("b", 0.4, 0.6),
            mock_cs("c", 0.35, 0.55),
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

    /// Helper to create an initial confidence sequence for testing compute_updates
    fn create_initial_cs(name: &str) -> MeanBettingConfidenceSequence {
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

    /// A simple scoring function that returns the first evaluator's value as the score
    struct FirstEvaluatorScore;

    impl ScoringFunction for FirstEvaluatorScore {
        fn score(&self, evaluations: &crate::evaluators::EvaluationResult) -> Option<f64> {
            // Return the first Ok(Some(value)) we find, converted to f64
            for result in evaluations.values() {
                if let Ok(Some(value)) = result
                    && let Some(num) = value.as_f64()
                {
                    return Some(num);
                }
            }
            None
        }
    }

    #[test]
    fn test_compute_updates_empty_results() {
        let scoring_fn = FirstEvaluatorScore;
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [(
            "test_variant".to_string(),
            create_initial_cs("test_variant"),
        )]
        .into_iter()
        .collect();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> = [(
            "test_variant".to_string(),
            create_initial_cs("test_variant"),
        )]
        .into_iter()
        .collect();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [("evaluator1".to_string(), create_initial_cs("evaluator1"))]
                .into_iter()
                .collect();

        // Empty results should not change any confidence sequences
        let result = compute_updates(
            "test_variant",
            &[],
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

    #[test]
    fn test_compute_updates_only_cancelled() {
        let scoring_fn = FirstEvaluatorScore;
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [(
            "test_variant".to_string(),
            create_initial_cs("test_variant"),
        )]
        .into_iter()
        .collect();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> = [(
            "test_variant".to_string(),
            create_initial_cs("test_variant"),
        )]
        .into_iter()
        .collect();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
            [("evaluator1".to_string(), create_initial_cs("evaluator1"))]
                .into_iter()
                .collect();

        // Cancelled results should not affect any statistics
        let results = vec![
            crate::BatchItemResult::Cancelled,
            crate::BatchItemResult::Cancelled,
        ];

        let result = compute_updates(
            "test_variant",
            &results,
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

    #[test]
    fn test_compute_updates_variant_failures() {
        let scoring_fn = FirstEvaluatorScore;
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [(
            "test_variant".to_string(),
            create_initial_cs("test_variant"),
        )]
        .into_iter()
        .collect();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> = [(
            "test_variant".to_string(),
            create_initial_cs("test_variant"),
        )]
        .into_iter()
        .collect();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();

        // Only errors - this should update variant_failures only
        let results = vec![
            crate::BatchItemResult::Error(crate::DatapointVariantError {
                datapoint_id: Uuid::now_v7(),
                variant: None,
                message: "test error 1".to_string(),
            }),
            crate::BatchItemResult::Error(crate::DatapointVariantError {
                datapoint_id: Uuid::now_v7(),
                variant: None,
                message: "test error 2".to_string(),
            }),
        ];

        let result = compute_updates(
            "test_variant",
            &results,
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

    #[test]
    fn test_compute_updates_missing_variant_in_map() {
        let scoring_fn = FirstEvaluatorScore;
        // Don't include "test_variant" in any maps - should handle gracefully
        let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            HashMap::new();
        let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();
        let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();

        let results = vec![crate::BatchItemResult::Error(
            crate::DatapointVariantError {
                datapoint_id: Uuid::now_v7(),
                variant: None,
                message: "test error".to_string(),
            },
        )];

        // Should not panic, just skip updates for missing variants
        let result = compute_updates(
            "test_variant",
            &results,
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
}
