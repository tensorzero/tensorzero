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

use crate::EvaluationVariant;
use crate::betting_confidence_sequences::MeanBettingConfidenceSequence;

#[expect(dead_code)]
const EVALUATOR_FAILURE_THRESHOLD: f32 = 0.05;
#[expect(dead_code)]
const VARIANT_FAILURE_THRESHOLD: f32 = 0.05;

// Enum for variant status during an evals run.
// Will to used in run() function, to be implemented.
#[expect(dead_code)]
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
// the highest ranked condition takes precedence. The order of the last three is fairly arbitrary.
pub enum GlobalStoppingReason {
    // If top-k found, return the k that caused stopping (largest k satisfied in k_max..k_min)
    TopKFound(u32),
    // Datapoint limit hit
    MaxDatapointsReached,
    // If evaluator(s) failed, return name(s) of failed evaluator(s).
    // An evaluator fails if the lower bound of the confidence sequence for its
    // failure rate exceeds EVALUATOR_FAILURE_THRESHOLD.
    EvaluatorsFailed(Vec<String>),
    // If too many variants failed (VariantStatus::Failed), return name(s) of failed variant(s).
    // If more than num_variants - k_min variants have failed, we can no longer identify the top-k
    // variants for any k in k_min..k_max.
    TooManyVariantsFailed(Vec<String>),
}

// Arguments to main run() function, to be implemented
#[expect(dead_code)]
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

// Struct for the output of the run() function, to be implemented
pub struct AdaptiveEvalStoppingResults {
    pub variant_performance: Vec<MeanBettingConfidenceSequence>,
    pub variant_failure_rates: Vec<MeanBettingConfidenceSequence>,
    pub evaluator_failure_rates: Vec<MeanBettingConfidenceSequence>,
    pub stopping_reason: GlobalStoppingReason,
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
) -> TopKStoppingResult {
    let epsilon = epsilon.unwrap_or(0.0);
    let num_variants = variant_performance.len();

    if num_variants == 0 || k_min == 0 || k_max < k_min {
        return TopKStoppingResult {
            stopped: false,
            k: None,
            top_variants: vec![],
        };
    }

    // Collect upper bounds into a sorted vec for binary search
    let mut upper_bounds: Vec<f64> = variant_performance.values().map(|cs| cs.cs_upper).collect();
    upper_bounds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // For each variant, count how many upper bounds its lower bound exceeds
    // This tells us how many variants it "confidently beats"
    let mut variants_with_n_beaten: Vec<(&String, usize)> = variant_performance
        .iter()
        .map(|(name, cs)| {
            // Binary search to find how many upper bounds are strictly less than this lower bound
            let num_beaten = upper_bounds.partition_point(|&ub| ub - epsilon < cs.cs_lower);
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

            return TopKStoppingResult {
                stopped: true,
                k: Some(k),
                top_variants,
            };
        }
    }

    TopKStoppingResult {
        stopped: false,
        k: None,
        top_variants: vec![],
    }
}

/// Convenience wrapper to check if a specific k can be identified.
///
/// This is equivalent to calling `check_topk_stopping` with k_min = k_max = k.
pub fn check_topk(
    variant_performance: &HashMap<String, MeanBettingConfidenceSequence>,
    k: u32,
    epsilon: Option<f64>,
) -> TopKStoppingResult {
    check_topk_stopping(variant_performance, k, k, epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::betting_confidence_sequences::WealthProcesses;

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
                    m_values: None,
                    resolution: Some(101),
                    wealth_upper: vec![1.0; 101],
                    wealth_lower: vec![1.0; 101],
                },
            },
        )
    }

    #[test]
    fn test_check_topk_stopping_empty() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();
        let result = check_topk_stopping(&variant_performance, 1, 1, None);
        assert!(!result.stopped);
        assert!(result.k.is_none());
        assert!(result.top_variants.is_empty());
    }

    #[test]
    fn test_check_topk_stopping_invalid_k() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs("a", 0.5, 0.7)].into_iter().collect();

        // k_min = 0 is invalid
        let result = check_topk_stopping(&variant_performance, 0, 1, None);
        assert!(!result.stopped);

        // k_max < k_min is invalid
        let result = check_topk_stopping(&variant_performance, 2, 1, None);
        assert!(!result.stopped);
    }

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

        let result = check_topk_stopping(&variant_performance, 1, 1, None);
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert_eq!(result.top_variants.len(), 1);
        assert!(result.top_variants.contains(&"a".to_string()));
    }

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

        let result = check_topk_stopping(&variant_performance, 2, 2, None);
        assert!(result.stopped);
        assert_eq!(result.k, Some(2));
        assert_eq!(result.top_variants.len(), 2);
        assert!(result.top_variants.contains(&"a".to_string()));
        assert!(result.top_variants.contains(&"b".to_string()));
    }

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

        let result = check_topk_stopping(&variant_performance, 1, 1, None);
        assert!(!result.stopped);
        assert!(result.k.is_none());
    }

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
        let result = check_topk_stopping(&variant_performance, 1, 2, None);
        assert!(result.stopped);
        assert_eq!(result.k, Some(2));
        assert_eq!(result.top_variants.len(), 2);

        // When k_min=1, k_max=1, should return k=1
        let result = check_topk_stopping(&variant_performance, 1, 1, None);
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert_eq!(result.top_variants.len(), 1);
        assert!(result.top_variants.contains(&"a".to_string()));
    }

    #[test]
    fn test_check_topk_stopping_k_larger_than_variants() {
        // Only 2 variants but asking for top-3
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs("a", 0.7, 0.9), mock_cs("b", 0.3, 0.5)]
                .into_iter()
                .collect();

        let result = check_topk_stopping(&variant_performance, 3, 3, None);
        assert!(!result.stopped);
    }

    #[test]
    fn test_check_topk_single_variant() {
        // Single variant - should be identified as top-1
        // It needs to beat >= (1 - 1) = 0 variants, which it does trivially
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
            [mock_cs("a", 0.5, 0.7)].into_iter().collect();

        let result = check_topk_stopping(&variant_performance, 1, 1, None);
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
        assert_eq!(result.top_variants, vec!["a".to_string()]);
    }

    #[test]
    fn test_check_topk_wrapper() {
        let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
            mock_cs("a", 0.7, 0.9),
            mock_cs("b", 0.3, 0.5),
            mock_cs("c", 0.2, 0.4),
        ]
        .into_iter()
        .collect();

        let result = check_topk(&variant_performance, 1, None);
        assert!(result.stopped);
        assert_eq!(result.k, Some(1));
    }
}
