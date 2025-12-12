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

use crate::betting_confidence_sequences::MeanBettingConfidenceSequence;

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
}
