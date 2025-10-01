#![expect(dead_code)]

use thiserror::Error;
use typed_builder::TypedBuilder;

use super::estimate_optimal_probabilities::argmax;
use crate::db::FeedbackByVariant;

pub struct PairwiseGLRArgs {
    leader_pulls: u64,
    leader_mean: f64,
    leader_variance: f64,
    challenger_pulls: u64,
    challenger_mean: f64,
    challenger_variance: f64,
    epsilon: f64,
}

#[derive(Debug, Error)]
pub enum PairwiseGLRError {
    #[error("Error in compute_pairwise+_glr")]
    GLRError,
}

#[derive(TypedBuilder)]
pub struct CheckStoppingArgs {
    #[builder(setter(into))]
    pull_counts: Vec<u64>,
    #[builder(setter(into))]
    means: Vec<f64>,
    #[builder(setter(into))]
    variances: Vec<f64>,
    #[builder(setter(into))]
    min_pulls: u64,
    #[builder(default = 1e-6)]
    ridge_variance: f64,
    #[builder(default = 0.0)]
    epsilon: f64,
    #[builder(default = 0.05)]
    delta: f64,
}

#[derive(Debug, Error)]
pub enum CheckStoppingError {
    #[error("Error in computing pairwise GLR")]
    GLRError,
    #[error("Error in checking stopping condition")]
    StoppingError,
}

fn compute_pairwise_glr(args: PairwiseGLRArgs) -> Result<f64, PairwiseGLRError> {
    let PairwiseGLRArgs {
        leader_pulls,
        leader_mean,
        leader_variance,
        challenger_pulls,
        challenger_mean,
        challenger_variance,
        epsilon,
    } = args;
    // TODO: right way to validate inputs here?
    if leader_mean < challenger_mean {
        return Err(PairwiseGLRError::GLRError);
    }

    // No data means no evidence
    if (leader_pulls == 0) | (challenger_pulls == 0) {
        return Ok(0.0);
    }

    // Compute the empirical gap (with ε-adjustment)
    let empirical_gap: f64 = leader_mean - challenger_mean + epsilon;

    // Only consider positive gaps (evidence for H₁)
    if empirical_gap <= 0.0 {
        return Ok(0.0);
    }

    // Compute pooled standard error
    let pooled_variance: f64 = leader_variance / (leader_pulls.max(1) as f64)
        + challenger_variance / (challenger_pulls.max(1) as f64);

    if pooled_variance <= 0.0 {
        return Ok(0.0);
    }

    // GLR statistic: (gap)^2 / (2 * pooled_variance)
    let glr_statistic: f64 = (empirical_gap.powi(2)) / (2.0 * pooled_variance);

    Ok(glr_statistic.max(0.0))
}

pub struct StoppingResult {
    can_stop: bool,
    leader_arm: usize,
}

///     Decide whether to stop the experiment and which arm to recommend.
///     Uses parallel GLR testing with uniform challenger weights π_j = 1/(K-1),
///     and per-pair time t_{L,j} = n_L + n_j.
pub fn check_stopping(
    feedback: Vec<FeedbackByVariant>,
    min_pulls: u64,
    ridge_variance: Option<f64>,
    epsilon: Option<f64>,
    delta: Option<f64>,
) -> Result<StoppingResult, CheckStoppingError> {
    // TODO: how to validate inputs?
    let ridge_variance: f64 = ridge_variance.unwrap_or(1e-12);
    let epsilon: f64 = epsilon.unwrap_or(0.0);
    let delta: f64 = delta.unwrap_or(0.05);

    let pull_counts: Vec<u64> = feedback.iter().map(|x| x.count).collect();
    let means: Vec<f64> = feedback.iter().map(|x| x.mean as f64).collect();
    let variances: Vec<f64> = feedback
        .iter()
        .map(|x| (x.variance as f64).max(ridge_variance))
        .collect();

    // TODO: in case of tie, choose arm with largest value of variance / pull_count, for both stopping condition and return value. Log a warning.
    let num_arms: usize = pull_counts.len();
    let leader_arm: usize = argmax(&means).ok_or(CheckStoppingError::StoppingError)?;

    if pull_counts.iter().any(|&x| x < min_pulls) {
        return Ok(StoppingResult {
            can_stop: false,
            leader_arm,
        });
    }

    // Compute likelihood statistic for all non-leader arms
    let mut glr_vals: Vec<f64> = vec![];

    for challenger_arm in 0..num_arms {
        if challenger_arm == leader_arm {
            continue;
        }

        let leader_pulls: u64 = pull_counts[leader_arm];
        let leader_mean: f64 = means[leader_arm];
        let leader_variance: f64 = variances[leader_arm];
        let challenger_pulls: u64 = pull_counts[challenger_arm];
        let challenger_mean: f64 = means[challenger_arm];
        let challenger_variance: f64 = variances[challenger_arm];
        // Get GLR statistic for this pair
        let glr_stat = compute_pairwise_glr(PairwiseGLRArgs {
            leader_pulls,
            leader_mean,
            leader_variance,
            challenger_pulls,
            challenger_mean,
            challenger_variance,
            epsilon,
        });
        if let Ok(value) = glr_stat {
            glr_vals.push(value);
        } else {
            return Err(CheckStoppingError::GLRError);
        }
    }

    // Get stopping threshold
    let total_pulls: u64 = pull_counts.iter().sum();
    let threshold: f64 = ((1.0 + (total_pulls as f64).ln()) / delta).ln();

    // Stopping condition check
    let glr_min = glr_vals.iter().copied().reduce(f64::min);
    match glr_min {
        Some(min_val) => {
            if min_val > threshold {
                Ok(StoppingResult {
                    can_stop: true,
                    leader_arm,
                })
            } else {
                Ok(StoppingResult {
                    can_stop: false,
                    leader_arm,
                })
            }
        }
        None => Err(CheckStoppingError::StoppingError),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create FeedbackByVariant for tests
    fn make_feedback(
        pull_counts: Vec<u64>,
        means: Vec<f64>,
        variances: Vec<f64>,
    ) -> Vec<FeedbackByVariant> {
        pull_counts
            .into_iter()
            .zip(means)
            .zip(variances)
            .enumerate()
            .map(|(i, ((count, mean), variance))| FeedbackByVariant {
                variant_name: format!("variant_{i}"),
                mean: mean as f32,
                variance: variance as f32,
                count,
            })
            .collect()
    }

    // Tests for compute_pairwise_glr

    #[test]
    fn test_pairwise_glr_basic() {
        // Leader clearly better than challenger
        let glr = compute_pairwise_glr(PairwiseGLRArgs {
            leader_pulls: 100,
            leader_mean: 0.7,
            leader_variance: 0.1,
            challenger_pulls: 100,
            challenger_mean: 0.3,
            challenger_variance: 0.1,
            epsilon: 0.0,
        })
        .unwrap();

        // GLR should be positive and significant
        assert!(glr > 0.0);
        assert!(glr > 10.0, "Strong evidence should give large GLR");
    }

    #[test]
    fn test_pairwise_glr_with_epsilon() {
        // Test that epsilon adjustment affects the gap correctly
        let glr_no_epsilon = compute_pairwise_glr(PairwiseGLRArgs {
            leader_pulls: 50,
            leader_mean: 0.6,
            leader_variance: 0.2,
            challenger_pulls: 50,
            challenger_mean: 0.5,
            challenger_variance: 0.2,
            epsilon: 0.0,
        })
        .unwrap();

        let glr_with_epsilon = compute_pairwise_glr(PairwiseGLRArgs {
            leader_pulls: 50,
            leader_mean: 0.6,
            leader_variance: 0.2,
            challenger_pulls: 50,
            challenger_mean: 0.5,
            challenger_variance: 0.2,
            epsilon: 0.1,
        })
        .unwrap();

        // Adding epsilon should increase the gap and thus increase GLR
        assert!(glr_with_epsilon > glr_no_epsilon);
    }

    #[test]
    fn test_pairwise_glr_negative_gap() {
        // When leader has lower mean, empirical gap is negative
        let glr = compute_pairwise_glr(PairwiseGLRArgs {
            leader_pulls: 100,
            leader_mean: 0.3,
            leader_variance: 0.1,
            challenger_pulls: 100,
            challenger_mean: 0.7,
            challenger_variance: 0.1,
            epsilon: 0.0,
        });

        // Should error because leader_mean < challenger_mean
        assert!(glr.is_err());
    }

    #[test]
    fn test_pairwise_glr_zero_pulls() {
        // Zero pulls should return 0 GLR (no evidence)
        let glr = compute_pairwise_glr(PairwiseGLRArgs {
            leader_pulls: 0,
            leader_mean: 0.7,
            leader_variance: 0.1,
            challenger_pulls: 100,
            challenger_mean: 0.3,
            challenger_variance: 0.1,
            epsilon: 0.0,
        })
        .unwrap();

        assert_eq!(glr, 0.0);

        let glr2 = compute_pairwise_glr(PairwiseGLRArgs {
            leader_pulls: 100,
            leader_mean: 0.7,
            leader_variance: 0.1,
            challenger_pulls: 0,
            challenger_mean: 0.3,
            challenger_variance: 0.1,
            epsilon: 0.0,
        })
        .unwrap();

        assert_eq!(glr2, 0.0);
    }

    #[test]
    fn test_pairwise_glr_high_variance() {
        // Higher variance should reduce GLR (less confident)
        let glr_low_var = compute_pairwise_glr(PairwiseGLRArgs {
            leader_pulls: 100,
            leader_mean: 0.6,
            leader_variance: 0.1,
            challenger_pulls: 100,
            challenger_mean: 0.5,
            challenger_variance: 0.1,
            epsilon: 0.0,
        })
        .unwrap();

        let glr_high_var = compute_pairwise_glr(PairwiseGLRArgs {
            leader_pulls: 100,
            leader_mean: 0.6,
            leader_variance: 1.0,
            challenger_pulls: 100,
            challenger_mean: 0.5,
            challenger_variance: 1.0,
            epsilon: 0.0,
        })
        .unwrap();

        assert!(glr_low_var > glr_high_var);
    }

    #[test]
    fn test_pairwise_glr_sample_size_effect() {
        // More samples should increase GLR (more confident)
        let glr_small = compute_pairwise_glr(PairwiseGLRArgs {
            leader_pulls: 10,
            leader_mean: 0.6,
            leader_variance: 0.2,
            challenger_pulls: 10,
            challenger_mean: 0.5,
            challenger_variance: 0.2,
            epsilon: 0.0,
        })
        .unwrap();

        let glr_large = compute_pairwise_glr(PairwiseGLRArgs {
            leader_pulls: 1000,
            leader_mean: 0.6,
            leader_variance: 0.2,
            challenger_pulls: 1000,
            challenger_mean: 0.5,
            challenger_variance: 0.2,
            epsilon: 0.0,
        })
        .unwrap();

        assert!(glr_large > glr_small);
    }

    // Tests for check_stopping

    #[test]
    fn test_check_stopping_insufficient_pulls() {
        // Should return can_stop: false if any arm has fewer than min_pulls
        let feedback = make_feedback(vec![5, 15, 25], vec![0.3, 0.7, 0.5], vec![0.1, 0.1, 0.1]);
        let result = check_stopping(feedback, 10, None, Some(0.0), None).unwrap();
        assert!(!result.can_stop);
    }

    #[test]
    fn test_check_stopping_clear_winner_should_stop() {
        // Clear winner with many samples should trigger stopping
        let feedback = make_feedback(
            vec![1000, 1000, 1000],
            vec![0.3, 0.9, 0.5],
            vec![0.1, 0.1, 0.1],
        );
        let result = check_stopping(feedback, 10, None, Some(0.0), Some(0.05)).unwrap();
        assert!(result.can_stop, "Should stop with clear winner");
        assert_eq!(
            result.leader_arm, 1,
            "Should recommend arm with highest mean"
        );
    }

    #[test]
    fn test_check_stopping_close_competition_should_not_stop() {
        // Close competition with small sample size should not stop
        let feedback = make_feedback(
            vec![50, 50, 50],
            vec![0.50, 0.52, 0.51],
            vec![0.2, 0.2, 0.2],
        );
        let result = check_stopping(feedback, 10, None, Some(0.0), Some(0.05)).unwrap();
        assert!(!result.can_stop, "Should not stop with close competition");
    }

    #[test]
    fn test_check_stopping_returns_empirical_leader() {
        let feedback = make_feedback(vec![50, 50, 50], vec![0.3, 0.7, 0.5], vec![0.1, 0.1, 0.1]);
        let result = check_stopping(feedback, 10, None, Some(0.0), None).unwrap();
        assert_eq!(result.leader_arm, 1, "Should return arm with highest mean");
    }

    #[test]
    fn test_check_stopping_with_epsilon() {
        // Epsilon-optimality: arm 0 is within epsilon of the best
        let feedback_no_epsilon = make_feedback(vec![500, 500], vec![0.68, 0.70], vec![0.1, 0.1]);
        let result_no_eps =
            check_stopping(feedback_no_epsilon, 10, None, Some(0.0), Some(0.05)).unwrap();

        let feedback_with_epsilon = make_feedback(vec![500, 500], vec![0.68, 0.70], vec![0.1, 0.1]);
        let result_with_eps =
            check_stopping(feedback_with_epsilon, 10, None, Some(0.05), Some(0.05)).unwrap();

        // With epsilon, should be more likely to stop (easier to satisfy)
        // Note: exact behavior depends on sample sizes and variances
        // This test mainly verifies epsilon is used without error
        assert!(
            result_with_eps.can_stop || !result_no_eps.can_stop,
            "Epsilon should make stopping easier or equivalent"
        );
    }

    #[test]
    fn test_check_stopping_ridge_variance_applied() {
        // Very small variances should be bounded by ridge_variance
        let feedback = make_feedback(vec![100, 100], vec![0.5, 0.7], vec![1e-10, 1e-10]);
        // Should not panic due to division by near-zero variance
        let result = check_stopping(feedback, 10, Some(0.01), Some(0.0), None);
        assert!(result.is_ok(), "Ridge variance should prevent degeneracy");
    }

    #[test]
    fn test_check_stopping_delta_affects_threshold() {
        // Larger delta should make stopping easier (lower threshold)
        // Choose parameters where the evidence is borderline
        let feedback_strict = make_feedback(
            vec![150, 150, 150],
            vec![0.45, 0.65, 0.50],
            vec![0.15, 0.15, 0.15],
        );
        let result_strict =
            check_stopping(feedback_strict, 10, None, Some(0.0), Some(0.01)).unwrap();

        let feedback_lenient = make_feedback(
            vec![150, 150, 150],
            vec![0.45, 0.65, 0.50],
            vec![0.15, 0.15, 0.15],
        );
        let result_lenient =
            check_stopping(feedback_lenient, 10, None, Some(0.0), Some(0.2)).unwrap();

        // Strict delta should not stop, but lenient delta should
        assert!(
            !result_strict.can_stop,
            "Strict delta should not stop with borderline evidence"
        );
        assert!(
            result_lenient.can_stop,
            "Lenient delta should stop with same evidence"
        );
    }

    #[test]
    fn test_check_stopping_two_arms() {
        // Simplest case: two arms
        let feedback = make_feedback(vec![100, 100], vec![0.3, 0.8], vec![0.1, 0.1]);
        let result = check_stopping(feedback, 10, None, Some(0.0), Some(0.05)).unwrap();
        assert_eq!(result.leader_arm, 1);
        // With clear difference and enough samples, should likely stop
        assert!(result.can_stop);
    }

    #[test]
    fn test_check_stopping_many_arms() {
        // Test with 5 arms
        let feedback = make_feedback(
            vec![200, 200, 200, 200, 200],
            vec![0.3, 0.9, 0.5, 0.4, 0.6],
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
        );
        let result = check_stopping(feedback, 50, None, Some(0.0), Some(0.05)).unwrap();
        assert_eq!(
            result.leader_arm, 1,
            "Should recommend arm with highest mean"
        );
        // With clear winner, should stop
        assert!(result.can_stop);
    }

    #[test]
    fn test_check_stopping_increasing_samples_increases_stopping() {
        // With fixed means/variances, more samples should eventually lead to stopping
        // Use moderate difference and moderate variance to create borderline case
        let feedback_small = make_feedback(vec![80, 80], vec![0.55, 0.70], vec![0.25, 0.25]);
        let result_small = check_stopping(feedback_small, 10, None, Some(0.0), Some(0.05)).unwrap();

        let feedback_large = make_feedback(vec![1000, 1000], vec![0.55, 0.70], vec![0.25, 0.25]);
        let result_large = check_stopping(feedback_large, 10, None, Some(0.0), Some(0.05)).unwrap();

        // Small sample should not stop, but large sample should
        assert!(
            !result_small.can_stop,
            "Should not stop with small sample size"
        );
        assert!(
            result_large.can_stop,
            "Should stop with large sample and clear difference"
        );
    }
}
