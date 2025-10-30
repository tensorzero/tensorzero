use core::f64;
use std::cmp::Ordering;
use thiserror::Error;

use crate::{config::MetricConfigOptimize, db::feedback::FeedbackByVariant};

/// Find all indices with the maximum value in `values`.
///
/// Returns a vector of all indices where the value is within a small tolerance
/// (1e-10) of the maximum value. This allows for handling floating-point ties.
///
/// # Returns
///
/// A vector of indices with the maximum value. Returns an empty vector if the
/// input slice is empty.
///
/// # Examples
///
/// ```ignore
/// let values = vec![0.3, 0.7, 0.5];
/// assert_eq!(argmax_with_ties(&values), vec![1]);
///
/// let values = vec![0.7, 0.5, 0.7];
/// assert_eq!(argmax_with_ties(&values), vec![0, 2]);
/// ```
fn argmax_with_ties(values: &[f64]) -> Vec<usize> {
    let max_val: f64 = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Find all indices with the maximum value, up to tolerance
    let tolerance: f64 = 1e-10;
    let max_indices: Vec<usize> = values
        .iter()
        .enumerate()
        .filter(|(_, &v)| (v - max_val).abs() < tolerance)
        .map(|(i, _)| i)
        .collect();

    max_indices
}

/// Choose the leader arm based on means, breaking ties by smallest variance per pull.
///
/// This function identifies the arm(s) with the highest mean, then breaks ties by
/// selecting the arm with the smallest `variance / pull_count` ratio. If there is
/// still a tie, it selects the arm with the smallest index, for stability.
///
/// When ties in means are detected, a warning is logged to help identify situations
/// where arms are performing similarly.
///
/// # Arguments
///
/// * `means` - The mean rewards for each arm
/// * `variances` - The variances for each arm
/// * `pull_counts` - The number of times each arm has been pulled
///
/// # Returns
///
/// The index of the chosen leader arm, or `None` if the input slices are empty.
///
/// # Panics
///
/// This function assumes all three slices have the same length. If they don't,
/// it may panic or return incorrect results.
///
/// # Examples
///
/// ```ignore
/// let means = vec![0.7, 0.7, 0.5];
/// let variances = vec![0.3, 0.1, 0.2];
/// let pull_counts = vec![100, 100, 100];
/// // Returns 1 (highest mean, smallest variance per pull among tied)
/// assert_eq!(choose_leader(&means, &variances, &pull_counts), Some(1));
/// ```
pub(super) fn choose_leader(
    means: &[f64],
    variances: &[f64],
    pull_counts: &[u64],
) -> Option<usize> {
    let leaders = argmax_with_ties(means);
    let leader = if leaders.len() == 1 {
        leaders[0]
    } else {
        // Log warning about tie in means
        tracing::warn!(
            "Tie in leader selection: {} arms have approximately equal means. Breaking tie by variance/pull_count ratio.",
            leaders.len()
        );

        // tie-break by smallest variance_per_pull; then by smallest index
        let variance_per_pull: Vec<f64> = variances
            .iter()
            .zip(pull_counts.iter())
            .map(|(&var, &count)| var / count.max(1) as f64)
            .collect();

        *leaders.iter().min_by(|&&i, &&j| {
            variance_per_pull[i]
                .partial_cmp(&variance_per_pull[j])
                .unwrap_or(Ordering::Equal)
                .then_with(|| i.cmp(&j))
        })?
    };

    Some(leader)
}

/// Arguments for computing pairwise generalized likelihood ratio (GLR) statistics.
///
/// Used to test whether a leader arm is significantly better than a challenger arm
/// in the context of epsilon-best arm identification.
pub struct PairwiseGLRArgs {
    /// Number of pulls for leader arm (arm with highest empirical mean reward)
    leader_pulls: u64,
    /// Empirical mean reward for leader arm
    leader_mean: f64,
    /// Empirical variance of leader arm
    leader_variance: f64,
    /// Number of pulls for challenger arm
    challenger_pulls: u64,
    /// Empirical mean reward for challenger arm
    challenger_mean: f64,
    /// Empirical variance of challenger arm
    challenger_variance: f64,
    /// Sub-optimality tolerance
    epsilon: f64,
}

/// Errors that can occur when computing pairwise GLR statistics.
/// TODO: Make errors more informative
#[derive(Debug, Error)]
pub enum PairwiseGLRError {
    #[error("Error in compute_pairwise_glr")]
    GLRError,
}

/// Arguments for checking the stopping condition in epsilon-best arm identification
///
/// This struct encapsulates all parameters needed to determine whether an experiment
/// should stop and which arm should be recommended as the winner.
pub(super) struct CheckStoppingArgs<'a> {
    /// Reward observations and pull counts for each arm
    pub feedback: &'a Vec<FeedbackByVariant>,
    /// Required minimum number of pulls per arm before stopping can be considered
    pub min_pulls: u64,
    /// Value used to lower bound empirical variance, for stability
    pub variance_floor: Option<f64>,
    /// Sub-optimality tolerance
    pub epsilon: Option<f64>,
    /// Type 1 error tolerance, aka 1 minus the confidence level
    pub delta: Option<f64>,
    /// Optimization direction (Min or Max)
    pub metric_optimize: MetricConfigOptimize,
}

/// Errors that can occur when checking stopping conditions.
/// TODO: Make errors more informative
#[derive(Debug, Error)]
pub enum CheckStoppingError {
    #[error("Error in computing pairwise GLR")]
    GLRError,
    #[error("Error in checking stopping condition")]
    StoppingError,
    #[error("Missing variance for variant '{variant_name}' - variance must be non-null")]
    MissingVariance { variant_name: String },
}

/// Compute the pairwise generalized likelihood ratio (GLR) statistic.
///
/// This function computes the GLR statistic for testing whether an empirical leader arm
/// is epsilon-close to the best arm, in the sense that either (1) it is the best arm, or
/// (2) its mean is within epsilon of the best arm.
///
/// The GLR statistic is based on the empirical reward gap adjusted by epsilon:
/// `GLR = (leader_mean - challenger_mean + epsilon)^2 / (2 * pooled_variance)`
///
/// where `pooled_variance` is the sum of the per-sample variances.
///
/// # Arguments
///
/// * `args` - The pairwise GLR arguments including pull counts, means, variances,
///   and epsilon for both the leader and challenger arms
///
/// # Returns
///
/// The GLR statistic (non-negative), or an error if:
/// - The leader mean is less than the challenger mean
///
/// Returns 0.0 if:
/// - Either arm has 0 pulls (no evidence)
/// - The empirical gap is non-positive
/// - The pooled variance is non-positive
///
/// # Errors
///
/// Returns `PairwiseGLRError::GLRError` if the leader mean is less than the
/// challenger mean, as this violates the assumption that we're testing whether
/// the leader is better.
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

/// The result of checking whether to stop a bandit experiment.
#[derive(Debug, PartialEq)]
pub enum StoppingResult {
    /// The experiment should continue (not enough evidence to stop)
    NotStopped,
    /// The experiment should stop and recommend the winner variant
    Winner(String),
}

/// Decide whether to stop the experiment and which arm to recommend.
///
/// This function implements a sequential testing procedure for epsilon-best arm
/// identification using parallel generalized likelihood ratio (GLR) tests. It
/// determines whether there is sufficient statistical evidence to stop the
/// experiment and recommend a winner.
///
/// The algorithm:
/// 1. Identifies the empirical leader (arm with highest mean)
/// 2. Computes GLR statistics comparing the leader to all other arms
/// 3. Checks if the minimum GLR exceeds a threshold based on the confidence level (1 - delta)
/// 4. If all pairwise tests pass, declares the leader as the winner
///
/// # Arguments
///
/// * `args` - The stopping check arguments, including:
///   - `feedback`: Performance data for each variant (must be non-empty)
///   - `min_pulls`: Minimum number of pulls required per arm before stopping (must be > 0)
///   - `variance_floor`: Lower bound for empirical variances (must be > 0, default: 1e-12)
///   - `epsilon`: Epsilon-optimality tolerance (must be ≥ 0, default: 0.0)
///   - `delta`: Error tolerance for stopping (must be in (0, 1), default: 0.05)
///
/// # Returns
///
/// * `Ok(StoppingResult::NotStopped)` - Continue the experiment
/// * `Ok(StoppingResult::Winner(name))` - Stop and recommend variant `name`
/// * `Err(CheckStoppingError)` - An error occurred during computation
///
/// # Errors
///
/// Returns an error if:
/// - No valid leader can be identified (empty feedback)
/// - GLR computation fails for any pairwise comparison
///
/// # Preconditions
///
/// The function expects (but does not currently validate):
/// - `feedback` is non-empty
/// - `min_pulls > 0`
/// - `variance_floor > 0` (if provided)
/// - `epsilon ≥ 0` (if provided)
/// - `delta ∈ (0, 1)` (if provided)
///
/// # Notes
///
/// - Ties in means are broken by selecting the arm with smallest variance/pull_count
/// - All arms must have at least `min_pulls` samples before stopping is possible
/// - Empirical variances are lower bounded at `variance_floor`
pub fn check_stopping(args: CheckStoppingArgs<'_>) -> Result<StoppingResult, CheckStoppingError> {
    let CheckStoppingArgs {
        feedback,
        min_pulls,
        variance_floor,
        epsilon,
        delta,
        metric_optimize,
    } = args;
    let variance_floor: f64 = variance_floor.unwrap_or(1e-12);
    let epsilon: f64 = epsilon.unwrap_or(0.0);
    let delta: f64 = delta.unwrap_or(0.05);
    let pull_counts: Vec<u64> = feedback.iter().map(|x| x.count).collect();

    // Negate the means if we're minimizing, so that we can always use argmax
    let means: Vec<f64> = feedback
        .iter()
        .map(|x| {
            let mean = x.mean as f64;
            match metric_optimize {
                MetricConfigOptimize::Min => -mean,
                MetricConfigOptimize::Max => mean,
            }
        })
        .collect();

    // Can't stop the experiment if any arms haven't been pulled up to the min pull count
    if pull_counts.iter().any(|&x| x < min_pulls) {
        return Ok(StoppingResult::NotStopped);
    }

    // Validate and extract variances - all must be present
    let variances: Result<Vec<f64>, CheckStoppingError> = feedback
        .iter()
        .map(|x| {
            x.variance
                .map(|v| (v as f64).max(variance_floor))
                .ok_or_else(|| CheckStoppingError::MissingVariance {
                    variant_name: x.variant_name.clone(),
                })
        })
        .collect();
    let variances = variances?;
    let num_arms: usize = pull_counts.len();

    // Set leader arm
    let leader_arm: usize =
        choose_leader(&means, &variances, &pull_counts).ok_or(CheckStoppingError::StoppingError)?;

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
                Ok(StoppingResult::Winner(
                    feedback[leader_arm].variant_name.clone(),
                ))
            } else {
                Ok(StoppingResult::NotStopped)
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
                variance: Some(variance as f32),
                count,
            })
            .collect()
    }

    // Tests for argmax_with_ties

    #[test]
    fn test_argmax_with_ties_no_tie() {
        let values = vec![0.3, 0.7, 0.5];
        assert_eq!(argmax_with_ties(&values), vec![1]);
    }

    #[test]
    fn test_argmax_with_ties_two_way_tie() {
        let values = vec![0.7, 0.5, 0.7];
        assert_eq!(argmax_with_ties(&values), vec![0, 2]);
    }

    #[test]
    fn test_argmax_with_ties_all_tied() {
        let values = vec![0.5, 0.5, 0.5];
        assert_eq!(argmax_with_ties(&values), vec![0, 1, 2]);
    }

    #[test]
    fn test_argmax_with_ties_empty() {
        let values: Vec<f64> = vec![];
        assert_eq!(argmax_with_ties(&values), vec![] as Vec<usize>);
    }

    // Tests for choose_leader

    #[test]
    fn test_choose_leader_no_tie() {
        let means = vec![0.3, 0.7, 0.5];
        let variances = vec![0.1, 0.1, 0.1];
        let pull_counts = vec![100, 100, 100];
        assert_eq!(choose_leader(&means, &variances, &pull_counts), Some(1));
    }

    #[test]
    fn test_choose_leader_tie_breaks_by_smallest_variance_per_pull() {
        let means = vec![0.7, 0.7, 0.5];
        let variances = vec![0.3, 0.1, 0.2]; // Same pull counts, so variance order matters
        let pull_counts = vec![100, 100, 100];
        // Indices 0 and 1 are tied at 0.7
        // variance_per_pull: [0.003, 0.001, 0.002]
        // Should pick 1 (smallest variance_per_pull among tied)
        assert_eq!(choose_leader(&means, &variances, &pull_counts), Some(1));
    }

    #[test]
    fn test_choose_leader_tie_breaks_by_smallest_index_when_variance_tied() {
        let means = vec![0.7, 0.7, 0.5];
        let variances = vec![0.1, 0.1, 0.2]; // Same variance_per_pull for first two
        let pull_counts = vec![100, 100, 100];
        // Indices 0 and 1 are tied at 0.7 with same variance_per_pull (0.001)
        // Should pick 0 (smaller index)
        assert_eq!(choose_leader(&means, &variances, &pull_counts), Some(0));
    }

    #[test]
    fn test_choose_leader_all_tied() {
        let means = vec![0.5, 0.5, 0.5];
        let variances = vec![0.3, 0.1, 0.2];
        let pull_counts = vec![100, 100, 100];
        // All tied in means, should pick 1 (smallest variance_per_pull: 0.001)
        assert_eq!(choose_leader(&means, &variances, &pull_counts), Some(1));
    }

    #[test]
    fn test_choose_leader_empty() {
        let means: Vec<f64> = vec![];
        let variances: Vec<f64> = vec![];
        let pull_counts: Vec<u64> = vec![];
        assert_eq!(choose_leader(&means, &variances, &pull_counts), None);
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
        // Should return NotStopped if any arm has fewer than min_pulls
        let feedback = make_feedback(vec![5, 15, 25], vec![0.3, 0.7, 0.5], vec![0.1, 0.1, 0.1]);
        let result = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: None,
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();
        assert_eq!(result, StoppingResult::NotStopped);
    }

    #[test]
    fn test_check_stopping_clear_winner_should_stop() {
        // Clear winner with many samples should trigger stopping
        let feedback = make_feedback(
            vec![1000, 1000, 1000],
            vec![0.3, 0.9, 0.5],
            vec![0.1, 0.1, 0.1],
        );
        let result = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();
        assert_eq!(
            result,
            StoppingResult::Winner("variant_1".to_string()),
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
        let result = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();
        assert_eq!(
            result,
            StoppingResult::NotStopped,
            "Should not stop with close competition"
        );
    }

    #[test]
    fn test_check_stopping_returns_empirical_leader() {
        let feedback = make_feedback(vec![50, 50, 50], vec![0.3, 0.7, 0.5], vec![0.1, 0.1, 0.1]);
        let result = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: None,
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();
        if let StoppingResult::Winner(name) = result {
            assert_eq!(name, "variant_1", "Should return arm with highest mean");
        }
        // Note: This test doesn't assert whether it stopped or not, just that IF it stops,
        // it returns the correct winner
    }

    #[test]
    fn test_check_stopping_with_epsilon() {
        // Epsilon-optimality: arm 0 is within epsilon of the best
        let feedback_no_epsilon = make_feedback(vec![500, 500], vec![0.68, 0.70], vec![0.1, 0.1]);
        let result_no_eps = check_stopping(CheckStoppingArgs {
            feedback: &feedback_no_epsilon,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        let feedback_with_epsilon = make_feedback(vec![500, 500], vec![0.68, 0.70], vec![0.1, 0.1]);
        let result_with_eps = check_stopping(CheckStoppingArgs {
            feedback: &feedback_with_epsilon,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.05),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // With epsilon, should be more likely to stop (easier to satisfy)
        // Note: exact behavior depends on sample sizes and variances
        // This test mainly verifies epsilon is used without error
        assert!(
            matches!(result_with_eps, StoppingResult::Winner(_))
                || matches!(result_no_eps, StoppingResult::NotStopped),
            "Epsilon should make stopping easier or equivalent"
        );
    }

    #[test]
    fn test_check_stopping_variance_floor_applied() {
        // Very small variances should be bounded by variance_floor
        let feedback = make_feedback(vec![100, 100], vec![0.5, 0.7], vec![1e-10, 1e-10]);
        // Should not panic due to division by near-zero variance
        let result = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 10,
            variance_floor: Some(0.01),
            epsilon: Some(0.0),
            delta: None,
            metric_optimize: MetricConfigOptimize::Max,
        });
        assert!(result.is_ok(), "Variance floor should prevent degeneracy");
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
        let result_strict = check_stopping(CheckStoppingArgs {
            feedback: &feedback_strict,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.01),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        let feedback_lenient = make_feedback(
            vec![150, 150, 150],
            vec![0.45, 0.65, 0.50],
            vec![0.15, 0.15, 0.15],
        );
        let result_lenient = check_stopping(CheckStoppingArgs {
            feedback: &feedback_lenient,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.2),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Strict delta should not stop, but lenient delta should
        assert!(
            matches!(result_strict, StoppingResult::NotStopped),
            "Strict delta should not stop with borderline evidence"
        );
        assert!(
            matches!(result_lenient, StoppingResult::Winner(_)),
            "Lenient delta should stop with same evidence"
        );
    }

    #[test]
    fn test_check_stopping_two_arms() {
        // Simplest case: two arms
        let feedback = make_feedback(vec![100, 100], vec![0.3, 0.8], vec![0.1, 0.1]);
        let result = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();
        // With clear difference and enough samples, should stop
        if let StoppingResult::Winner(name) = result {
            assert_eq!(name, "variant_1");
        } else {
            panic!("Expected winner with clear difference and enough samples");
        }
    }

    #[test]
    fn test_check_stopping_many_arms() {
        // Test with 5 arms
        let feedback = make_feedback(
            vec![200, 200, 200, 200, 200],
            vec![0.3, 0.9, 0.5, 0.4, 0.6],
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
        );
        let result = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 50,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();
        // With clear winner, should stop
        if let StoppingResult::Winner(name) = result {
            assert_eq!(name, "variant_1", "Should recommend arm with highest mean");
        } else {
            panic!("Expected winner with clear difference and enough samples");
        }
    }

    #[test]
    fn test_check_stopping_tie_breaking() {
        // Test that ties are broken using variance/pull_count
        // Create three arms with identical means but different variances
        let feedback = make_feedback(
            vec![100, 100, 100],
            vec![0.7, 0.7, 0.7], // All tied
            vec![0.1, 0.3, 0.2], // Different variances
        );
        let result = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Should select variant_1 which has highest variance/pull_count (0.3/100 = 0.003)
        if let StoppingResult::Winner(name) = result {
            assert_eq!(
                name, "variant_1",
                "Should break tie by selecting arm with highest variance/pull_count"
            );
        }
        // Note: May not stop depending on the threshold, but if it does, should be variant_1
    }

    #[test]
    fn test_check_stopping_increasing_samples_increases_stopping() {
        // With fixed means/variances, more samples should eventually lead to stopping
        // Use moderate difference and moderate variance to create borderline case
        let feedback_small = make_feedback(vec![80, 80], vec![0.55, 0.70], vec![0.25, 0.25]);
        let result_small = check_stopping(CheckStoppingArgs {
            feedback: &feedback_small,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        let feedback_large = make_feedback(vec![1000, 1000], vec![0.55, 0.70], vec![0.25, 0.25]);
        let result_large = check_stopping(CheckStoppingArgs {
            feedback: &feedback_large,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Small sample should not stop, but large sample should
        assert!(
            matches!(result_small, StoppingResult::NotStopped),
            "Should not stop with small sample size"
        );
        assert!(
            matches!(result_large, StoppingResult::Winner(_)),
            "Should stop with large sample and clear difference"
        );
    }

    // ============================================================================
    // Tests for optimize=min
    // ============================================================================

    #[test]
    fn test_check_stopping_two_arms_optimize_min() {
        // Test that with optimize=min, the arm with the lowest mean wins
        let feedback = make_feedback(vec![100, 100], vec![0.3, 0.8], vec![0.1, 0.1]);
        let result = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();
        if let StoppingResult::Winner(name) = result {
            assert_eq!(
                name, "variant_0",
                "Should recommend arm with lowest mean when optimize=min"
            );
        } else {
            panic!("Expected variant_0 to be labeled the winner");
        }
    }

    #[test]
    fn test_check_stopping_many_arms_optimize_min() {
        // Test with 5 arms, optimize=min selects the one with lowest mean
        let feedback = make_feedback(
            vec![200, 200, 200, 200, 200],
            vec![0.9, 0.1, 0.5, 0.4, 0.6],
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
        );
        let result = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 50,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();
        // variant_1 has the lowest mean, so it should win
        if let StoppingResult::Winner(name) = result {
            assert_eq!(
                name, "variant_1",
                "Should recommend arm with lowest mean when optimize=min"
            );
        } else {
            panic!("Expected variant_1 to be labeled the winne");
        }
    }

    #[test]
    fn test_check_stopping_tie_breaking_optimize_min() {
        // Test that ties are broken correctly with optimize=min
        // Create three arms with identical means but different variances
        let feedback = make_feedback(
            vec![100, 100, 100],
            vec![0.5, 0.5, 0.5], // All tied at 0.5
            vec![0.1, 0.3, 0.2], // Different variances
        );
        let result = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 10,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        // Even with optimize=min, tie-breaking should work the same way
        // Should select variant_1 which has highest variance/pull_count (0.3/100 = 0.003)
        if let StoppingResult::Winner(name) = result {
            assert_eq!(
                name, "variant_1",
                "Should break tie by selecting arm with highest variance/pull_count, even with optimize=min"
            );
        }
    }

    #[test]
    fn test_check_stopping_min_vs_max_different_winners() {
        // Verify that the same data produces different winners for min vs max
        let feedback = make_feedback(
            vec![200, 200, 200],
            vec![0.3, 0.7, 0.5], // variant_0 lowest, variant_1 highest
            vec![0.1, 0.1, 0.1],
        );

        let result_max = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 50,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        let result_min = check_stopping(CheckStoppingArgs {
            feedback: &feedback,
            min_pulls: 50,
            variance_floor: None,
            epsilon: Some(0.0),
            delta: Some(0.05),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        // With optimize=max, variant_1 (highest mean 0.7) should win
        if let StoppingResult::Winner(name_max) = result_max {
            assert_eq!(name_max, "variant_1", "Max should pick highest mean");
        } else {
            panic!("Expected winner for optimize=max");
        }

        // With optimize=min, variant_0 (lowest mean 0.3) should win
        if let StoppingResult::Winner(name_min) = result_min {
            assert_eq!(name_min, "variant_0", "Min should pick lowest mean");
        } else {
            panic!("Expected winner for optimize=min");
        }
    }
}
