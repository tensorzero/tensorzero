#![allow(non_snake_case)]
use clarabel::algebra::CscMatrix;
use clarabel::solver::SolverStatus;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SecondOrderConeT, SupportedConeT,
    ZeroConeT,
};
use std::collections::HashMap;
use thiserror::Error;

use crate::config::MetricConfigOptimize;
use crate::db::feedback::FeedbackByVariant;
use crate::experimentation::track_and_stop::check_stopping::choose_leader;

/// Arguments for computing optimal sampling probabilities.
///
/// This struct encapsulates the parameters needed for the ε-aware optimal allocation
/// algorithm, which computes sampling probabilities to efficiently identify the best arm
/// while respecting an ε-tolerance for sub-optimality.
pub struct EstimateOptimalProbabilitiesArgs {
    /// Reward observations and pull counts for each arm
    pub feedback: Vec<FeedbackByVariant>,
    /// Sub-optimality tolerance (ε ≥ 0). Arms within ε of the best arm's mean are considered
    /// "good enough". Larger values lead to faster stopping.
    /// Default: 0.0
    pub epsilon: Option<f64>,
    /// Value used to lower bound empirical variance, for stability. Prevents numerical issues
    /// when observed variances are very small. Default: 1e-12
    pub variance_floor: Option<f64>,
    /// Lower bound on per-arm sampling probability (must be in (0, 1/K) where K is number of arms).
    /// Ensures all arms receive minimum exploration for numerical stability. Default: 1e-6
    pub min_prob: Option<f64>,
    /// Regularization coefficient (≥ 0) to encourage proximity to uniform distribution.
    /// Smooths the progression of sampling distributions toward the optimum. Default: 0.01
    pub reg0: Option<f64>,
    /// Optimization direction (Min or Max)
    pub metric_optimize: MetricConfigOptimize,
}
/// Errors that can occur when computing optimal sampling probabilities.
#[derive(Debug, Error)]
pub enum EstimateOptimalProbabilitiesError {
    #[error("Length mismatch: pull_counts = {pull_counts_len}, means = {means_len}, variances = {variances_len}")]
    MismatchedLengths {
        pull_counts_len: usize,
        means_len: usize,
        variances_len: usize,
    },
    #[error("Need at least two arms, got {num_arms}")]
    TooFewArms { num_arms: usize },
    #[error("{field} has a non-finite value at position {index}")]
    NotFinite { field: String, index: usize },
    #[error("{field} has a negative value at position {index}")]
    Negative { field: String, index: usize },
    #[error("{field} has an out-of-range value at position {index}")]
    OutOfRange { field: String, index: usize },
    #[error("The `min_prob` value of {prob} is too large for the number of arms")]
    MinProbTooLarge { prob: f64 },
    #[error("Could not compute leader arm")]
    CouldntComputeArgmax,
    #[error("Failed to build Clarabel solver")]
    CouldntBuildSolver,
    #[error("Missing variance for variant '{variant_name}' - variance must be non-null")]
    MissingVariance { variant_name: String },
}
/// Compute optimal sampling proportions for ε-best arm identification.
///
/// This function implements an allocation strategy for multi-armed bandits that
/// efficiently identifies near-optimal arms. It solves a second-order cone program (SOCP)
/// to find sampling probabilities that minimize the expected number of samples needed
/// to distinguish arms that are within ε of the best arm.
///
/// # Arguments
///
/// * `args` - A struct containing:
///   * `feedback` - Observed rewards and pull counts for each arm
///   * `epsilon` - Sub-optimality tolerance (ε ≥ 0), default 0.0
///   * `variance_floor` - Lower bound on variances for stability, default 1e-12
///   * `min_prob` - Minimum probability per arm, default 1e-6
///   * `reg0` - Regularization coefficient, default 0.01
///
/// # Returns
///
/// Returns a `HashMap` mapping variant names to their optimal sampling probabilities.
/// Probabilities are guaranteed to:
/// - Sum to 1.0 (within numerical precision)
/// - Each be ≥ min_prob
/// - Each be ≥ 0
///
/// # Errors
///
/// Returns `EstimateOptimalProbabilitiesError` if:
/// - Input vectors have mismatched lengths
/// - Fewer than 2 arms are provided
/// - Any values are non-finite (NaN or infinite)
/// - Any pull counts or variances are negative
/// - min_prob is too large for the number of arms
/// - The leader arm cannot be determined
/// - The SOCP solver fails to build
///
/// # Preconditions
///
/// The function expects (but does not currently validate):
/// - epsilon ≥ 0
/// - variance_floor ≥ 0
/// - min_prob ∈ (0, 1/K) where K is the number of arms
/// - reg0 ≥ 0
/// - All pull counts > 0
/// - All variances > 0 (after variance floor adjustment)
///
/// # Special Cases
///
/// If all arms have essentially equal means (within 1e-10) and equal variances
/// (range < 1e-10), the function returns a uniform distribution to avoid
/// numerical issues with degenerate optimization problems.
///
/// # Notes
///
/// - The function uses the CLARABEL solver for the SOCP formulation
/// - Regularization strength scales with α_t = reg0 / √(total_pulls)
/// - For ε = 0, this targets exact best-arm identification
/// - For ε > 0, arms within ε of the best are considered acceptable
///
/// # References
///
/// This implementation is based on the sample complexity lower bound described in:
///
/// Garivier, A., & Kaufmann, E. (2016). Optimal Best Arm Identification with Fixed Confidence.
/// In: JMLR: Workshop and Conference Proceedings. Vol. 49, pp. 1–30
/// <https://proceedings.mlr.press/v49/garivier16a.html>
///
/// That bound is for one-parameter exponential family models. We use the form of the
/// bound for (sub-)Gaussian rewards with known variances and simply substitute in
/// estimated variances.
pub fn estimate_optimal_probabilities(
    args: EstimateOptimalProbabilitiesArgs,
) -> Result<HashMap<String, f64>, EstimateOptimalProbabilitiesError> {
    let EstimateOptimalProbabilitiesArgs {
        feedback,
        epsilon,
        variance_floor,
        min_prob,
        reg0,
        metric_optimize,
    } = args;
    // TODO(https://github.com/tensorzero/tensorzero/issues/4282): Consider nonzero default value for epsilon
    let epsilon: f64 = epsilon.unwrap_or(0.0);
    let variance_floor: f64 = variance_floor.unwrap_or(1e-12);
    // Default min_prob is 0.0, but we apply a floor of 1e-6 for numerical stability in the optimization
    let min_prob: f64 = min_prob.unwrap_or(0.0).max(1e-6);
    let reg0: f64 = reg0.unwrap_or(0.01);

    let pull_counts: Vec<u64> = feedback.iter().map(|x| x.count).collect();

    // Negate means if we're minimizing, so we can always use argmax
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

    // Validate and extract variances - all must be present
    let variances: Result<Vec<f64>, EstimateOptimalProbabilitiesError> = feedback
        .iter()
        .map(|x| {
            x.variance
                .map(|v| (v as f64).max(variance_floor))
                .ok_or_else(|| EstimateOptimalProbabilitiesError::MissingVariance {
                    variant_name: x.variant_name.clone(),
                })
        })
        .collect();
    let variances = variances?;
    let variant_names: Vec<String> = feedback.into_iter().map(|x| x.variant_name).collect();

    // Gather the quantities required for the optimization
    let num_arms: usize = means.len();
    let num_decision_vars: usize = 2 * num_arms + 1;
    let total_pulls: u64 = pull_counts.iter().sum();
    let alpha_t: f64 = reg0 / (total_pulls as f64).sqrt(); // regularization coefficient
    let leader_arm: usize = choose_leader(&means, &variances, &pull_counts)
        .ok_or(EstimateOptimalProbabilitiesError::CouldntComputeArgmax)?;
    let leader_mean: f64 = means[leader_arm];

    // ε-margins: gap_i = μ_L - μ_i + ε  (strictly > 0 for ε > 0)
    let gaps: Vec<f64> = means.iter().map(|&x| leader_mean - x + epsilon).collect();
    // Floor gap² at 1e-6 for numerical stability in SOCP solver
    // This prevents constraint coefficients from becoming too small
    let gaps2: Vec<f64> = gaps.iter().map(|&x| (x * x).max(1e-6)).collect();

    // Edge case: if all arms have essentially equal means and variances, return uniform distribution
    // This avoids numerical issues when the optimization problem approaches degeneracy
    let all_gaps_tiny = gaps.iter().all(|&g| g.abs() < 1e-10);
    let (min_var, max_var) = variances
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
    let variance_range = max_var - min_var;
    if all_gaps_tiny && variance_range < 1e-10 {
        let probs = vec![1.0 / num_arms as f64; num_arms];
        let out: HashMap<String, f64> = variant_names.into_iter().zip(probs).collect();
        return Ok(out);
    }

    // ---------- Objective: (1/2) x^T P x + q^T x ----------
    // Quadratic part: α_t ||w - u||^2 = α_t ||w||^2 - 2α_t u^T w + const
    // => P_ww = 2α I_K ; q_w = -2α_t u ; q_t = 1

    // Build q, the linear coefficients of the objective function
    let mut q = vec![0.0; num_decision_vars];
    q[2 * num_arms] = 1.0;
    if alpha_t > 0.0 {
        let u_val = 1.0 / (num_arms as f64); // to penalize deviation from uniform
        for val in &mut q[..num_arms] {
            *val = -2.0 * alpha_t * u_val;
        }
    }

    // P, the quadratic coefficients, is a diagonal matrix with 2α_t on the first K diagonal entries and 0 elsewhere
    let P = if alpha_t > 0.0 {
        // Create sparse matrix in triplet format (row, col, value)
        let mut rows = Vec::with_capacity(num_arms);
        let mut cols = Vec::with_capacity(num_arms);
        let mut vals = Vec::with_capacity(num_arms);

        for i in 0..num_arms {
            rows.push(i);
            cols.push(i);
            vals.push(2.0 * alpha_t);
        }

        CscMatrix::new_from_triplets(num_decision_vars, num_decision_vars, rows, cols, vals)
    } else {
        // Zero matrix (no regularization)
        CscMatrix::new_from_triplets(num_decision_vars, num_decision_vars, vec![], vec![], vec![])
    };

    // ---------- Constraints Ax + s = b, s ∈ K ----------
    let mut A_rows: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();

    // (A) Equality constraint: sum_i w_i = 1 (simplex equality) --> ZeroConeT(1)
    {
        let mut row = vec![0.0; num_decision_vars];
        for val in &mut row[..num_arms] {
            *val = 1.0;
        }
        A_rows.push(row);
        b.push(1.0);
        cones.push(ZeroConeT(1));
    }

    // (B) Linear inequalities --> NonNegativeConeT
    {
        // a \geq 0
        let mut row: Vec<f64> = vec![0.0; num_decision_vars];
        row[2 * num_arms] = -1.0;
        A_rows.push(row);
        b.push(0.0);

        // Lower bounding weights at min_prob
        for i in 0..num_arms {
            let mut row: Vec<f64> = vec![0.0; num_decision_vars];
            row[i] = -1.0;
            A_rows.push(row);
            b.push(-min_prob);
        }

        // Linear constraints for the slack variables for all arms besides leader
        for i in 0..num_arms {
            if i != leader_arm {
                let mut row: Vec<f64> = vec![0.0; num_decision_vars];
                row[num_arms + i] = 1.0;
                row[num_arms + leader_arm] = 1.0;
                row[2 * num_arms] = -0.5 * gaps2[i];
                A_rows.push(row);
                b.push(0.0);
            }
        }
        cones.push(NonnegativeConeT(2 * num_arms));
    }

    // (C) SOCP constraints (3 rows per arm, one cone per arm) --> SecondOrderConeT
    // We want the cone s to satisfy: s_1 >= ||(s_2, s_3)||_2
    // where s = b - Ax, so we set up rows such that:
    //   s_1 = w_i + s_i/2
    //   s_2 = w_i - s_i/2
    //   s_3 = √2 σ_i
    {
        for i in 0..num_arms {
            // First row: s_1 = b_1 - row1·x = w_i + s_i/2
            // So we need: row1·x = -w_i - s_i/2, b_1 = 0
            let mut row1: Vec<f64> = vec![0.0; num_decision_vars];
            row1[i] = -1.0;
            row1[num_arms + i] = -0.5;
            A_rows.push(row1);
            b.push(0.0);

            // Second row: s_2 = b_2 - row2·x = w_i - s_i/2
            // So we need: row2·x = -w_i + s_i/2, b_2 = 0
            let mut row2: Vec<f64> = vec![0.0; num_decision_vars];
            row2[i] = -1.0;
            row2[num_arms + i] = 0.5;
            A_rows.push(row2);
            b.push(0.0);

            // Third row: s_3 = b_3 - row3·x = √2 σ_i
            // So we need: row3·x = 0, b_3 = √2 σ_i
            let row3: Vec<f64> = vec![0.0; num_decision_vars];
            A_rows.push(row3);
            b.push(f64::sqrt(2.0 * variances[i]));

            // Each arm gets its own 3-dimensional second-order cone
            cones.push(SecondOrderConeT(3));
        }
    }

    // Solve for the optimal weights
    let A = CscMatrix::from(&A_rows);
    let settings = DefaultSettings::<f64> {
        verbose: false,
        ..Default::default()
    };
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings)
        .map_err(|_| EstimateOptimalProbabilitiesError::CouldntBuildSolver)?;
    solver.solve();

    // Check solver status and validate solution
    match solver.solution.status {
        SolverStatus::Solved => {
            // Solution is valid, extract weights
            let x = solver.solution.x.as_slice();
            let w_star = x[0..num_arms].to_vec();

            let out: HashMap<String, f64> = variant_names.into_iter().zip(w_star).collect();
            Ok(out)
        }
        _ => {
            // Solver failed - this can happen when the problem is degenerate
            // (e.g., when means are equal and epsilon is very small)
            // Fall back to uniform distribution
            tracing::warn!(
                "SOCP solver failed with status {:?}, falling back to uniform distribution",
                solver.solution.status
            );
            let probs = vec![1.0 / num_arms as f64; num_arms];
            let out: HashMap<String, f64> = variant_names.into_iter().zip(probs).collect();
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

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

    // Helper to extract probabilities in variant order
    fn hashmap_to_vec(map: &HashMap<String, f64>, num_variants: usize) -> Vec<f64> {
        (0..num_variants)
            .map(|i| *map.get(&format!("variant_{i}")).unwrap())
            .collect()
    }

    fn assert_vecs_almost_equal(slice1: &[f64], slice2: &[f64], tol: Option<f64>) {
        assert_eq!(
            slice1.len(),
            slice2.len(),
            "Vector lengths differ: {} vs {}",
            slice1.len(),
            slice2.len()
        );

        let tol = tol.unwrap_or(1e-10);
        for (idx, (val1, val2)) in slice1.iter().zip(slice2.iter()).enumerate() {
            let diff: f64 = (val1 - val2).abs();
            assert!(
                !diff.is_nan(),
                "NaN encountered at index {idx}: {val1} vs {val2}"
            );
            assert!(diff < tol, "Values differ at index {idx}: {val1} vs {val2}");
        }
    }

    // ============================================================================
    // Tests for the ordering and basic properties of the returned probabilities
    // ============================================================================
    #[test]
    fn test_two_arms_different_variances() {
        let feedback = make_feedback(vec![10, 10], vec![0.3, 0.7], vec![1.1, 1.0]);
        let probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.1),
            variance_floor: None,
            min_prob: Some(0.05),
            reg0: Some(1.0),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        assert!(
            probs.get("variant_0") > probs.get("variant_1"),
            // probs[0] > probs[1],
            "Arm with higher variance should have higher probability"
        );
        assert!((probs.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_two_arms_equal_variances() {
        let feedback = make_feedback(vec![10, 10], vec![0.5, 0.5], vec![0.1, 0.1]);
        let probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.01),
            variance_floor: None,
            min_prob: None,
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        assert!((probs.values().sum::<f64>() - 1.0).abs() < 1e-6);
        // With equal variances and only two arms, probabilities should be roughly equal
        assert!((probs.get("variant_0").unwrap() - probs.get("variant_1").unwrap()).abs() < 0.1);
    }

    #[test]
    fn test_equal_means_different_variances_above_floor() {
        let feedback = make_feedback(vec![100, 100], vec![0.5, 0.5], vec![0.1, 0.5]);
        let probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.0),
            variance_floor: None,
            min_prob: Some(1e-6),
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Solver should succeed and return valid probabilities
        assert!(
            (probs.values().sum::<f64>() - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1"
        );

        // The higher-variance arm should get more sampling
        assert!(
            probs.get("variant_1").unwrap() > probs.get("variant_0").unwrap(),
            "Higher variance arm should get more probability"
        );
    }

    #[test]
    fn test_equal_means_different_variances_small_epsilon() {
        // Test with a small but non-zero epsilon
        let feedback = make_feedback(vec![100, 100], vec![0.5, 0.5], vec![0.1, 0.5]);
        let probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.01),
            variance_floor: None,
            min_prob: Some(1e-6),
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        assert!(
            (probs.values().sum::<f64>() - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1"
        );
    }

    #[test]
    fn test_nearly_equal_means_different_variances() {
        // Test with nearly equal means (within floating point tolerance)
        let feedback = make_feedback(vec![100, 100], vec![0.5, 0.5 + 1e-12], vec![0.1, 0.5]);
        let probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.0),
            variance_floor: None,
            min_prob: Some(1e-6),
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // With gap² floor, solver should succeed
        assert!(
            (probs.values().sum::<f64>() - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1"
        );
    }

    #[test]
    fn test_equal_means_equal_variances_above_floor() {
        // Test that variance_floor is applied when both variances are below it
        let feedback = make_feedback(vec![100, 100], vec![0.5, 0.5], vec![1.0, 1.0 + 1e-10]);
        let probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.0),
            variance_floor: Some(0.01),
            min_prob: Some(1e-6),
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Should return valid probabilities that sum to 1
        assert!(
            (probs.values().sum::<f64>() - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1"
        );

        // Since both variances are floored to the same value and means are equal,
        // probabilities should be approximately equal
        assert!(
            (probs.get("variant_0").unwrap() - probs.get("variant_1").unwrap()).abs() < 1e-10,
            "With equal floored variances and equal means, probabilities should be similar"
        );
    }

    #[test]
    fn test_equal_means_equal_variances_below_floor() {
        // Test that variance_floor is applied when both variances are below it
        let feedback = make_feedback(vec![100, 100], vec![0.5, 0.5], vec![1e-15, 1e-14]);
        let probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.0),
            variance_floor: Some(0.01),
            min_prob: Some(1e-6),
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Should return valid probabilities that sum to 1
        assert!(
            (probs.values().sum::<f64>() - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1"
        );

        // Since both variances are floored to the same value and means are equal,
        // probabilities should be approximately equal
        assert!(
            (probs.get("variant_0").unwrap() - probs.get("variant_1").unwrap()).abs() < 1e-10,
            "With equal floored variances and equal means, probabilities should be similar"
        );
    }

    #[test]
    fn test_min_prob_constraint() {
        // All arms should respect minimum probability
        let min_prob = 0.1;
        let feedback = make_feedback(vec![10, 10, 10], vec![0.2, 0.3, 0.9], vec![0.1, 0.1, 10.0]);
        let probs = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.01),
            variance_floor: None,
            min_prob: Some(min_prob),
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        for &p in probs.values() {
            assert!(
                p >= min_prob - 1e-6,
                "Probability {p} violates min_prob {min_prob}"
            );
        }
        assert!((probs.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_high_variance_arm() {
        // Higher variance means more sampling needed for accurate mean estimate
        let feedback = make_feedback(vec![10, 10], vec![0.5, 0.5], vec![0.01, 0.5]);
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.01),
            variance_floor: None,
            min_prob: None,
            reg0: Some(1.0),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        let probs = hashmap_to_vec(&probs_map, 2);
        assert!(probs[1] > probs[0], "High variance arm needs more samples");
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_regularization_effect() {
        // Higher reg0 should pull probabilities toward uniform
        let feedback_no_reg = make_feedback(vec![100, 100], vec![0.3, 0.7], vec![0.3, 0.1]);
        let probs_no_reg = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback: feedback_no_reg,
            epsilon: Some(0.0),
            variance_floor: None,
            min_prob: None,
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        let feedback_with_reg = make_feedback(vec![100, 100], vec![0.3, 0.7], vec![0.3, 0.1]);
        let probs_with_reg = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback: feedback_with_reg,
            epsilon: Some(0.0),
            variance_floor: None,
            min_prob: None,
            reg0: Some(10.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // With regularization, probabilities should be closer to 0.5
        let diff_no_reg = (*probs_no_reg.get("variant_0").unwrap() - 0.5).abs();
        let diff_with_reg = (*probs_with_reg.get("variant_0").unwrap() - 0.5).abs();
        assert!(diff_with_reg < diff_no_reg);
    }

    #[test]
    fn test_many_arms() {
        // Test with 10 arms - spread out means
        let min_prob = 0.05;
        let feedback = make_feedback(
            vec![10; 10],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            vec![0.1; 10],
        );
        // Test with optimize=max
        let probs_map_max = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.0),
            variance_floor: None,
            min_prob: Some(min_prob),
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // All probabilities should be non-negative
        for &p in probs_map_max.values() {
            assert!(p >= min_prob - 1e-6);
        }
        // Sum to 1
        assert!((probs_map_max.values().sum::<f64>() - 1.0).abs() < 1e-6);
        // The highest should get most sampling
        let probs_vec = hashmap_to_vec(&probs_map_max, 10);
        assert!(probs_vec[9] >= probs_vec[0]);

        let feedback = make_feedback(
            vec![10; 10],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            vec![0.1; 10],
        );
        // Test with optimize=min
        let probs_map_min = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.0),
            variance_floor: None,
            min_prob: Some(min_prob),
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        // All probabilities should be non-negative
        for &p in probs_map_min.values() {
            assert!(p >= min_prob - 1e-6);
        }
        // Sum to 1
        assert!((probs_map_min.values().sum::<f64>() - 1.0).abs() < 1e-6);
        // The lowest should get most sampling
        let probs_vec = hashmap_to_vec(&probs_map_min, 10);
        assert!(probs_vec[0] >= probs_vec[9]);
    }

    #[test]
    fn test_close_competition() {
        // Two arms very close in mean - both need substantial sampling
        let feedback = make_feedback(vec![10, 10, 10], vec![0.50, 0.51, 0.3], vec![0.1, 0.1, 0.1]);
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.0),
            variance_floor: None,
            min_prob: None,
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
        // The close competitors (arms 0 and 1) should together get most probability
        let probs = hashmap_to_vec(&probs_map, 3);
        assert!(probs[0] + probs[1] > probs[2]);
    }

    #[test]
    fn test_variance_floor_applied() {
        // Variance floor should lower bound all variances
        let feedback = make_feedback(vec![10, 10], vec![0.5, 0.5], vec![1e-20, 1e-20]);
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.0),
            variance_floor: Some(0.01),
            min_prob: None,
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Should not fail and should return valid probabilities
        let probs = hashmap_to_vec(&probs_map, 2);
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
        assert_vecs_almost_equal(&probs, &[0.5, 0.5], Some(1e-4));
    }

    #[test]
    fn test_zero_variance_with_floor() {
        // Zero variance should be handled by variance floor
        let feedback = make_feedback(
            vec![10, 10, 10],
            vec![0.3, 0.5, 0.7],
            vec![0.0001, 0.0, 0.0],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.0),
            variance_floor: Some(0.001),
            min_prob: None,
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
        // Arm with smallest mean should get most sampling
        let probs = hashmap_to_vec(&probs_map, 3);
        assert!(probs[0] > probs[2]);
    }

    #[test]
    fn test_basic_constraints() {
        // Test that basic constraints are always satisfied
        let feedback = make_feedback(
            vec![15, 30, 20, 35],
            vec![0.25, 0.55, 0.40, 0.60],
            vec![0.05, 0.15, 0.10, 0.20],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.05),
            variance_floor: Some(1e-8),
            min_prob: Some(0.05),
            reg0: Some(2.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Verify basic constraints
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
        for &p in probs_map.values() {
            assert!(p >= 0.05 - 1e-9, "min_prob constraint violated");
            assert!(p >= 0.0, "Probability is negative");
        }
    }

    // ===================================================================================
    // Tests with reference solutions from cvxpy (using CLARABEL solver) or scipy.minimize
    // ===================================================================================
    #[test]
    fn test_three_arms_varied_cvxpy() {
        let feedback = make_feedback(
            vec![10, 10, 10],
            vec![0.20, 0.60, 0.40],
            vec![0.10, 0.20, 0.15],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.02),
            variance_floor: None,
            min_prob: Some(0.05),
            reg0: Some(0.5),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![0.050000000981238, 0.509054656289642, 0.440945342729120];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_four_arms_with_reg_cvxpy() {
        let feedback = make_feedback(
            vec![15, 30, 20, 35],
            vec![0.25, 0.55, 0.40, 0.60],
            vec![0.05, 0.15, 0.10, 0.20],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.05),
            variance_floor: None,
            min_prob: Some(0.05),
            reg0: Some(2.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![
            0.050000000010182,
            0.417754702981922,
            0.050000000506407,
            0.482245296501354,
        ];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_close_competition_cvxpy() {
        let feedback = make_feedback(
            vec![10, 10, 10],
            vec![0.50, 0.51, 0.30],
            vec![0.10, 0.10, 0.10],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.01),
            variance_floor: None,
            min_prob: Some(0.01),
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![0.494996029290799, 0.495003970556052, 0.010000000211085];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ten_arms_cvxpy() {
        let feedback = make_feedback(
            vec![10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            vec![0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
            vec![0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.02),
            variance_floor: None,
            min_prob: Some(0.01),
            reg0: Some(0.1),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![
            0.009999998037405,
            0.009999998021350,
            0.009999998050981,
            0.009999998100266,
            0.011316045868696,
            0.017600153556836,
            0.031270947718469,
            0.072188003456080,
            0.409811419219896,
            0.417813437971073,
        ];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    // Tests with high variance (variance > 1) representing different reward scales
    #[test]
    fn test_high_variance_gaussians_cvxpy() {
        let feedback = make_feedback(
            vec![20, 20, 20],
            vec![10.00, 12.00, 11.00],
            vec![2.00, 3.50, 2.50],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.5),
            variance_floor: None,
            min_prob: Some(0.05),
            reg0: Some(0.1),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![0.411818058014078, 0.144184113193170, 0.443997828792751];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_very_high_variance_cvxpy() {
        let feedback = make_feedback(vec![30, 30], vec![50.00, 55.00], vec![10.00, 15.00]);
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(1.0),
            variance_floor: None,
            min_prob: Some(0.01),
            reg0: Some(0.5),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![0.450083389074646, 0.549916610924264];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mixed_variance_scales_cvxpy() {
        let feedback = make_feedback(
            vec![15, 15, 15, 15],
            vec![1.00, 5.00, 3.00, 7.00],
            vec![0.10, 2.00, 0.50, 4.00],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.2),
            variance_floor: None,
            min_prob: Some(0.05),
            reg0: Some(1.0),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![
            0.164052326317122,
            0.274275620684497,
            0.320837210430485,
            0.240834842568170,
        ];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_small_means_large_variance_cvxpy() {
        let feedback = make_feedback(
            vec![25, 25, 25],
            vec![0.10, 0.30, 0.20],
            vec![5.00, 8.00, 6.00],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.05),
            variance_floor: None,
            min_prob: Some(0.1),
            reg0: Some(0.2),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![0.099999996666504, 0.482302162018304, 0.417697840883230];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_five_arms_high_variance_cvxpy() {
        let feedback = make_feedback(
            vec![10, 10, 10, 10, 10],
            vec![20.00, 25.00, 22.00, 28.00, 24.00],
            vec![3.00, 4.00, 2.50, 5.00, 3.50],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(1.0),
            variance_floor: None,
            min_prob: Some(0.05),
            reg0: Some(0.5),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![
            0.407341605286492,
            0.079666735242631,
            0.355686015295549,
            0.050000000075998,
            0.107305644099329,
        ];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_asymmetric_high_variance_cvxpy() {
        let feedback = make_feedback(vec![50, 50], vec![100.00, 105.00], vec![1.00, 20.00]);
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(2.0),
            variance_floor: None,
            min_prob: Some(0.01),
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![0.182752086850448, 0.817247913149552];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    // Tests using scipy.minimize solutions to verify SOCP reformulation

    #[test]
    fn test_two_arms_simple_scipy() {
        let feedback = make_feedback(vec![20, 20], vec![0.50, 0.70], vec![0.10, 0.15]);
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.1),
            variance_floor: None,
            min_prob: Some(0.05),
            reg0: Some(0.1),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Expected solution from scipy (trust-constr with SLSQP fallback)
        let expected = vec![0.449525654401765, 0.550474345598233];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_three_arms_moderate_scipy() {
        let feedback = make_feedback(
            vec![15, 15, 15],
            vec![1.00, 1.50, 1.20],
            vec![0.20, 0.30, 0.25],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.2),
            variance_floor: None,
            min_prob: Some(0.05),
            reg0: Some(0.2),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        // Expected solution from scipy (trust-constr with SLSQP fallback)
        let expected = vec![0.423761527414760, 0.114067986187023, 0.462170486398217];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_very_high_variance_scipy() {
        let feedback = make_feedback(vec![30, 30], vec![50.00, 55.00], vec![10.00, 15.00]);
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(1.0),
            variance_floor: None,
            min_prob: Some(0.01),
            reg0: Some(0.5),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Expected solution from scipy (trust-constr with SLSQP fallback)
        let expected = vec![0.450070015802657, 0.549929984197342];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_four_arms_different_scales_scipy() {
        let feedback = make_feedback(
            vec![20, 20, 20, 20],
            vec![2.00, 3.00, 2.50, 3.50],
            vec![0.50, 1.00, 0.75, 1.50],
        );
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.3),
            variance_floor: None,
            min_prob: Some(0.05),
            reg0: Some(0.3),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();

        // Expected solution from scipy (trust-constr with SLSQP fallback)
        let expected = vec![
            0.352531945768924,
            0.139366907316944,
            0.408479860960564,
            0.099621285953567,
        ];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_small_epsilon_scipy() {
        let feedback = make_feedback(vec![50, 50], vec![10.00, 10.50], vec![1.00, 1.20]);
        let probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback,
            epsilon: Some(0.05),
            variance_floor: None,
            min_prob: Some(0.1),
            reg0: Some(0.1),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();

        // Expected solution from scipy (trust-constr with SLSQP fallback)
        let expected = vec![0.477229489718064, 0.522770510281936];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    // Helper function to compute L2 distance between two probability vectors
    fn l2_distance(v1: &[f64], v2: &[f64]) -> f64 {
        assert_eq!(v1.len(), v2.len(), "Vectors must have same length");
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    // Helper function to generate random offsets for each arm, scaled by a factor
    fn generate_random_offsets(rng: &mut StdRng, num_arms: usize, scale: f64) -> Vec<f64> {
        (0..num_arms).map(|_| rng.random::<f64>() * scale).collect()
    }

    // Helper function to check that averaged L2 distances show monotonic convergence
    // Averaging over multiple runs should cancel out nonomonotonicity from the nonlinear optimization
    fn check_monotone_decreasing_distances(l2_distances: &[f64]) {
        for i in 1..l2_distances.len() {
            assert!(
                l2_distances[i] <= l2_distances[i - 1],
                "Averaged L2 distance should be monotone decreasing: distance[{}] = {} > distance[{}] = {}",
                i,
                l2_distances[i],
                i - 1,
                l2_distances[i - 1]
            );
        }
    }

    // ======================================================================================
    // Tests for convergence of estimated optimal probabilities to true optimal probabilities
    // as sample means and variances converge. This convergence may not be monotone in any
    // given problem instance, so we average over multiple random instances. Due to
    // randomness, these tests could fail sometimes.
    // ======================================================================================
    #[test]
    fn test_convergence_two_arms() {
        const NUM_RUNS: usize = 10;

        // True means and variances
        let true_means = vec![0.5, 0.7];
        let true_variances = vec![0.1, 0.2];
        let pull_counts = vec![100, 100];
        let num_arms = true_means.len();

        // Compute true optimal probabilities
        let true_feedback = make_feedback(
            pull_counts.clone(),
            true_means.clone(),
            true_variances.clone(),
        );
        let true_probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback: true_feedback,
            epsilon: Some(0.0),
            variance_floor: None,
            min_prob: Some(1e-6),
            reg0: Some(0.0),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();
        let true_probs = hashmap_to_vec(&true_probs_map, num_arms);

        // Create sequence with decreasing scales, each arm gets random offset
        let scales = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001];

        // Average L2 distances over multiple runs
        let mut avg_l2_distances = vec![0.0; scales.len()];

        for run in 0..NUM_RUNS {
            let mut rng = StdRng::seed_from_u64(42 + run as u64);

            for (scale_idx, &scale) in scales.iter().enumerate() {
                let mean_offsets = generate_random_offsets(&mut rng, num_arms, scale);
                let variance_offsets = generate_random_offsets(&mut rng, num_arms, scale);

                let sample_means: Vec<f64> = true_means
                    .iter()
                    .zip(&mean_offsets)
                    .map(|(&m, &offset)| m + offset)
                    .collect();
                let sample_variances: Vec<f64> = true_variances
                    .iter()
                    .zip(&variance_offsets)
                    .map(|(&v, &offset)| v + offset)
                    .collect();

                let sample_feedback =
                    make_feedback(pull_counts.clone(), sample_means, sample_variances);
                let sample_probs_map =
                    estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
                        feedback: sample_feedback,
                        epsilon: Some(0.0),
                        variance_floor: None,
                        min_prob: Some(1e-6),
                        reg0: Some(0.0),
                        metric_optimize: MetricConfigOptimize::Max,
                    })
                    .unwrap();
                let sample_probs = hashmap_to_vec(&sample_probs_map, num_arms);

                let distance = l2_distance(&sample_probs, &true_probs);
                avg_l2_distances[scale_idx] += distance;
            }
        }

        check_monotone_decreasing_distances(&avg_l2_distances);
    }

    #[test]
    fn test_convergence_three_arms_varied() {
        const NUM_RUNS: usize = 10;

        let true_means = vec![0.3, 0.6, 0.5];
        let true_variances = vec![0.1, 0.2, 0.15];
        let pull_counts = vec![50, 50, 50];
        let num_arms = true_means.len();

        let true_feedback = make_feedback(
            pull_counts.clone(),
            true_means.clone(),
            true_variances.clone(),
        );
        let true_probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback: true_feedback,
            epsilon: Some(0.05),
            variance_floor: None,
            min_prob: Some(0.01),
            reg0: Some(0.1),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();
        let true_probs = hashmap_to_vec(&true_probs_map, num_arms);

        let scales = [0.15, 0.1, 0.05, 0.02, 0.01, 0.005];
        let mut avg_l2_distances = vec![0.0; scales.len()];

        for run in 0..NUM_RUNS {
            let mut rng = StdRng::seed_from_u64(123 + run as u64);

            for (scale_idx, &scale) in scales.iter().enumerate() {
                let mean_offsets = generate_random_offsets(&mut rng, num_arms, scale);
                let variance_offsets = generate_random_offsets(&mut rng, num_arms, scale);

                let sample_means: Vec<f64> = true_means
                    .iter()
                    .zip(&mean_offsets)
                    .map(|(&m, &offset)| m + offset)
                    .collect();
                let sample_variances: Vec<f64> = true_variances
                    .iter()
                    .zip(&variance_offsets)
                    .map(|(&v, &offset)| v + offset)
                    .collect();

                let sample_feedback =
                    make_feedback(pull_counts.clone(), sample_means, sample_variances);
                let sample_probs_map =
                    estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
                        feedback: sample_feedback,
                        epsilon: Some(0.05),
                        variance_floor: None,
                        min_prob: Some(0.01),
                        reg0: Some(0.1),
                        metric_optimize: MetricConfigOptimize::Min,
                    })
                    .unwrap();
                let sample_probs = hashmap_to_vec(&sample_probs_map, num_arms);

                avg_l2_distances[scale_idx] += l2_distance(&sample_probs, &true_probs);
            }
        }

        check_monotone_decreasing_distances(&avg_l2_distances);
    }

    #[test]
    fn test_convergence_five_arms() {
        const NUM_RUNS: usize = 10;

        let true_means = vec![0.2, 0.5, 0.4, 0.6, 0.3];
        let true_variances = vec![0.1, 0.3, 0.2, 0.4, 0.15];
        let pull_counts = vec![100, 100, 100, 100, 100];
        let num_arms = true_means.len();

        let true_feedback = make_feedback(
            pull_counts.clone(),
            true_means.clone(),
            true_variances.clone(),
        );
        let true_probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback: true_feedback,
            epsilon: Some(0.02),
            variance_floor: None,
            min_prob: Some(0.05),
            reg0: Some(0.5),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();
        let true_probs = hashmap_to_vec(&true_probs_map, num_arms);

        let scales = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002];
        let mut avg_l2_distances = vec![0.0; scales.len()];

        for run in 0..NUM_RUNS {
            let mut rng = StdRng::seed_from_u64(456 + run as u64);

            for (scale_idx, &scale) in scales.iter().enumerate() {
                let mean_offsets = generate_random_offsets(&mut rng, num_arms, scale);
                let variance_offsets = generate_random_offsets(&mut rng, num_arms, scale);

                let sample_means: Vec<f64> = true_means
                    .iter()
                    .zip(&mean_offsets)
                    .map(|(&m, &offset)| m + offset)
                    .collect();
                let sample_variances: Vec<f64> = true_variances
                    .iter()
                    .zip(&variance_offsets)
                    .map(|(&v, &offset)| v + offset)
                    .collect();

                let sample_feedback =
                    make_feedback(pull_counts.clone(), sample_means, sample_variances);
                let sample_probs_map =
                    estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
                        feedback: sample_feedback,
                        epsilon: Some(0.02),
                        variance_floor: None,
                        min_prob: Some(0.05),
                        reg0: Some(0.5),
                        metric_optimize: MetricConfigOptimize::Max,
                    })
                    .unwrap();
                let sample_probs = hashmap_to_vec(&sample_probs_map, num_arms);

                avg_l2_distances[scale_idx] += l2_distance(&sample_probs, &true_probs);
            }
        }

        check_monotone_decreasing_distances(&avg_l2_distances);
    }

    #[test]
    fn test_convergence_high_variance_arms() {
        const NUM_RUNS: usize = 10;

        let true_means = vec![10.0, 15.0, 12.0];
        let true_variances = vec![2.0, 5.0, 3.0];
        let pull_counts = vec![200, 200, 200];
        let num_arms = true_means.len();

        let true_feedback = make_feedback(
            pull_counts.clone(),
            true_means.clone(),
            true_variances.clone(),
        );
        let true_probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback: true_feedback,
            epsilon: Some(0.5),
            variance_floor: None,
            min_prob: Some(0.01),
            reg0: Some(0.1),
            metric_optimize: MetricConfigOptimize::Min,
        })
        .unwrap();
        let true_probs = hashmap_to_vec(&true_probs_map, num_arms);

        let scales = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05];
        let mut avg_l2_distances = vec![0.0; scales.len()];

        for run in 0..NUM_RUNS {
            let mut rng = StdRng::seed_from_u64(789 + run as u64);

            for (scale_idx, &scale) in scales.iter().enumerate() {
                let mean_offsets = generate_random_offsets(&mut rng, num_arms, scale);
                let variance_offsets = generate_random_offsets(&mut rng, num_arms, scale);

                let sample_means: Vec<f64> = true_means
                    .iter()
                    .zip(&mean_offsets)
                    .map(|(&m, &offset)| m + offset)
                    .collect();
                let sample_variances: Vec<f64> = true_variances
                    .iter()
                    .zip(&variance_offsets)
                    .map(|(&v, &offset)| v + offset)
                    .collect();

                let sample_feedback =
                    make_feedback(pull_counts.clone(), sample_means, sample_variances);
                let sample_probs_map =
                    estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
                        feedback: sample_feedback,
                        epsilon: Some(0.5),
                        variance_floor: None,
                        min_prob: Some(0.01),
                        reg0: Some(0.1),
                        metric_optimize: MetricConfigOptimize::Min,
                    })
                    .unwrap();
                let sample_probs = hashmap_to_vec(&sample_probs_map, num_arms);

                avg_l2_distances[scale_idx] += l2_distance(&sample_probs, &true_probs);
            }
        }

        check_monotone_decreasing_distances(&avg_l2_distances);
    }

    #[test]
    fn test_convergence_ten_arms() {
        const NUM_RUNS: usize = 10;

        let true_means = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let true_variances = vec![0.1; 10];
        let pull_counts = vec![100; 10];
        let num_arms = true_means.len();

        let true_feedback = make_feedback(
            pull_counts.clone(),
            true_means.clone(),
            true_variances.clone(),
        );
        let true_probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback: true_feedback,
            epsilon: Some(0.05),
            variance_floor: None,
            min_prob: Some(0.01),
            reg0: Some(0.2),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();
        let true_probs = hashmap_to_vec(&true_probs_map, num_arms);

        let scales = [0.08, 0.05, 0.03, 0.02, 0.01, 0.005];
        let mut avg_l2_distances = vec![0.0; scales.len()];

        for run in 0..NUM_RUNS {
            let mut rng = StdRng::seed_from_u64(101112 + run as u64);

            for (scale_idx, &scale) in scales.iter().enumerate() {
                let mean_offsets = generate_random_offsets(&mut rng, num_arms, scale);
                let variance_offsets = generate_random_offsets(&mut rng, num_arms, scale);

                let sample_means: Vec<f64> = true_means
                    .iter()
                    .zip(&mean_offsets)
                    .map(|(&m, &offset)| m + offset)
                    .collect();
                let sample_variances: Vec<f64> = true_variances
                    .iter()
                    .zip(&variance_offsets)
                    .map(|(&v, &offset)| v + offset)
                    .collect();

                let sample_feedback =
                    make_feedback(pull_counts.clone(), sample_means, sample_variances);
                let sample_probs_map =
                    estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
                        feedback: sample_feedback,
                        epsilon: Some(0.05),
                        variance_floor: None,
                        min_prob: Some(0.01),
                        reg0: Some(0.2),
                        metric_optimize: MetricConfigOptimize::Max,
                    })
                    .unwrap();
                let sample_probs = hashmap_to_vec(&sample_probs_map, num_arms);

                avg_l2_distances[scale_idx] += l2_distance(&sample_probs, &true_probs);
            }
        }

        check_monotone_decreasing_distances(&avg_l2_distances);
    }

    #[test]
    fn test_convergence_close_competition() {
        const NUM_RUNS: usize = 10;

        // Test with very close means - convergence should still be monotone
        let true_means = vec![0.50, 0.51, 0.30];
        let true_variances = vec![0.1, 0.1, 0.1];
        let pull_counts = vec![100, 100, 100];
        let num_arms = true_means.len();

        let true_feedback = make_feedback(
            pull_counts.clone(),
            true_means.clone(),
            true_variances.clone(),
        );
        let true_probs_map = estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
            feedback: true_feedback,
            epsilon: Some(0.01),
            variance_floor: None,
            min_prob: Some(0.01),
            reg0: Some(0.1),
            metric_optimize: MetricConfigOptimize::Max,
        })
        .unwrap();
        let true_probs = hashmap_to_vec(&true_probs_map, num_arms);

        let scales = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002];
        let mut avg_l2_distances = vec![0.0; scales.len()];

        for run in 0..NUM_RUNS {
            let mut rng = StdRng::seed_from_u64(131415 + run as u64);

            for (scale_idx, &scale) in scales.iter().enumerate() {
                let mean_offsets = generate_random_offsets(&mut rng, num_arms, scale);
                let variance_offsets = generate_random_offsets(&mut rng, num_arms, scale);

                let sample_means: Vec<f64> = true_means
                    .iter()
                    .zip(&mean_offsets)
                    .map(|(&m, &offset)| m + offset)
                    .collect();
                let sample_variances: Vec<f64> = true_variances
                    .iter()
                    .zip(&variance_offsets)
                    .map(|(&v, &offset)| v + offset)
                    .collect();

                let sample_feedback =
                    make_feedback(pull_counts.clone(), sample_means, sample_variances);
                let sample_probs_map =
                    estimate_optimal_probabilities(EstimateOptimalProbabilitiesArgs {
                        feedback: sample_feedback,
                        epsilon: Some(0.01),
                        variance_floor: None,
                        min_prob: Some(0.01),
                        reg0: Some(0.1),
                        metric_optimize: MetricConfigOptimize::Max,
                    })
                    .unwrap();
                let sample_probs = hashmap_to_vec(&sample_probs_map, num_arms);

                avg_l2_distances[scale_idx] += l2_distance(&sample_probs, &true_probs);
            }
        }

        check_monotone_decreasing_distances(&avg_l2_distances);
    }
}
