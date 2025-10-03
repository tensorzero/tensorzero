#![allow(non_snake_case)]
#![expect(dead_code)]
use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SecondOrderConeT, SupportedConeT,
    ZeroConeT,
};
use std::collections::HashMap;
use thiserror::Error;

use crate::db::FeedbackByVariant;

pub fn argmax<T: PartialOrd>(slice: &[T]) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
}
// TODO: Make sure inputs are validated upstream:
//    - Vectors of same length K, with K >= 2, no NAs
//    - pull_counts > 0
//    - variances > 0
#[derive(Debug, Error)]
pub enum OptimalProbsError {
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
}

/// Compute ε-aware optimal sampling proportions for sub-Gaussian rewards
pub fn estimate_optimal_probabilities(
    feedback: Vec<FeedbackByVariant>,
    epsilon: Option<f64>,
    ridge_variance: Option<f64>,
    min_prob: Option<f64>,
    reg0: Option<f64>,
) -> Result<HashMap<String, f64>, OptimalProbsError> {
    let epsilon: f64 = epsilon.unwrap_or(0.0);
    let ridge_variance: f64 = ridge_variance.unwrap_or(1e-12);
    let min_prob: f64 = min_prob.unwrap_or(1e-6);
    let reg0: f64 = reg0.unwrap_or(0.01);

    let variant_names: Vec<String> = feedback.iter().map(|x| x.variant_name.clone()).collect();
    let pull_counts: Vec<u64> = feedback.iter().map(|x| x.count).collect();
    let means: Vec<f64> = feedback.iter().map(|x| x.mean as f64).collect();
    let variances: Vec<f64> = feedback
        .iter()
        .map(|x| (x.variance as f64).max(ridge_variance))
        .collect();

    // Gather the quantities required for the optimization
    let num_arms: usize = means.len();
    let num_decision_vars: usize = 2 * num_arms + 1;
    let total_pulls: u64 = pull_counts.iter().sum();
    let alpha_t: f64 = reg0 / (total_pulls as f64).sqrt(); // regularization coefficient
    let leader_arm: usize = argmax(&means).ok_or(OptimalProbsError::CouldntComputeArgmax)?;
    let leader_mean: f64 = means[leader_arm];

    // ε-margins: gap_i = μ_L - μ_i + ε  (strictly > 0 for ε > 0)
    let gaps: Vec<f64> = means.iter().map(|&x| leader_mean - x + epsilon).collect();
    let gaps2: Vec<f64> = gaps.iter().map(|&x| (x * x).max(1e-16)).collect();

    // Edge case: if all arms have essentially equal means and variances, return uniform distribution
    // This avoids numerical issues when the optimization problem becomes degenerate
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
        .map_err(|_| OptimalProbsError::CouldntBuildSolver)?;
    solver.solve();

    let x = solver.solution.x.as_slice();
    let w_star = x[0..num_arms].to_vec();

    let out: HashMap<String, f64> = variant_names.into_iter().zip(w_star).collect();
    Ok(out)
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

    #[test]
    fn test_two_arms_different_variances() {
        let feedback = make_feedback(vec![10, 10], vec![0.3, 0.7], vec![1.1, 1.0]);
        let probs =
            estimate_optimal_probabilities(feedback, Some(0.1), None, Some(0.05), Some(1.0))
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
        let probs =
            estimate_optimal_probabilities(feedback, Some(0.01), None, None, Some(0.0)).unwrap();

        assert!((probs.values().sum::<f64>() - 1.0).abs() < 1e-6);
        // With equal variances and only two arms, probabilities should be roughly equal
        assert!((probs.get("variant_0").unwrap() - probs.get("variant_1").unwrap()).abs() < 0.1);
    }

    #[test]
    fn test_min_prob_constraint() {
        // All arms should respect minimum probability
        let min_prob = 0.1;
        let feedback = make_feedback(vec![10, 10, 10], vec![0.2, 0.3, 0.9], vec![0.1, 0.1, 10.0]);
        let probs =
            estimate_optimal_probabilities(feedback, Some(0.01), None, Some(min_prob), Some(0.0))
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.01), // epsilon
            None,       // ridge_variance
            None,       // min_prob
            Some(1.0),  // reg0
        )
        .unwrap();

        let probs = hashmap_to_vec(&probs_map, 2);
        assert!(probs[1] > probs[0], "High variance arm needs more samples");
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_regularization_effect() {
        // Higher reg0 should pull probabilities toward uniform
        let feedback_no_reg = make_feedback(vec![100, 100], vec![0.3, 0.7], vec![0.3, 0.1]);
        let probs_no_reg = estimate_optimal_probabilities(
            feedback_no_reg,
            Some(0.0), // epsilon
            None,      // ridge_variance
            None,      // min_prob
            Some(0.0), // reg0
        )
        .unwrap();

        let feedback_with_reg = make_feedback(vec![100, 100], vec![0.3, 0.7], vec![0.3, 0.1]);
        let probs_with_reg = estimate_optimal_probabilities(
            feedback_with_reg,
            Some(0.0),  // epsilon
            None,       // ridge_variance
            None,       // min_prob
            Some(10.0), // reg0
        )
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.0),      // epsilon
            None,           // ridge_variance
            Some(min_prob), // min_prob
            Some(0.0),      // reg0
        )
        .unwrap();

        // All probabilities should be non-negative
        for &p in probs_map.values() {
            assert!(p >= min_prob - 1e-6);
        }
        // Sum to 1
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
        // The highest should get most sampling
        let probs_vec = hashmap_to_vec(&probs_map, 10);
        assert!(probs_vec[9] >= probs_vec[0]);
    }

    #[test]
    fn test_close_competition() {
        // Two arms very close in mean - both need substantial sampling
        let feedback = make_feedback(vec![10, 10, 10], vec![0.50, 0.51, 0.3], vec![0.1, 0.1, 0.1]);
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.0), // epsilon
            None,      // ridge_variance
            None,      // min_prob
            Some(0.0), // reg0
        )
        .unwrap();

        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
        // The close competitors (arms 0 and 1) should together get most probability
        let probs = hashmap_to_vec(&probs_map, 3);
        assert!(probs[0] + probs[1] > probs[2]);
    }

    #[test]
    fn test_ridge_variance_applied() {
        // Ridge variance should lower bound all variances
        let feedback = make_feedback(vec![10, 10], vec![0.5, 0.5], vec![1e-20, 1e-20]);
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.0),  // epsilon
            Some(0.01), // ridge_variance
            None,       // min_prob
            Some(0.0),  // reg0
        )
        .unwrap();

        // Should not fail and should return valid probabilities
        let probs = hashmap_to_vec(&probs_map, 2);
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
        assert_vecs_almost_equal(&probs, &[0.5, 0.5], Some(1e-4));
    }

    #[test]
    fn test_zero_variance_with_ridge() {
        // Zero variance should be handled by ridge
        let feedback = make_feedback(vec![10, 10], vec![0.3, 0.7], vec![0.0001, 0.0]);
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.0),   // epsilon
            Some(0.001), // ridge_variance
            None,        // min_prob
            Some(0.0),   // reg0
        )
        .unwrap();

        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
        // Arm with higher mean and variance needs more sampling
        let probs = hashmap_to_vec(&probs_map, 2);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_basic_constraints() {
        // Test that basic constraints are always satisfied
        let feedback = make_feedback(
            vec![15, 30, 20, 35],
            vec![0.25, 0.55, 0.40, 0.60],
            vec![0.05, 0.15, 0.10, 0.20],
        );
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.05), // epsilon
            Some(1e-8), // ridge_variance
            Some(0.05), // min_prob
            Some(2.0),  // reg0
        )
        .unwrap();

        // Verify basic constraints
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
        for &p in probs_map.values() {
            assert!(p >= 0.05 - 1e-9, "min_prob constraint violated");
            assert!(p >= 0.0, "Probability is negative");
        }
    }

    // Tests with reference solutions from cvxpy (using CLARABEL solver) or scipy.minimize
    #[test]
    fn test_three_arms_varied_cvxpy() {
        let feedback = make_feedback(
            vec![10, 10, 10],
            vec![0.20, 0.60, 0.40],
            vec![0.10, 0.20, 0.15],
        );
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.02), // epsilon
            None,       // ridge_variance
            Some(0.05), // min_prob
            Some(0.5),  // reg0
        )
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.05), // epsilon
            None,       // ridge_variance
            Some(0.05), // min_prob
            Some(2.0),  // reg0
        )
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.01), // epsilon
            None,       // ridge_variance
            Some(0.01), // min_prob
            Some(0.0),  // reg0
        )
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.02), // epsilon
            None,       // ridge_variance
            Some(0.01), // min_prob
            Some(0.1),  // reg0
        )
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.5),  // epsilon
            None,       // ridge_variance
            Some(0.05), // min_prob
            Some(0.1),  // reg0
        )
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![0.069702913270257, 0.508019009489254, 0.422278077240489];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_very_high_variance_cvxpy() {
        let feedback = make_feedback(vec![30, 30], vec![50.00, 55.00], vec![10.00, 15.00]);
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(1.0),  // epsilon
            None,       // ridge_variance
            Some(0.01), // min_prob
            Some(0.5),  // reg0
        )
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.2),  // epsilon
            None,       // ridge_variance
            Some(0.05), // min_prob
            Some(1.0),  // reg0
        )
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![
            0.049999991957649,
            0.373495754083455,
            0.049999994032051,
            0.526504259929341,
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.05), // epsilon
            None,       // ridge_variance
            Some(0.1),  // min_prob
            Some(0.2),  // reg0
        )
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(1.0),  // epsilon
            None,       // ridge_variance
            Some(0.05), // min_prob
            Some(0.5),  // reg0
        )
        .unwrap();

        // Expected solution from cvxpy (using CLARABEL solver)
        let expected = vec![
            0.049999999144349,
            0.342706084262522,
            0.049999999158130,
            0.417175997176582,
            0.140117920263625,
        ];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_asymmetric_high_variance_cvxpy() {
        let feedback = make_feedback(vec![50, 50], vec![100.00, 105.00], vec![1.00, 20.00]);
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(2.0),  // epsilon
            None,       // ridge_variance
            Some(0.01), // min_prob
            Some(0.0),  // reg0
        )
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.1),  // epsilon
            None,       // ridge_variance
            Some(0.05), // min_prob
            Some(0.1),  // reg0
        )
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.2),  // epsilon
            None,       // ridge_variance
            Some(0.05), // min_prob
            Some(0.2),  // reg0
        )
        .unwrap();

        // Expected solution from scipy (trust-constr with SLSQP fallback)
        let expected = vec![0.111715272862871, 0.473942231288105, 0.414342495849023];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_very_high_variance_scipy() {
        let feedback = make_feedback(vec![30, 30], vec![50.00, 55.00], vec![10.00, 15.00]);
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(1.0),  // epsilon
            None,       // ridge_variance
            Some(0.01), // min_prob
            Some(0.5),  // reg0
        )
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
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.3),  // epsilon
            None,       // ridge_variance
            Some(0.05), // min_prob
            Some(0.3),  // reg0
        )
        .unwrap();

        // Expected solution from scipy (trust-constr with SLSQP fallback)
        let expected = vec![
            0.050000000000000,
            0.394460771374844,
            0.064092683568538,
            0.491446545056619,
        ];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_small_epsilon_scipy() {
        let feedback = make_feedback(vec![50, 50], vec![10.00, 10.50], vec![1.00, 1.20]);
        let probs_map = estimate_optimal_probabilities(
            feedback,
            Some(0.05), // epsilon
            None,       // ridge_variance
            Some(0.1),  // min_prob
            Some(0.1),  // reg0
        )
        .unwrap();

        // Expected solution from scipy (trust-constr with SLSQP fallback)
        let expected = vec![0.477229489718064, 0.522770510281936];
        let probs = hashmap_to_vec(&probs_map, expected.len());
        assert_vecs_almost_equal(&probs, &expected, Some(1e-4));
        assert!((probs_map.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
