#![allow(non_snake_case)]
// use argminmax::ArgMinMax;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SecondOrderConeT, SupportedConeT,
    ZeroConeT,
};
use thiserror::Error;
use typed_builder::TypedBuilder;

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
    #[error("{field} has a an out-of-range value at position {index}")]
    OutOfRange { field: String, index: usize },
    #[error("The `min_prob` value of {prob} is too large for the number of arms")]
    MinProbTooLarge { prob: f64 },
    #[error("Could not compute leader arm")]
    CouldntComputeArgmax,
    #[error("Failed to build Clarabel solver")]
    CouldntBuildSolver,
}

// impl Default for OptimalProbsArgs
#[derive(TypedBuilder)]
pub struct OptimalProbsArgs {
    // required arguments (no defaults)
    #[builder(setter(into))]
    pull_counts: Vec<usize>,
    #[builder(setter(into))]
    means: Vec<f64>,
    #[builder(setter(into))]
    variances: Vec<f64>,

    // optional arguments with defaults
    #[builder(default = 0.0)]
    epsilon: f64,
    #[builder(default = 1e-12)]
    ridge_variance: f64,
    #[builder(default = 1e-6)]
    min_prob: f64,
    #[builder(default = 1e-2)]
    reg0: f64,
}

/// Compute ε-aware optimal sampling proportions for sub-Gaussian rewards
pub fn estimate_optimal_probabilities(
    args: OptimalProbsArgs,
) -> Result<Vec<f64>, OptimalProbsError> {
    let OptimalProbsArgs {
        pull_counts,
        means,
        mut variances,
        epsilon,
        ridge_variance,
        min_prob,
        reg0,
    } = args;

    // Lower bound the variances at `ridge_variance`
    for v in &mut variances {
        *v = v.max(ridge_variance);
    }

    // Gather the quantities required for the optimization
    let num_arms = means.len();
    let num_decision_vars = 2 * num_arms + 1;
    let total_pulls: usize = pull_counts.iter().sum();
    let alpha_t: f64 = reg0 / (total_pulls as f64).sqrt(); // regularization coefficient
    let leader_arm = argmax(&means).ok_or(OptimalProbsError::CouldntComputeArgmax)?;
    let leader_mean = means[leader_arm];

    // ε-margins: gap_i = μ_L - μ_i + ε  (strictly > 0 for ε > 0)
    let gaps: Vec<f64> = means.iter().map(|&x| leader_mean - x + epsilon).collect();
    let gaps2: Vec<f64> = gaps.iter().map(|&x| (x * x).max(1e-16)).collect();

    // ---------- Objective: (1/2) x^T P x + q^T x ----------
    // Quadratic part: α_t ||w - u||^2 = α_t ||w||^2 - 2α_t u^T w + const
    // => P_ww = 2α I_K ; q_w = -2α_t u ; q_t = 1

    // Build q, the linear coefficients of the objective function
    let mut q = vec![0.0; num_decision_vars];
    q[2 * num_arms] = 1.0;
    if alpha_t > 0.0 {
        let u_val = 1.0 / (num_arms as f64);
        // for i in 0..num_arms {
        //     q[i] = -2.0 * alpha_t * u_val;
        // }
        for val in &mut q[..num_arms] {
            *val = -2.0 * alpha_t * u_val;
        }
    }
    // Build P, the quadratic coefficients, as diagonal on the w block: P_ii = 2α_t for i in [0..K-1]
    let mut P_dense = vec![vec![0.0; num_decision_vars]; num_decision_vars];
    if alpha_t > 0.0 {
        for (i, row) in P_dense[..num_arms].iter_mut().enumerate() {
            row[i] = 2.0 * alpha_t;
        }
    }
    // Convert dense matrix to compressed sparse column matrix expected by Clarabel
    let P = CscMatrix::from(&P_dense);

    // ---------- Constraints Ax + s = b, s ∈ K ----------
    let mut A_rows: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();

    // (A) Equality constraint: sum_i w_i = 1 (simplex equality) --> ZeroConeT(1)
    {
        let row = vec![1.0; num_decision_vars];
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
            row[i] = -min_prob;
            A_rows.push(row);
            b.push(0.0);
        }

        // Linear constraints for the slack variables for all arms besides leader
        for i in 0..num_arms {
            if i != leader_arm {
                let mut row: Vec<f64> = vec![0.0; num_decision_vars];
                row[num_arms + i] = 1.0;
                row[num_arms + leader_arm] = 1.0;
                row[2 * num_arms] = -2.0 * gaps2[i];
                A_rows.push(row);
                b.push(0.0);
            }
        }
        cones.push(NonnegativeConeT(2 * num_arms));
    }

    // (C) SOCP constraints (3 per arms) --> SecondOrderConeT
    {
        for i in 0..num_arms {
            let mut row1: Vec<f64> = vec![0.0; num_decision_vars];
            row1[i] = 1.0;
            row1[num_arms + i] = 0.5;
            A_rows.push(row1);
            b.push(0.0);

            let mut row2: Vec<f64> = vec![0.0; num_decision_vars];
            row2[i] = 1.0;
            row2[num_arms + i] = -0.5;
            A_rows.push(row2);
            b.push(0.0);

            let row3: Vec<f64> = vec![0.0; num_decision_vars];
            A_rows.push(row3);
            b.push(f64::sqrt(2.0) * variances[i]);
        }
        cones.push(SecondOrderConeT(3 * num_arms));
    }

    // Solve for the optimal weights
    let A = CscMatrix::from(&A_rows);
    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings)
        .map_err(|_| OptimalProbsError::CouldntBuildSolver)?;
    solver.solve();

    let x = solver.solution.x.as_slice();
    let w_star = x[0..num_arms].to_vec();

    Ok(w_star)
}
