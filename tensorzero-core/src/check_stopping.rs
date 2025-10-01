use crate::estimate_optimal_probabilities::argmax;

pub struct PairwiseGLRArgs {
    leader_pulls: usize,
    leader_mean: f64,
    leader_variance: f64,
    challenger_pulls: usize,
    challenger_mean: f64,
    challenger_variance: f64,
    epsilon: f64,
}

pub enum PairwiseGLRError {
    GLRError,
}

pub struct CheckStoppingArgs {
    pull_counts: Vec<usize>,
    means: Vec<f64>,
    variances: Vec<f64>,
    min_pulls: usize,
    ridge_variance: f64,
    epsilon: f64,
    delta: f64,
}

pub enum CheckStoppingError {
    GLRError,
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

///     Decide whether to stop the experiment and which arm to recommend.
///     Uses parallel GLR testing with uniform challenger weights π_j = 1/(K-1),
///     and per-pair time t_{L,j} = n_L + n_j.
pub fn check_stopping(args: CheckStoppingArgs) -> Result<(bool, usize), CheckStoppingError> {
    let CheckStoppingArgs {
        pull_counts,
        means,
        mut variances,
        min_pulls,
        ridge_variance,
        epsilon,
        delta,
    } = args;
    // TODO: return struct instead of tuple
    // TODO: how to validate inputs?
    // TODO: Should probably return false inside of throwing error
    if pull_counts.iter().any(|&x| x < min_pulls) {
        return Err(CheckStoppingError::StoppingError);
    }

    // Lower bound variances by `ridge_variance` for numerical stability
    for v in &mut variances {
        *v = v.max(ridge_variance);
    }

    // TODO: implement tie-breaking for leader arm
    // TODO: in case of tie, choose arm with largest value of variance / pull_count, for both stopping condition and return value. Log a warning
    let num_arms: usize = pull_counts.len();
    let leader_arm: usize = argmax(&means).ok_or(CheckStoppingError::StoppingError)?;

    // Compute likelihood statistic for all non-leader arms
    let mut glr_vals: Vec<f64> = vec![];

    for challenger_arm in 0..num_arms {
        if challenger_arm == leader_arm {
            continue;
        }

        let leader_pulls: usize = pull_counts[leader_arm];
        let leader_mean: f64 = means[leader_arm];
        let leader_variance: f64 = variances[leader_arm];
        let challenger_pulls: usize = pull_counts[challenger_arm];
        let challenger_mean: f64 = means[leader_arm];
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
    let total_pulls: usize = pull_counts.iter().sum();
    let threshold: f64 = ((1.0 + (total_pulls as f64).ln()) / delta).ln();

    // Stopping condition check
    let glr_min = glr_vals.iter().copied().reduce(f64::min);
    match glr_min {
        Some(min_val) => {
            if min_val > threshold {
                Ok((true, leader_arm))
            } else {
                Ok((false, leader_arm))
            }
        }
        None => Err(CheckStoppingError::StoppingError),
    }
}
