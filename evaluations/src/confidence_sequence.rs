//! Betting-based confidence sequences for bounded means.
//!
//! This module implements confidence sequences using the betting martingale framework
//! from Waudby-Smith & Ramdas (2024), "Estimating means of bounded random variables
//! by betting" (JRSS-B). These provide time-uniform confidence sequences, which consist
//! of confidence intervals that are valid at any stopping time. This makes them suitable
//! for sequential experimentation and continuous monitoring.
//!
//! The key idea is to construct wealth processes that grow when the true mean differs
//! from a candidate value m. Each wealth process can be understood as the wealth
//! accumulated by a gambler making bets about where the true mean lies relative to m.
//! The confidence set at time t is the set of all m values for which the wealth process
//! has not exceeded the threshold 1/α, where α is the confidence level.

use itertools;

const DEFAULT_M_RESOLUTION: usize = 1001;
const BET_TRUNCATION_LEVEL: f64 = 0.5;

/// Compute the predictable-plugin-Bernstein-type bet for a given time step.
///
/// * `t` - The current time step (1-indexed)
/// * `prev_variance` - The variance estimate from the previous time step.
///   Bets must be predictable (measurable on the previous sigma-algebra),
///   so can only use information from the previous time step.
/// * `alpha` - The significance level (e.g., 0.05 for 95% confidence)
fn compute_bet(t: u64, prev_variance: f64, alpha: f64) -> f64 {
    let num = 2.0 * (2.0 / alpha).ln();
    let denom = prev_variance * (t as f64) * ((t as f64) + 1.0).ln();
    (num / denom).sqrt()
}

/// Tracks the wealth processes used to construct confidence sequences.
///
/// The confidence set is defined as {m : wealth_hedged(m) < 1/α}, where
/// wealth_hedged is a combination of the upper and lower wealth processes.
/// These processes are evaluated on a grid of candidate mean values.
pub struct WealthProcesses {
    /// Grid of candidate mean values. If None, defaults to linspace(0, 1, resolution).
    pub m_values: Option<Vec<f64>>,
    /// Number of grid points. If None, defaults to DEFAULT_M_RESOLUTION (1001).
    pub resolution: Option<usize>,
    /// Upper wealth process K_t^+(m) at each grid point.
    pub wealth_upper: Vec<f64>,
    /// Lower wealth process K_t^-(m) at each grid point.
    pub wealth_lower: Vec<f64>,
}

impl WealthProcesses {
    fn resolution(&self) -> usize {
        self.resolution.unwrap_or(DEFAULT_M_RESOLUTION)
    }

    pub fn m_values_iter(&self) -> impl Iterator<Item = f64> + '_ {
        let resolution = self.resolution();
        self.m_values
            .as_ref()
            .map(|v| itertools::Either::Left(v.iter().copied()))
            .unwrap_or_else(|| {
                itertools::Either::Right(
                    (0..resolution).map(move |i| i as f64 / (resolution - 1) as f64),
                )
            })
    }

    pub fn m_values(&self) -> Vec<f64> {
        self.m_values_iter().collect()
    }
}

/// A confidence sequence for a bounded mean, constructed via betting.
///
/// This struct maintains the state needed to incrementally update a confidence
/// sequence as new observations arrive. The confidence interval [cs_lower, cs_upper]
/// is valid at any stopping time with coverage probability at least 1 - α.
pub struct MeanBettingConfidenceSequence {
    /// Identifier for this sequence (e.g., variant or evaluator name).
    pub name: String,
    /// Regularized running mean, used for variance estimation.
    pub mean_regularized: f64,
    /// Regularized running variance, used for computing bets.
    pub variance_regularized: f64,
    /// Number of observations processed so far.
    pub count: u64,
    /// Point estimate of the mean (minimizer of the hedged wealth process).
    pub mean_est: f64,
    /// Lower bound of the confidence interval.
    pub cs_lower: f64,
    /// Upper bound of the confidence interval.
    pub cs_upper: f64,
    /// Significance level (e.g., 0.05 for 95% confidence).
    pub alpha: f32,
    /// Underlying wealth processes used to construct the confidence set.
    pub wealth: WealthProcesses,
}

/// Update a confidence sequence with new observations.
///
/// Processes a batch of new observations and returns an updated confidence sequence.
/// The update involves:
/// 1. Computing regularized running mean and variance estimates
/// 2. Computing predictable bets based on previous variance estimates
/// 3. Updating the upper and lower wealth processes for each candidate mean
/// 4. Combining the wealth processes into a single hedged process and finding the new confidence bounds
///
/// This function uses a hedged capital process that takes the maximum of the weighted
/// upper and lower wealth processes. This combination is guaranteed to produce confidence
/// sets that are intervals, which is not necessarily true for other combination types
/// (e.g., convex/sum combinations can produce non-convex confidence sets).
///
/// # Arguments
/// * `prev_results` - The current state of the confidence sequence
/// * `new_observations` - New observations to incorporate (must be in [0, 1])
/// * `hedge_weight_upper` - Weight for the upper wealth process (must be in [0, 1]).
///   The lower process receives weight (1 - hedge_weight_upper). If None, defaults to 0.5.
///
/// # Returns
/// Updated `MeanBettingConfidenceSequence` with new bounds and wealth process values.
pub fn update_betting_cs(
    prev_results: MeanBettingConfidenceSequence,
    new_observations: Vec<f64>,
    hedge_weight_upper: Option<f64>,
) -> MeanBettingConfidenceSequence {
    let hedge_weight_upper = hedge_weight_upper.unwrap_or(0.5);
    let n = new_observations.len();
    let prev_count = prev_results.count;

    // Times go from prev_count+1 to prev_count+n (inclusive)
    // These represent the time index for each new observation
    let times: Vec<u64> = ((prev_count + 1)..=(prev_count + n as u64)).collect();

    // Cumulative sum of new observations
    let cum_sums: Vec<f64> = new_observations
        .iter()
        .scan(0.0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect();

    // Regularized means: for each time t, compute the updated regularized mean
    // means_reg[i] = (prev_mean_reg * (prev_count + 1) + cum_sums[i]) / (t + 1)
    let means_reg: Vec<f64> = times
        .iter()
        .zip(cum_sums.iter())
        .map(|(&t, &cum_sum)| {
            let numerator = prev_results.mean_regularized * (prev_count + 1) as f64 + cum_sum;
            let denominator = (t + 1) as f64;
            numerator / denominator
        })
        .collect();

    // Cumulative sum of squared deviations of new observations from the regularized means
    let cum_sum_squares: Vec<f64> = new_observations
        .iter()
        .zip(means_reg.iter())
        .scan(0.0, |acc, (&obs, &mean)| {
            let deviation = obs - mean;
            *acc += deviation * deviation;
            Some(*acc)
        })
        .collect();

    // Regularized variances: for each time t, compute the updated regularized variance
    let variances_reg: Vec<f64> = times
        .iter()
        .zip(cum_sum_squares.iter())
        .map(|(&t, &cum_sq)| {
            let numerator = prev_results.variance_regularized * (prev_count + 1) as f64 + cum_sq;
            let denominator = (t + 1) as f64;
            numerator / denominator
        })
        .collect();

    // Compute predictable-plugin-Bernstein-type bets for each time step
    // Bets must be predictable (F_{t-1}-measurable), so use the previous variance estimate
    let lagged_variances = std::iter::once(prev_results.variance_regularized)
        .chain(variances_reg.iter().take(n - 1).copied());
    let bets: Vec<f64> = times
        .iter()
        .zip(lagged_variances)
        .map(|(&t, prev_variance)| compute_bet(t, prev_variance, prev_results.alpha as f64))
        .collect();

    // Update upper and lower wealth processes for each candidate mean m
    // Use log-sum-exp for numerical stability: prod(1 + bet*(x-m)) = exp(sum(log(1 + bet*(x-m))))
    // Truncate bets to c/m (upper) or c/(1-m) (lower) to ensure wealth processes stay non-negative
    let (new_wealth_upper, new_wealth_lower): (Vec<f64>, Vec<f64>) = prev_results
        .wealth
        .m_values_iter()
        .zip(prev_results.wealth.wealth_upper.iter())
        .zip(prev_results.wealth.wealth_lower.iter())
        .map(|((m, &prev_upper), &prev_lower)| {
            let bet_upper_max = if m == 0.0 {
                f64::INFINITY
            } else {
                BET_TRUNCATION_LEVEL / m
            };
            let bet_lower_max = if m == 1.0 {
                f64::INFINITY
            } else {
                BET_TRUNCATION_LEVEL / (1.0 - m)
            };
            let (log_prod_upper, log_prod_lower) = bets.iter().zip(new_observations.iter()).fold(
                (0.0, 0.0),
                |(acc_upper, acc_lower), (&bet, &x)| {
                    let bet_upper = bet.min(bet_upper_max);
                    let bet_lower = bet.min(bet_lower_max);
                    (
                        acc_upper + (1.0 + bet_upper * (x - m)).ln(),
                        acc_lower + (1.0 - bet_lower * (x - m)).ln(),
                    )
                },
            );
            (
                prev_upper * log_prod_upper.exp(),
                prev_lower * log_prod_lower.exp(),
            )
        })
        .unzip();

    // Compute hedged wealth process as the max of the weighted upper and lower wealth processes
    let threshold = 1.0 / prev_results.alpha as f64;
    let wealth_hedged: Vec<f64> = new_wealth_upper
        .iter()
        .zip(new_wealth_lower.iter())
        .map(|(&w_upper, &w_lower)| {
            let weighted_upper = hedge_weight_upper * w_upper;
            let weighted_lower = (1.0 - hedge_weight_upper) * w_lower;
            weighted_upper.max(weighted_lower)
        })
        .collect();

    // Find confidence bounds using binary search
    // The confidence set is {m : wealth_hedged(m) < 1/alpha}
    // We assume this forms an interval and find its boundaries
    let m_values = prev_results.wealth.m_values();
    let n_grid = m_values.len();

    // Binary search for lower bound: find smallest m where wealth_hedged < threshold
    // Take the outer (smaller) grid point for coverage
    let (cs_lower, idx_lower) = {
        let mut lo = 0;
        let mut hi = n_grid;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if wealth_hedged[mid] >= threshold {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is the first index where wealth_hedged < threshold
        // Take the outer point (one index lower) for coverage, but clamp to valid range
        if lo > 0 {
            (m_values[lo - 1], lo - 1)
        } else {
            (m_values[0], 0)
        }
    };

    // Binary search for upper bound: find largest m where wealth_hedged < threshold
    // Take the outer (larger) grid point for coverage
    let (cs_upper, idx_upper) = {
        let mut lo = 0;
        let mut hi = n_grid;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if wealth_hedged[mid] < threshold {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is the first index where wealth_hedged >= threshold (from the right side of interval)
        // Take the outer point (lo itself) for coverage, but clamp to valid range
        if lo < n_grid {
            (m_values[lo], lo)
        } else {
            (m_values[n_grid - 1], n_grid - 1)
        }
    };

    // Compute the final mean estimate as the minimizer of wealth_hedged
    // Search only within the confidence interval since the minimum must be there
    let mean_est = m_values[idx_lower..=idx_upper]
        .iter()
        .zip(wealth_hedged[idx_lower..=idx_upper].iter())
        .min_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(&m, _)| m)
        .unwrap_or(prev_results.mean_est);

    let variance_reg_final = *variances_reg
        .last()
        .unwrap_or(&prev_results.variance_regularized);

    MeanBettingConfidenceSequence {
        name: prev_results.name,
        mean_regularized: mean_est,
        variance_regularized: variance_reg_final,
        count: prev_count + n as u64,
        mean_est,
        cs_lower,
        cs_upper,
        alpha: prev_results.alpha,
        wealth: WealthProcesses {
            m_values: prev_results.wealth.m_values,
            resolution: prev_results.wealth.resolution,
            wealth_upper: new_wealth_upper,
            wealth_lower: new_wealth_lower,
        },
    }
}
