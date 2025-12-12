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

use anyhow::{Result, bail};

const DEFAULT_M_RESOLUTION: usize = 1001;
const BET_TRUNCATION_LEVEL: f64 = 0.5;

/// Specifies the grid of candidate mean values where the wealth processes will be tracked.
///
/// Finer grids mean more precise (less conservative) confidence sequences.
#[derive(Debug, Clone)]
pub enum WealthProcessGridPoints {
    /// Custom grid of candidate mean values (for unequally spaced points).
    MValues(Vec<f64>),
    /// Number of equally spaced points in [0, 1].
    /// E.g., resolution 1001 gives grid [0.0, 0.001, ..., 0.999, 1.0].
    Resolution(usize),
}

impl Default for WealthProcessGridPoints {
    fn default() -> Self {
        Self::Resolution(DEFAULT_M_RESOLUTION)
    }
}

impl WealthProcessGridPoints {
    pub fn new_from_values(mvalues: Vec<f64>) -> Result<WealthProcessGridPoints> {
        let num_values = mvalues.len();
        if num_values < 2 {
            bail!("At least 2 grid points must be specified, got {num_values}")
        }
        Ok(WealthProcessGridPoints::MValues(mvalues))
    }

    pub fn new_from_resolution(resolution: usize) -> Result<WealthProcessGridPoints> {
        if resolution < 2 {
            bail!("Grid resolution must be at least 2, got {resolution}")
        }
        Ok(WealthProcessGridPoints::Resolution(resolution))
    }

    /// Returns the grid of candidate mean values.
    pub fn m_values(&self) -> Vec<f64> {
        match self {
            Self::MValues(v) => v.clone(),
            Self::Resolution(res) => (0..*res).map(|i| i as f64 / (*res - 1) as f64).collect(),
        }
    }

    /// Returns the number of grid points.
    #[expect(clippy::len_without_is_empty)] // is_empty() not meaningful; grids must have >= 2 points
    pub fn len(&self) -> usize {
        match self {
            Self::MValues(v) => v.len(),
            Self::Resolution(res) => *res,
        }
    }
}

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

/// Find the lower bound of the confidence set.
///
/// Searches in the range [0, min_idx] where wealth is monotonically decreasing.
/// Finds the first index where wealth drops below threshold,
/// then returns one index lower (outer bound) for conservative coverage.
///
/// Assumes: wealth_hedged[0] >= threshold (left endpoint is outside the confidence set)
///
/// Returns: (lower_bound_value, lower_bound_index)
fn find_cs_lower(
    wealth_hedged: &[f64],
    m_values: &[f64],
    threshold: f64,
    min_idx: usize,
) -> (f64, usize) {
    // partition_point returns the first index where the predicate is false
    let first_below = wealth_hedged[..=min_idx].partition_point(|w| *w >= threshold);
    let idx = first_below.saturating_sub(1);
    (m_values[idx], idx)
}

/// Find the upper bound of the confidence set.
///
/// Searches in the range [min_idx, n-1] where wealth is monotonically increasing.
/// Finds the last index where wealth is below threshold,
/// then returns one index higher (outer bound) for conservative coverage.
///
/// Assumes: wealth_hedged[n-1] >= threshold (right endpoint is outside the confidence set)
///
/// Returns: (upper_bound_value, upper_bound_index)
fn find_cs_upper(
    wealth_hedged: &[f64],
    m_values: &[f64],
    threshold: f64,
    min_idx: usize,
) -> (f64, usize) {
    let n = wealth_hedged.len();
    let first_above_relative = wealth_hedged[min_idx..].partition_point(|w| *w < threshold);
    let idx = (min_idx + first_above_relative).min(n - 1);
    (m_values[idx], idx)
}

/// Tracks the wealth processes used to construct confidence sequences.
///
/// The confidence set is defined as {m : wealth_hedged(m) < 1/α}, where
/// wealth_hedged is a combination of the upper and lower wealth processes.
/// These processes are evaluated on a grid of candidate mean values.
#[derive(Debug)]
pub struct WealthProcesses {
    /// Grid specification for candidate mean values.
    pub grid: WealthProcessGridPoints,
    /// Upper wealth process K_t^+(m) at each grid point.
    pub wealth_upper: Vec<f64>,
    /// Lower wealth process K_t^-(m) at each grid point.
    pub wealth_lower: Vec<f64>,
}

/// A confidence sequence for a bounded mean, constructed via betting.
///
/// This struct maintains the state needed to incrementally update a confidence
/// sequence as new observations arrive. The confidence interval [cs_lower, cs_upper]
/// is valid at any stopping time with coverage probability at least 1 - α.
#[derive(Debug)]
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
///
/// # Errors
/// Returns an error if:
/// - Any observation is outside [0, 1]
/// - `hedge_weight_upper` is outside [0, 1]
/// - The wealth process resolution is less than 2
pub fn update_betting_cs(
    prev_results: MeanBettingConfidenceSequence,
    new_observations: Vec<f64>,
    hedge_weight_upper: Option<f64>,
) -> Result<MeanBettingConfidenceSequence> {
    // Handle empty observations - return unchanged
    if new_observations.is_empty() {
        tracing::warn!("update_betting_cs called with empty observations, returning unchanged");
        return Ok(prev_results);
    }

    // Validate inputs
    if let Some(invalid) = new_observations
        .iter()
        .find(|&&x| !(0.0..=1.0).contains(&x))
    {
        bail!("All observations must be in [0, 1], got {invalid}");
    }
    if let Some(hw) = hedge_weight_upper
        && !(0.0..=1.0).contains(&hw)
    {
        bail!("hedge_weight_upper must be in [0, 1], got {hw}");
    }

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
    let m_values = prev_results.wealth.grid.m_values();
    let (new_wealth_upper, new_wealth_lower): (Vec<f64>, Vec<f64>) = m_values
        .iter()
        .zip(prev_results.wealth.wealth_upper.iter())
        .zip(prev_results.wealth.wealth_lower.iter())
        .map(|((&m, &prev_upper), &prev_lower)| {
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
    let n_grid = m_values.len();

    // Find the minimum of wealth_hedged - this is the point estimate (mean_est)
    // and is guaranteed to be inside the confidence set
    let min_idx = wealth_hedged
        .iter()
        .enumerate()
        .min_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mean_est = m_values[min_idx];

    // Determine confidence interval bounds based on endpoint behavior
    // The confidence set is {m : wealth_hedged(m) < threshold}
    let left_outside = wealth_hedged[0] >= threshold;
    let right_outside = wealth_hedged[n_grid - 1] >= threshold;

    let (cs_lower, _idx_lower) = if left_outside {
        // Binary search in [0, min_idx] to find lower bound
        find_cs_lower(&wealth_hedged, &m_values, threshold, min_idx)
    } else {
        // Left endpoint is inside confidence set
        (m_values[0], 0)
    };

    let (cs_upper, _idx_upper) = if right_outside {
        // Binary search in [min_idx, n-1] to find upper bound
        find_cs_upper(&wealth_hedged, &m_values, threshold, min_idx)
    } else {
        // Right endpoint is inside confidence set
        (m_values[n_grid - 1], n_grid - 1)
    };

    let variance_reg_final = *variances_reg
        .last()
        .unwrap_or(&prev_results.variance_regularized);

    let mean_reg_final = *means_reg.last().unwrap_or(&prev_results.mean_regularized);

    Ok(MeanBettingConfidenceSequence {
        name: prev_results.name,
        mean_regularized: mean_reg_final,
        variance_regularized: variance_reg_final,
        count: prev_count + n as u64,
        mean_est,
        cs_lower,
        cs_upper,
        alpha: prev_results.alpha,
        wealth: WealthProcesses {
            grid: prev_results.wealth.grid,
            wealth_upper: new_wealth_upper,
            wealth_lower: new_wealth_lower,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOAT_TOLERANCE: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < FLOAT_TOLERANCE
    }

    fn create_initial_cs(resolution: usize, alpha: f32) -> MeanBettingConfidenceSequence {
        MeanBettingConfidenceSequence {
            name: "test".to_string(),
            mean_regularized: 0.5,
            variance_regularized: 0.25,
            count: 0,
            mean_est: 0.5,
            cs_lower: 0.0,
            cs_upper: 1.0,
            alpha,
            wealth: WealthProcesses {
                grid: WealthProcessGridPoints::Resolution(resolution),
                wealth_upper: vec![1.0; resolution],
                wealth_lower: vec![1.0; resolution],
            },
        }
    }

    // Tests for grid construction validation

    #[test]
    fn test_resolution_one_returns_error() {
        let result = WealthProcessGridPoints::new_from_resolution(1);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Grid resolution must be at least 2")
        );
    }

    #[test]
    fn test_empty_m_values_returns_error() {
        let result = WealthProcessGridPoints::new_from_values(vec![]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("At least 2 grid points must be specified")
        );
    }

    #[test]
    fn test_single_m_value_returns_error() {
        let result = WealthProcessGridPoints::new_from_values(vec![0.5]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("At least 2 grid points must be specified")
        );
    }

    // Tests for find_cs_lower

    #[test]
    fn test_find_cs_lower_basic() {
        // Wealth: [100, 50, 10, 5, 10, 50, 100] - above-below-above pattern
        let wealth = vec![100.0, 50.0, 10.0, 5.0, 10.0, 50.0, 100.0];
        let m_values: Vec<f64> = (0..7).map(|i| i as f64 / 6.0).collect();
        let threshold = 20.0;
        let min_idx = 3; // index of minimum value (5.0)

        let (lower, idx) = find_cs_lower(&wealth, &m_values, threshold, min_idx);

        // First index where wealth < threshold is 2 (wealth=10)
        // Outer bound is index 1
        assert_eq!(idx, 1, "Lower bound index should be 1");
        assert!(
            (lower - m_values[1]).abs() < 1e-10,
            "Lower bound value should match m_values[1]"
        );
    }

    #[test]
    fn test_find_cs_lower_at_start() {
        // Wealth: [100, 5, 5, 5, 5, 5, 5] - drops below threshold at index 1
        let wealth = vec![100.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let m_values: Vec<f64> = (0..7).map(|i| i as f64 / 6.0).collect();
        let threshold = 20.0;
        let min_idx = 1; // smallest index of minimum value (5.0)

        let (_, idx) = find_cs_lower(&wealth, &m_values, threshold, min_idx);

        // First index where wealth < threshold is 1
        // Outer bound is index 0
        assert_eq!(idx, 0, "Lower bound index should be 0");
    }

    #[test]
    fn test_find_cs_lower_late_drop() {
        // Wealth: [100, 100, 100, 100, 5, 5, 5] - drops below threshold at index 4
        let wealth = vec![100.0, 100.0, 100.0, 100.0, 5.0, 5.0, 5.0];
        let m_values: Vec<f64> = (0..7).map(|i| i as f64 / 6.0).collect();
        let threshold = 20.0;
        let min_idx = 4; // index of minimum value (any index 4-6, use first)

        let (_, idx) = find_cs_lower(&wealth, &m_values, threshold, min_idx);

        // First index where wealth < threshold is 4
        // Outer bound is index 3
        assert_eq!(idx, 3, "Lower bound index should be 3");
    }

    // Tests for find_cs_upper

    #[test]
    fn test_find_cs_upper_basic() {
        // Wealth: [100, 50, 10, 5, 10, 50, 100] - above-below-above pattern
        let wealth = vec![100.0, 50.0, 10.0, 5.0, 10.0, 50.0, 100.0];
        let m_values: Vec<f64> = (0..7).map(|i| i as f64 / 6.0).collect();
        let threshold = 20.0;
        let min_idx = 3; // index of minimum value (5.0)

        let (upper, idx) = find_cs_upper(&wealth, &m_values, threshold, min_idx);

        // Last index where wealth < threshold is 4 (wealth=10)
        // Outer bound is index 5
        assert_eq!(idx, 5, "Upper bound index should be 5");
        assert!(
            (upper - m_values[5]).abs() < 1e-10,
            "Upper bound value should match m_values[5]"
        );
    }

    #[test]
    fn test_find_cs_upper_at_end() {
        // Wealth: [5, 5, 5, 5, 5, 5, 100] - rises above threshold only at end
        let wealth = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0];
        let m_values: Vec<f64> = (0..7).map(|i| i as f64 / 6.0).collect();
        let threshold = 20.0;
        let min_idx = 0; // index of minimum value (any index 0-5, use first)

        let (_, idx) = find_cs_upper(&wealth, &m_values, threshold, min_idx);

        // Last index where wealth < threshold is 5
        // Outer bound is index 6
        assert_eq!(idx, 6, "Upper bound index should be 6");
    }

    #[test]
    fn test_find_cs_upper_early_rise() {
        // Wealth: [5, 5, 5, 100, 100, 100, 100] - rises above threshold at index 3
        let wealth = vec![5.0, 5.0, 5.0, 100.0, 100.0, 100.0, 100.0];
        let m_values: Vec<f64> = (0..7).map(|i| i as f64 / 6.0).collect();
        let threshold = 20.0;
        let min_idx = 0; // index of minimum value (any index 0-2, use first)

        let (_, idx) = find_cs_upper(&wealth, &m_values, threshold, min_idx);

        // Last index where wealth < threshold is 2
        // Outer bound is index 3
        assert_eq!(idx, 3, "Upper bound index should be 3");
    }

    #[test]
    fn test_find_cs_bounds_consistent() {
        // Verify that lower <= upper for a valid confidence set
        let wealth = vec![100.0, 50.0, 10.0, 5.0, 10.0, 50.0, 100.0];
        let m_values: Vec<f64> = (0..7).map(|i| i as f64 / 6.0).collect();
        let threshold = 20.0;
        let min_idx = 3; // index of minimum value (5.0)

        let (_, idx_lower) = find_cs_lower(&wealth, &m_values, threshold, min_idx);
        let (_, idx_upper) = find_cs_upper(&wealth, &m_values, threshold, min_idx);

        assert!(
            idx_lower <= idx_upper,
            "Lower bound index {idx_lower} should be <= upper bound index {idx_upper}"
        );
    }

    // Tests for compute_bet

    #[test]
    fn test_compute_bet_basic() {
        let bet = compute_bet(1, 0.25, 0.05);
        assert!(bet > 0.0, "Bet should be positive");
        assert!(bet.is_finite(), "Bet should be finite");
    }

    #[test]
    fn test_compute_bet_decreases_with_time() {
        let bet_early = compute_bet(1, 0.25, 0.05);
        let bet_late = compute_bet(100, 0.25, 0.05);
        assert!(
            bet_early > bet_late,
            "Bets should decrease over long time horizons: early={bet_early}, late={bet_late}"
        );
    }

    #[test]
    fn test_compute_bet_decreases_with_variance() {
        let bet_low_var = compute_bet(10, 0.1, 0.05);
        let bet_high_var = compute_bet(10, 0.5, 0.05);
        assert!(
            bet_low_var > bet_high_var,
            "Bets should decrease with higher variance: low_var={bet_low_var}, high_var={bet_high_var}"
        );
    }

    #[test]
    fn test_compute_bet_decreases_with_alpha() {
        let low_alpha_bet = compute_bet(10, 0.25, 0.01);
        let high_alpha_bet = compute_bet(10, 0.25, 0.10);
        assert!(
            low_alpha_bet > high_alpha_bet,
            "Bets should decrease with higher alpha (less strict): low_alpha_bet={high_alpha_bet}, high_alpha_bet={low_alpha_bet}"
        );
    }

    // Tests for WealthProcessGridPoints

    #[test]
    fn test_grid_default_resolution() {
        let grid = WealthProcessGridPoints::default();
        assert_eq!(grid.len(), DEFAULT_M_RESOLUTION);
    }

    #[test]
    fn test_grid_custom_resolution() {
        let grid = WealthProcessGridPoints::Resolution(101);
        assert_eq!(grid.len(), 101);
    }

    #[test]
    fn test_grid_m_values_from_resolution() {
        let grid = WealthProcessGridPoints::Resolution(11);
        let m_values = grid.m_values();
        assert_eq!(m_values.len(), 11);
        assert!(approx_eq(m_values[0], 0.0));
        assert!(approx_eq(m_values[5], 0.5));
        assert!(approx_eq(m_values[10], 1.0));
    }

    #[test]
    fn test_grid_m_values_custom() {
        let custom_m = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let grid = WealthProcessGridPoints::MValues(custom_m.clone());
        let m_values = grid.m_values();
        assert_eq!(m_values, custom_m);
    }

    // Tests for update_betting_cs input validation

    #[test]
    fn test_empty_observations_returns_unchanged() {
        let initial = create_initial_cs(101, 0.05);
        // Save initial values before moving
        let initial_count = initial.count;
        let initial_mean_est = initial.mean_est;
        let initial_cs_lower = initial.cs_lower;
        let initial_cs_upper = initial.cs_upper;

        let updated = update_betting_cs(initial, vec![], None).unwrap();

        // Should return unchanged
        assert_eq!(updated.count, initial_count);
        assert_eq!(updated.mean_est, initial_mean_est);
        assert_eq!(updated.cs_lower, initial_cs_lower);
        assert_eq!(updated.cs_upper, initial_cs_upper);
    }

    #[test]
    fn test_observation_below_zero_returns_error() {
        let initial = create_initial_cs(101, 0.05);
        let result = update_betting_cs(initial, vec![0.5, -0.1, 0.6], None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("All observations must be in [0, 1]")
        );
    }

    #[test]
    fn test_observation_above_one_returns_error() {
        let initial = create_initial_cs(101, 0.05);
        let result = update_betting_cs(initial, vec![0.5, 1.1, 0.6], None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("All observations must be in [0, 1]")
        );
    }

    #[test]
    fn test_hedge_weight_below_zero_returns_error() {
        let initial = create_initial_cs(101, 0.05);
        let result = update_betting_cs(initial, vec![0.5], Some(-0.1));
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("hedge_weight_upper must be in [0, 1]")
        );
    }

    #[test]
    fn test_hedge_weight_above_one_returns_error() {
        let initial = create_initial_cs(101, 0.05);
        let result = update_betting_cs(initial, vec![0.5], Some(1.1));
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("hedge_weight_upper must be in [0, 1]")
        );
    }

    // Tests for update_betting_cs

    #[test]
    fn test_update_betting_cs_count_increments() {
        let initial = create_initial_cs(101, 0.05);
        let observations = vec![0.6, 0.7, 0.5];
        let updated = update_betting_cs(initial, observations, None).unwrap();
        assert_eq!(updated.count, 3);
    }

    #[test]
    fn test_update_betting_cs_multiple_updates() {
        let initial = create_initial_cs(101, 0.05);
        let updated1 = update_betting_cs(initial, vec![0.6, 0.7], None).unwrap();
        assert_eq!(updated1.count, 2);

        let updated2 = update_betting_cs(updated1, vec![0.5, 0.8, 0.6], None).unwrap();
        assert_eq!(updated2.count, 5);
    }

    #[test]
    fn test_update_betting_cs_bounds_valid() {
        let initial = create_initial_cs(101, 0.05);
        let observations = vec![0.6, 0.7, 0.5, 0.6, 0.7];
        let updated = update_betting_cs(initial, observations, None).unwrap();

        assert!(
            updated.cs_lower >= 0.0,
            "Lower bound should be >= 0: {}",
            updated.cs_lower
        );
        assert!(
            updated.cs_upper <= 1.0,
            "Upper bound should be <= 1: {}",
            updated.cs_upper
        );
        assert!(
            updated.cs_lower <= updated.cs_upper,
            "Lower bound should be <= upper bound: [{}, {}]",
            updated.cs_lower,
            updated.cs_upper
        );
    }

    #[test]
    fn test_update_betting_cs_mean_est_in_interval() {
        let initial = create_initial_cs(101, 0.05);
        let observations = vec![0.6, 0.7, 0.5, 0.6, 0.7, 0.65, 0.55];
        let updated = update_betting_cs(initial, observations, None).unwrap();

        assert!(
            updated.mean_est >= updated.cs_lower,
            "Mean estimate {} should be >= lower bound {}",
            updated.mean_est,
            updated.cs_lower
        );
        assert!(
            updated.mean_est <= updated.cs_upper,
            "Mean estimate {} should be <= upper bound {}",
            updated.mean_est,
            updated.cs_upper
        );
    }

    #[test]
    fn test_update_betting_cs_wealth_processes_updated() {
        let initial = create_initial_cs(11, 0.05);
        let observations = vec![0.6];
        let updated = update_betting_cs(initial, observations, None).unwrap();

        // Wealth processes should no longer all be 1.0 after an update
        let all_ones_upper = updated
            .wealth
            .wealth_upper
            .iter()
            .all(|&w| approx_eq(w, 1.0));
        let all_ones_lower = updated
            .wealth
            .wealth_lower
            .iter()
            .all(|&w| approx_eq(w, 1.0));
        assert!(
            !all_ones_upper || !all_ones_lower,
            "Wealth processes should be updated after observations"
        );
    }

    #[test]
    fn test_update_betting_cs_preserves_resolution() {
        let initial = create_initial_cs(51, 0.05);
        let observations = vec![0.5, 0.6];
        let updated = update_betting_cs(initial, observations, None).unwrap();

        assert_eq!(updated.wealth.grid.len(), 51);
        assert_eq!(updated.wealth.wealth_upper.len(), 51);
        assert_eq!(updated.wealth.wealth_lower.len(), 51);
    }

    #[test]
    fn test_update_betting_cs_preserves_alpha() {
        let initial = create_initial_cs(101, 0.10);
        let observations = vec![0.5, 0.6];
        let updated = update_betting_cs(initial, observations, None).unwrap();

        // Use exact f32 comparison since alpha is stored as f32
        assert_eq!(updated.alpha, 0.10f32, "Alpha should be preserved");
    }

    #[test]
    fn test_update_betting_cs_hedge_weight_default() {
        let initial1 = create_initial_cs(101, 0.05);
        let initial2 = create_initial_cs(101, 0.05);
        let observations = vec![0.6, 0.7, 0.5];

        let updated_default = update_betting_cs(initial1, observations.clone(), None).unwrap();
        let updated_explicit = update_betting_cs(initial2, observations, Some(0.5)).unwrap();

        assert!(
            approx_eq(updated_default.mean_est, updated_explicit.mean_est),
            "Default hedge weight should be 0.5"
        );
    }

    #[test]
    fn test_update_betting_cs_different_hedge_weights() {
        let initial1 = create_initial_cs(101, 0.05);
        let initial2 = create_initial_cs(101, 0.05);
        let observations = vec![0.9, 0.85, 0.8, 0.75, 0.7];

        let updated_low = update_betting_cs(initial1, observations.clone(), Some(0.3)).unwrap();
        let updated_high = update_betting_cs(initial2, observations, Some(0.7)).unwrap();

        // wealth_upper and wealth_lower are independent of hedge_weight,
        // but the hedged combination affects cs_lower, cs_upper, and mean_est
        assert_eq!(
            updated_low.wealth.wealth_upper, updated_high.wealth.wealth_upper,
            "wealth_upper should be the same regardless of hedge weight"
        );
        assert_eq!(
            updated_low.wealth.wealth_lower, updated_high.wealth.wealth_lower,
            "wealth_lower should be the same regardless of hedge weight"
        );

        // With data not centered around 0.5, the confidence interval bounds should
        // differ due to hedge weight
        let low_width = updated_low.cs_upper - updated_low.cs_lower;
        let high_width = updated_high.cs_upper - updated_high.cs_lower;
        assert!(
            !approx_eq(low_width, high_width)
                || !approx_eq(updated_low.mean_est, updated_high.mean_est),
            "Different hedge weights should affect confidence interval or mean estimate"
        );
    }

    // Tests for interval tightening behavior

    #[test]
    fn test_interval_tightens_with_more_observations() {
        let initial = create_initial_cs(101, 0.05);

        // Add observations centered around 0.5 with some variance
        let obs_batch1 = vec![0.4, 0.6, 0.45, 0.55, 0.5, 0.48, 0.52, 0.47, 0.53, 0.5];
        let updated1 = update_betting_cs(initial, obs_batch1, None).unwrap();
        let width1 = updated1.cs_upper - updated1.cs_lower;

        // Add more observations centered around 0.5
        let obs_batch2 = vec![
            0.42, 0.58, 0.46, 0.54, 0.5, 0.49, 0.51, 0.47, 0.53, 0.5, 0.44, 0.56, 0.48, 0.52, 0.5,
            0.45, 0.55, 0.49, 0.51, 0.5, 0.43, 0.57, 0.47, 0.53, 0.5, 0.46, 0.54, 0.48, 0.52, 0.5,
            0.41, 0.59, 0.45, 0.55, 0.5, 0.44, 0.56, 0.49, 0.51, 0.5, 0.42, 0.58, 0.46, 0.54, 0.5,
            0.47, 0.53, 0.48, 0.52, 0.5,
        ];
        let updated2 = update_betting_cs(updated1, obs_batch2, None).unwrap();
        let width2 = updated2.cs_upper - updated2.cs_lower;

        assert!(
            width2 < width1,
            "Interval should tighten with more observations: width after 10={width1}, width after 60={width2}"
        );
    }

    #[test]
    fn test_mean_est_converges_to_true_mean() {
        let initial = create_initial_cs(101, 0.05);
        let true_mean = 0.7;

        // Generate 100 observations with mean 0.7 and some variance
        // Values oscillate around 0.7: 0.6, 0.8, 0.65, 0.75, 0.7, repeated 20 times
        let observations: Vec<f64> = (0..20)
            .flat_map(|_| vec![0.6, 0.8, 0.65, 0.75, 0.7])
            .collect();
        let updated = update_betting_cs(initial, observations, None).unwrap();

        let error = (updated.mean_est - true_mean).abs();
        assert!(
            error < 0.02,
            "Mean estimate {:.3} should be close to true mean {true_mean}: error={error}",
            updated.mean_est
        );
    }

    // Tests for edge cases

    #[test]
    fn test_update_betting_cs_all_zeros() {
        let initial = create_initial_cs(101, 0.05);
        let observations = vec![0.0; 10];
        let updated = update_betting_cs(initial, observations, None).unwrap();

        assert!(updated.cs_lower >= 0.0);
        assert!(updated.cs_upper <= 1.0);
        assert!(
            updated.mean_est <= 0.5,
            "Mean estimate should be low for all-zero observations"
        );
    }

    #[test]
    fn test_update_betting_cs_all_ones() {
        let initial = create_initial_cs(101, 0.05);
        let observations = vec![1.0; 10];
        let updated = update_betting_cs(initial, observations, None).unwrap();

        assert!(updated.cs_lower >= 0.0);
        assert!(updated.cs_upper <= 1.0);
        assert!(
            updated.mean_est >= 0.5,
            "Mean estimate should be high for all-one observations"
        );
    }

    #[test]
    fn test_update_betting_cs_single_observation() {
        let initial = create_initial_cs(101, 0.05);
        let observations = vec![0.75];
        let updated = update_betting_cs(initial, observations, None).unwrap();

        assert_eq!(updated.count, 1);
        assert!(updated.cs_lower >= 0.0);
        assert!(updated.cs_upper <= 1.0);
    }

    #[test]
    fn test_update_betting_cs_extreme_alpha() {
        // Very strict alpha
        let initial_strict = create_initial_cs(101, 0.001);
        let observations = vec![0.5; 10];
        let updated_strict = update_betting_cs(initial_strict, observations.clone(), None).unwrap();

        // Very loose alpha
        let initial_loose = create_initial_cs(101, 0.5);
        let updated_loose = update_betting_cs(initial_loose, observations, None).unwrap();

        let width_strict = updated_strict.cs_upper - updated_strict.cs_lower;
        let width_loose = updated_loose.cs_upper - updated_loose.cs_lower;

        assert!(
            width_strict >= width_loose,
            "Stricter alpha should give wider intervals: strict={width_strict}, loose={width_loose}"
        );
    }

    // Tests for numerical stability

    #[test]
    fn test_wealth_processes_non_negative() {
        let initial = create_initial_cs(101, 0.05);
        let observations = vec![0.1, 0.9, 0.0, 1.0, 0.5];
        let updated = update_betting_cs(initial, observations, None).unwrap();

        for (i, &w) in updated.wealth.wealth_upper.iter().enumerate() {
            assert!(
                w >= 0.0,
                "Wealth upper at index {i} should be non-negative: {w}"
            );
        }
        for (i, &w) in updated.wealth.wealth_lower.iter().enumerate() {
            assert!(
                w >= 0.0,
                "Wealth lower at index {i} should be non-negative: {w}"
            );
        }
    }

    #[test]
    fn test_wealth_processes_finite() {
        let initial = create_initial_cs(101, 0.05);
        let observations = vec![0.5; 50];
        let updated = update_betting_cs(initial, observations, None).unwrap();

        for (i, &w) in updated.wealth.wealth_upper.iter().enumerate() {
            assert!(
                w.is_finite(),
                "Wealth upper at index {i} should be finite: {w}"
            );
        }
        for (i, &w) in updated.wealth.wealth_lower.iter().enumerate() {
            assert!(
                w.is_finite(),
                "Wealth lower at index {i} should be finite: {w}"
            );
        }
    }

    #[test]
    fn test_many_observations_stability() {
        let initial = create_initial_cs(101, 0.05);
        // Generate 500 observations with mean 0.5 and variance
        // Pattern: 0.3, 0.7, 0.4, 0.6, 0.5 repeated 100 times
        let observations: Vec<f64> = (0..100)
            .flat_map(|_| vec![0.3, 0.7, 0.4, 0.6, 0.5])
            .collect();
        let updated = update_betting_cs(initial, observations, None).unwrap();

        assert!(updated.cs_lower.is_finite());
        assert!(updated.cs_upper.is_finite());
        assert!(updated.mean_est.is_finite());
        assert!(updated.cs_lower <= updated.cs_upper);
    }

    #[test]
    fn test_incremental_vs_batch_updates() {
        // Verify that processing observations incrementally gives the same result as batch
        let observations = vec![0.3, 0.7, 0.5, 0.6, 0.4];

        // Batch processing
        let initial_batch = create_initial_cs(101, 0.05);
        let batch_result = update_betting_cs(initial_batch, observations.clone(), None).unwrap();

        // Incremental processing
        let mut incremental = create_initial_cs(101, 0.05);
        for obs in observations {
            incremental = update_betting_cs(incremental, vec![obs], None).unwrap();
        }

        // Results should match
        assert_eq!(batch_result.count, incremental.count);
        assert!(
            (batch_result.variance_regularized - incremental.variance_regularized).abs() < 1e-10,
            "Variance should match: batch={}, incremental={}",
            batch_result.variance_regularized,
            incremental.variance_regularized
        );

        // Wealth processes should match
        for (i, (batch_w, incr_w)) in batch_result
            .wealth
            .wealth_upper
            .iter()
            .zip(incremental.wealth.wealth_upper.iter())
            .enumerate()
        {
            assert!(
                (batch_w - incr_w).abs() < 1e-10,
                "Wealth upper at index {i} should match: batch={batch_w}, incremental={incr_w}"
            );
        }
        for (i, (batch_w, incr_w)) in batch_result
            .wealth
            .wealth_lower
            .iter()
            .zip(incremental.wealth.wealth_lower.iter())
            .enumerate()
        {
            assert!(
                (batch_w - incr_w).abs() < 1e-10,
                "Wealth lower at index {i} should match: batch={batch_w}, incremental={incr_w}"
            );
        }
    }

    // ==================== Regression tests with known values ====================
    // These tests use pre-computed values to detect regressions in the algorithm.
    // If these fail after a code change, verify the change is intentional.

    #[test]
    fn test_known_values_compute_bet() {
        // bet = sqrt(2 * ln(2/alpha) / (prev_variance * t * ln(t+1)))
        // At t=1, prev_variance=0.25, alpha=0.05:
        // bet = sqrt(2 * ln(40) / (0.25 * 1 * ln(2))) = 6.524984655851606
        let bet = compute_bet(1, 0.25, 0.05);
        assert!(
            (bet - 6.524984655851606).abs() < 1e-10,
            "Bet at t=1 should be 6.524984655851606, got {bet}"
        );

        // At t=2, prev_variance=0.12625, alpha=0.05:
        // bet = sqrt(2 * ln(40) / (0.12625 * 2 * ln(3))) = 5.157144640499054
        let bet2 = compute_bet(2, 0.12625, 0.05);
        assert!(
            (bet2 - 5.157144640499054).abs() < 1e-10,
            "Bet at t=2 should be 5.157144640499054, got {bet2}"
        );
    }

    #[test]
    fn test_known_values_single_observation_at_half() {
        // Single observation x=0.5 with alpha=0.05
        // At m=0.5: wealth_upper = 1 + bet*(0.5-0.5) = 1.0
        //           wealth_lower = 1 - bet*(0.5-0.5) = 1.0
        let initial = create_initial_cs(101, 0.05);
        let updated = update_betting_cs(initial, vec![0.5], None).unwrap();

        // Find the index closest to m=0.5
        let m_values = updated.wealth.grid.m_values();
        let idx_half = m_values
            .iter()
            .position(|&m| (m - 0.5).abs() < 1e-10)
            .expect("Should have m=0.5 in grid");

        assert!(
            (updated.wealth.wealth_upper[idx_half] - 1.0).abs() < 1e-10,
            "Wealth upper at m=0.5 should be 1.0, got {}",
            updated.wealth.wealth_upper[idx_half]
        );
        assert!(
            (updated.wealth.wealth_lower[idx_half] - 1.0).abs() < 1e-10,
            "Wealth lower at m=0.5 should be 1.0, got {}",
            updated.wealth.wealth_lower[idx_half]
        );
    }

    #[test]
    fn test_known_values_single_observation_wealth_at_endpoints() {
        // Single observation x=0.5 with alpha=0.05, bet=6.524984655851606
        // At m=0.0:
        //   bet_upper_max = 0.5/0 = inf (no truncation), bet_upper = 6.52
        //   bet_lower_max = 0.5/(1-0) = 0.5, bet_lower = 0.5 (truncated)
        //   wealth_upper = 1 + 6.52*(0.5-0) = 4.262492
        //   wealth_lower = 1 - 0.5*(0.5-0) = 0.75
        // At m=1.0:
        //   bet_upper_max = 0.5/1 = 0.5, bet_upper = 0.5 (truncated)
        //   bet_lower_max = 0.5/(1-1) = inf (no truncation), bet_lower = 6.52
        //   wealth_upper = 1 + 0.5*(0.5-1) = 0.75
        //   wealth_lower = 1 - 6.52*(0.5-1) = 4.262492
        let initial = create_initial_cs(101, 0.05);
        let updated = update_betting_cs(initial, vec![0.5], None).unwrap();

        // First element is m=0.0
        // Use 1e-6 tolerance due to log-sum-exp numerical differences
        assert!(
            (updated.wealth.wealth_upper[0] - 4.262492327925803).abs() < 1e-6,
            "Wealth upper at m=0.0 should be ~4.262492, got {}",
            updated.wealth.wealth_upper[0]
        );
        assert!(
            (updated.wealth.wealth_lower[0] - 0.75).abs() < 1e-10,
            "Wealth lower at m=0.0 should be 0.75, got {}",
            updated.wealth.wealth_lower[0]
        );

        // Last element is m=1.0
        let last = updated.wealth.wealth_upper.len() - 1;
        assert!(
            (updated.wealth.wealth_upper[last] - 0.75).abs() < 1e-10,
            "Wealth upper at m=1.0 should be 0.75, got {}",
            updated.wealth.wealth_upper[last]
        );
        assert!(
            (updated.wealth.wealth_lower[last] - 4.262492327925803).abs() < 1e-6,
            "Wealth lower at m=1.0 should be ~4.262492, got {}",
            updated.wealth.wealth_lower[last]
        );
    }

    #[test]
    fn test_known_values_two_observations_mean_and_variance() {
        // Two observations [0.6, 0.4] with alpha=0.05
        // After both: mean_reg = (0.5 + 0.6 + 0.4) / 3 = 0.5
        //             var_reg = (0.25 + (0.6-0.55)^2 + (0.4-0.5)^2) / 3 = 0.0875
        let initial = create_initial_cs(101, 0.05);
        let updated = update_betting_cs(initial, vec![0.6, 0.4], None).unwrap();

        assert_eq!(updated.count, 2);

        // Note: mean_regularized is set to mean_est (minimizer of wealth_hedged)
        // which may differ from the regularized mean. Check variance instead.
        assert!(
            (updated.variance_regularized - 0.0875).abs() < 1e-10,
            "Variance regularized should be 0.0875, got {}",
            updated.variance_regularized
        );
    }

    #[test]
    fn test_known_values_two_observations_wealth_at_half() {
        // Two observations [0.6, 0.4] with alpha=0.05
        // At m=0.5, bet truncation is 0.5/0.5 = 1.0 for both upper and lower
        // Raw bets (6.52, 5.16) get truncated to 1.0
        // factor_upper for x=0.6: 1 + 1*(0.6-0.5) = 1.1
        // factor_lower for x=0.6: 1 - 1*(0.6-0.5) = 0.9
        // factor_upper for x=0.4: 1 + 1*(0.4-0.5) = 0.9
        // factor_lower for x=0.4: 1 - 1*(0.4-0.5) = 1.1
        // wealth_upper = 1.0 * 1.1 * 0.9 = 0.99
        // wealth_lower = 1.0 * 0.9 * 1.1 = 0.99
        let initial = create_initial_cs(101, 0.05);
        let updated = update_betting_cs(initial, vec![0.6, 0.4], None).unwrap();

        let m_values = updated.wealth.grid.m_values();
        let idx_half = m_values
            .iter()
            .position(|&m| (m - 0.5).abs() < 1e-10)
            .expect("Should have m=0.5 in grid");

        assert!(
            (updated.wealth.wealth_upper[idx_half] - 0.99).abs() < 1e-10,
            "Wealth upper at m=0.5 after [0.6, 0.4] should be 0.99, got {}",
            updated.wealth.wealth_upper[idx_half]
        );
        assert!(
            (updated.wealth.wealth_lower[idx_half] - 0.99).abs() < 1e-10,
            "Wealth lower at m=0.5 after [0.6, 0.4] should be 0.99, got {}",
            updated.wealth.wealth_lower[idx_half]
        );
    }

    #[test]
    fn test_known_values_wealth_increases_away_from_true_mean() {
        // When true mean is 0.7, wealth should be minimized near m=0.7
        // and increase as m moves away from the true mean
        let initial = create_initial_cs(101, 0.05);
        let observations = vec![0.7; 20]; // 20 observations at true mean 0.7
        let updated = update_betting_cs(initial, observations, None).unwrap();

        let m_values = updated.wealth.grid.m_values();
        let n = m_values.len();

        // Find index closest to 0.7
        let idx_70 = m_values
            .iter()
            .position(|&m| (m - 0.7).abs() < 0.01)
            .expect("Should have m near 0.7");

        // Wealth at m=0.0 should be much larger than at m=0.7
        assert!(
            updated.wealth.wealth_upper[0] > updated.wealth.wealth_upper[idx_70] * 10.0,
            "Wealth at m=0.0 ({}) should be much larger than at m=0.7 ({})",
            updated.wealth.wealth_upper[0],
            updated.wealth.wealth_upper[idx_70]
        );

        // Wealth at m=1.0 should be larger than at m=0.7 (but not as extreme as m=0.0)
        assert!(
            updated.wealth.wealth_lower[n - 1] > updated.wealth.wealth_lower[idx_70],
            "Wealth at m=1.0 ({}) should be larger than at m=0.7 ({})",
            updated.wealth.wealth_lower[n - 1],
            updated.wealth.wealth_lower[idx_70]
        );

        // Mean estimate should be close to 0.7
        assert!(
            (updated.mean_est - 0.7).abs() < 0.05,
            "Mean estimate should be close to 0.7, got {}",
            updated.mean_est
        );
    }

    #[test]
    fn test_known_values_full_confidence_sequence() {
        // 10 varied observations averaging to 0.7, alpha=0.05, resolution=101
        // Pre-computed values:
        //   cs_lower = 0.36
        //   cs_upper = 0.86
        //   mean_est = 0.70
        //   variance_regularized ≈ 0.0318
        //   mean_regularized ≈ 0.6818
        let initial = create_initial_cs(101, 0.05);
        let observations = vec![0.5, 0.9, 0.6, 0.8, 0.7, 0.75, 0.65, 0.72, 0.68, 0.7];
        let updated = update_betting_cs(initial, observations, None).unwrap();

        // Check count
        assert_eq!(updated.count, 10);

        // Check mean_est (should be 0.7, the grid point closest to sample mean)
        assert!(
            (updated.mean_est - 0.7).abs() < 1e-10,
            "mean_est should be 0.7, got {}",
            updated.mean_est
        );

        // Check regularized mean: (0.5*1 + 7.0) / 11 ≈ 0.6818
        let expected_mean_reg = (0.5 + 7.0) / 11.0;
        assert!(
            (updated.mean_regularized - expected_mean_reg).abs() < 1e-10,
            "mean_regularized should be {expected_mean_reg}, got {}",
            updated.mean_regularized
        );

        // Check variance_regularized ≈ 0.0318
        assert!(
            (updated.variance_regularized - 0.031827712868268).abs() < 1e-10,
            "variance_regularized should be ~0.0318, got {}",
            updated.variance_regularized
        );

        // Check confidence interval bounds
        // cs_lower = 0.36 (with resolution=101, m_values are 0.00, 0.01, ..., 1.00)
        assert!(
            (updated.cs_lower - 0.36).abs() < 1e-10,
            "cs_lower should be 0.36, got {}",
            updated.cs_lower
        );

        // cs_upper = 0.86
        assert!(
            (updated.cs_upper - 0.86).abs() < 1e-10,
            "cs_upper should be 0.86, got {}",
            updated.cs_upper
        );

        // Verify mean_est is within the confidence interval
        assert!(
            updated.cs_lower <= updated.mean_est && updated.mean_est <= updated.cs_upper,
            "mean_est {} should be within [{}, {}]",
            updated.mean_est,
            updated.cs_lower,
            updated.cs_upper
        );

        // Verify the true mean (0.7) is within the confidence interval
        assert!(
            updated.cs_lower <= 0.7 && 0.7 <= updated.cs_upper,
            "True mean 0.7 should be within [{}, {}]",
            updated.cs_lower,
            updated.cs_upper
        );
    }
}
