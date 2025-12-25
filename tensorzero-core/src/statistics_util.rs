//! Statistical utility functions.

/// Computes the mean of a slice of data.
pub fn mean(data: &[f32]) -> Option<f32> {
    let sum = data.iter().sum::<f32>();
    let count = data.len();

    if count == 0 {
        return None;
    }

    Some(sum / count as f32)
}

/// Computes the standard deviation of a slice of data.
pub fn std_deviation(data: &[f32]) -> Option<f32> {
    let count = data.len();
    if count == 0 {
        return None;
    }
    let data_mean = mean(data)?;

    let variance = data
        .iter()
        .map(|value| {
            let diff = data_mean - (*value);
            diff * diff
        })
        .sum::<f32>()
        / count as f32;

    Some(variance.sqrt())
}

/// Computes the Wald confidence interval for continuous data.
/// Uses the formula: mean ± 1.96 * (stddev / sqrt(n))
/// Returns (lower, upper) bounds, or (None, None) if insufficient data.
pub fn wald_confint(mean: f64, stdev: f64, count: u32) -> Option<(f64, f64)> {
    if count == 0 {
        return None;
    }

    let count_f64 = count as f64;
    let margin = 1.96 * (stdev / count_f64.sqrt());
    Some((mean - margin, mean + margin))
}

/// Computes the 95% Wilson score confidence interval for Bernoulli data.
///
/// This function takes precomputed mean (proportion) and count, and returns
/// the lower and upper bounds of the confidence interval.
///
/// The Wilson score interval is preferred over the Wald interval for proportions
/// because it handles extreme cases (p close to 0 or 1) better and doesn't produce
/// bounds outside [0, 1].
///
/// # Arguments
/// * `mean` - The proportion/mean of successes (should be in [0, 1] for Bernoulli data)
/// * `count` - The number of observations
///
/// # Returns
/// * `Some((lower, upper))` if count > 0
/// * `None` if count is 0
pub fn wilson_confint(mean: f64, count: u32) -> Option<(f64, f64)> {
    if count == 0 {
        return None;
    }

    let count_f64 = count as f64;
    let z: f64 = 1.96; // Standard normal quantile for 95% CI
    let z_squared: f64 = z.powi(2);
    let scale: f64 = 1.0 / (1.0 + z_squared / count_f64);
    let center: f64 = mean + z_squared / (2.0 * count_f64);
    let margin = z / (2.0 * count_f64) * (4.0 * count_f64 * mean * (1.0 - mean) + z_squared).sqrt();
    let ci_lower = (center - margin) * scale;
    let ci_upper = (center + margin) * scale;

    // Clamp to [0, 1] to handle floating-point precision errors
    let ci_lower = ci_lower.clamp(0.0, 1.0);
    let ci_upper = ci_upper.clamp(0.0, 1.0);

    Some((ci_lower, ci_upper))
}

/// Computes the 95% Wilson score confidence interval from raw Bernoulli data.
///
/// This is a convenience function that computes the mean from the data and
/// then calls `wilson_confint`.
///
/// # Arguments
/// * `data` - Slice of values (typically 0.0 or 1.0 for Bernoulli data)
///
/// # Returns
/// * `Some((lower, upper))` if data is non-empty
/// * `None` if data is empty
pub fn wilson_confint_from_data(data: &[f32]) -> Option<(f64, f64)> {
    let count = data.len();
    if count == 0 {
        return None;
    }

    let sum: f32 = data.iter().sum();
    let mean = (sum / count as f32) as f64;

    wilson_confint(mean, count as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for wald confidence interval
    #[test]
    fn test_wald_confint_basic() {
        // Test basic Wald CI computation
        // mean = 100, stdev = 10, count = 100
        // margin = 1.96 * (10 / sqrt(100)) = 1.96
        let (lower, upper) =
            wald_confint(100.0, 10.0, 100).expect("Should produce valid confidence interval");
        // 100 - 1.96 = 98.04, 100 + 1.96 = 101.96
        assert!((lower - 98.04).abs() < 0.01);
        assert!((upper - 101.96).abs() < 0.01);
    }

    #[test]
    fn test_wald_confint_zero_count() {
        // Test with zero count
        let confint = wald_confint(100.0, 10.0, 0);
        assert!(
            confint.is_none(),
            "Zero count should produce None as confidence interval"
        );
    }

    // Tests for wilson confidence interval

    #[test]
    fn test_wilson_confint_basic() {
        // Test Wilson CI for p = 0.5, n = 100
        // This should give approximately (0.40, 0.60)
        let result = wilson_confint(0.5, 100);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();
        // Wilson CI for p=0.5, n=100 should be approximately (0.40, 0.60)
        assert!(lower > 0.39 && lower < 0.41, "lower = {lower}");
        assert!(upper > 0.59 && upper < 0.61, "upper = {upper}");
    }

    #[test]
    fn test_wilson_confint_extreme_high() {
        // Test Wilson CI for p = 1.0, n = 1 (extreme case)
        let result = wilson_confint(1.0, 1);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();
        // Lower should be around 0.206
        assert!(
            (lower - 0.20654329147389294).abs() < 0.0001,
            "lower = {lower}"
        );
        assert!((upper - 1.0).abs() < 0.0001, "upper = {upper}");
    }

    #[test]
    fn test_wilson_confint_zero_count() {
        let result = wilson_confint(0.5, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_wilson_confint_known_values() {
        // Test against known values: p = 0.4489795918367347, n = 49
        let result = wilson_confint(0.4489795918367347, 49);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();
        assert!(
            (lower - 0.31852624929636336).abs() < 0.0001,
            "lower = {lower}"
        );
        assert!(
            (upper - 0.5868513320032188).abs() < 0.0001,
            "upper = {upper}"
        );
    }

    // Tests for wilson_confint_from_data (takes raw data)

    #[test]
    fn test_wilson_confint_from_data_all_zeros() {
        // All 0s should still produce non-zero width interval
        let data = vec![0.0; 10];
        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        // Should be bounded in [0, 1]
        assert!(lower == 0.0, "Lower bound should be = 0, got {lower}");
        assert!(upper < 1.0, "Upper bound should be < 1, got {upper}");

        // Should have non-zero width
        assert!(
            upper > lower,
            "Interval should have non-zero width, got [{lower}, {upper}]"
        );
        assert!(
            upper > 0.0,
            "Upper bound should be > 0 even with all zeros, got {upper}"
        );
    }

    #[test]
    fn test_wilson_confint_from_data_all_ones() {
        // All 1s should still produce non-zero width interval
        let data = vec![1.0; 10];
        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        // Should be bounded in [0, 1]
        assert!(lower > 0.0, "Lower bound should be > 0, got {lower}");
        assert!(upper == 1.0, "Upper bound should be = 1, got {upper}");

        // Should have non-zero width
        assert!(
            upper > lower,
            "Interval should have non-zero width, got [{lower}, {upper}]"
        );
        assert!(
            lower < 1.0,
            "Lower bound should be < 1 even with all ones, got {lower}"
        );
    }

    #[test]
    fn test_wilson_confint_from_data_half_half() {
        // 50/50 split should have symmetric interval around 0.5
        let data = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        // Should be bounded in [0, 1]
        assert!(lower > 0.0, "Lower bound should be > 0, got {lower}");
        assert!(upper < 1.0, "Upper bound should be < 1, got {upper}");

        // Should be roughly symmetric around 0.5
        let midpoint = (lower + upper) / 2.0;
        assert!(
            (midpoint - 0.5).abs() < 0.01,
            "Midpoint should be close to 0.5, got {midpoint}"
        );
    }

    #[test]
    fn test_wilson_confint_from_data_single_value() {
        // Single observation should still work
        let data = vec![1.0];
        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        // Should be bounded in [0, 1]
        assert!(lower >= 0.0);
        assert!(upper <= 1.0);
        assert!(upper > lower);
    }

    #[test]
    fn test_wilson_confint_from_data_empty() {
        // Empty data should return None
        let data: Vec<f32> = vec![];
        let result = wilson_confint_from_data(&data);
        assert!(result.is_none());
    }

    #[test]
    fn test_wilson_confint_from_data_known_values() {
        // Test against known values
        let data = vec![0.0; 10];
        let mut test_data = data;
        test_data.extend(vec![1.0; 10]);

        let result = wilson_confint_from_data(&test_data);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        assert!(
            (lower - 0.2992949).abs() < 1e-6,
            "Lower bound should be approximately 0.2992949, got {lower}"
        );
        assert!(
            (upper - 0.7007050).abs() < 1e-6,
            "Upper bound should be approximately 0.7007050, got {upper}"
        );
    }

    #[test]
    fn test_wilson_vs_wald_large_sample() {
        // With large samples, Wilson and Wald intervals should converge
        // Using n=10000, p=0.4 (4000 successes)
        let mut data = vec![0.0; 6000];
        data.extend(vec![1.0; 4000]);

        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (wilson_lower, wilson_upper) = result.unwrap();

        // Compute Wald interval: p̂ ± 1.96 * sqrt(p̂(1-p̂)/n)
        let p_hat: f64 = 0.4;
        let n: f64 = 10000.0;
        let wald_stderr: f64 = (p_hat * (1.0 - p_hat) / n).sqrt();
        let wald_margin: f64 = 1.96 * wald_stderr;
        let wald_lower: f64 = p_hat - wald_margin;
        let wald_upper: f64 = p_hat + wald_margin;

        // Wilson and Wald should be very close for large n
        assert!(
            (wilson_lower - wald_lower).abs() < 1e-4,
            "Wilson lower ({wilson_lower}) should be close to Wald lower ({wald_lower})"
        );
        assert!(
            (wilson_upper - wald_upper).abs() < 1e-4,
            "Wilson upper ({wilson_upper}) should be close to Wald upper ({wald_upper})"
        );

        // Verify both are well within [0, 1]
        assert!(wilson_lower > 0.0 && wilson_lower < 1.0);
        assert!(wilson_upper > 0.0 && wilson_upper < 1.0);
        assert!(wald_lower > 0.0 && wald_lower < 1.0);
        assert!(wald_upper > 0.0 && wald_upper < 1.0);
    }

    #[test]
    fn test_wilson_confint_from_data_versus_wald_extreme() {
        // Wilson should handle extreme proportions better than Wald
        // With p=0.05 (1 success in 20), Wald can give lower < 0
        let mut data = vec![0.0; 19];
        data.push(1.0);

        let result = wilson_confint_from_data(&data);
        assert!(result.is_some());
        let (lower, _upper) = result.unwrap();

        // Wilson should keep bounds within [0, 1]
        assert!(
            lower >= 0.0,
            "Wilson lower bound should be >= 0, got {lower}",
        );
    }
}
