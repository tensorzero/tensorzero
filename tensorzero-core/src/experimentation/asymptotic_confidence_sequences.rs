use crate::{
    db::feedback::{CumulativeFeedbackTimeSeriesPoint, InternalCumulativeFeedbackTimeSeriesPoint},
    error::{Error, ErrorDetails},
};

/// Computes asymptotic confidence sequences for the running average conditional mean.
///
/// This function computes an asymptotic confidence sequence for the running average
/// conditional mean that is valid under martingale dependence. If the data are i.i.d.,
/// then this returns an asymptotic confidence sequence for the mean. See the referencee
/// below for the definition of an asymptotic confidence sequence and the expression that
/// this function implements (Proposition 2.5).
///
/// # Arguments
///
/// * `feedback` - A vector of time series points, each containing cumulative statistics
///   (mean, variance, count), where count is equivalent to time.
/// * `alpha` - The significance level, in (0, 1). The confidence sequence
///   will have coverage probability 1 - alpha. For example, alpha = 0.05 gives 95% confidence.
/// * `rho` - Optional nonnegative tuning parameter that determines the "intrinsic time" scale. Controls
///   the time point at which the confidence sequence is tightest, relatively speaking. Can be
///   selected to make the confidence sequence is relatively tight at a given time point, if
///   the user anticipates checking the confidence sequence (e.g. for hypothesis testing
///   purposes) starting at or around that time point. If `None`, a value is computed which is
///   approximately optimal in the i.i.d. setting for times starting at 100. The confidence
///   sequence is not highly sensitive to rho, so it's fine to leave it unspecified.
///
/// # Returns
///
/// Returns a `Result` containing a vector of `CumulativeFeedbackTimeSeriesPoint` with confidence
/// sequence bounds (`cs_lower`, `cs_upper`) added to each point. The bounds are symmetric
/// around the mean, with margin computed as:
///
/// ```text
/// margin = sqrt(((n*v*rho^2 + 1) / (n^2 * rho^2)) * ln((n*v*rho^2 + 1) / alpha^2))
/// ```
///
/// where n is the count, v is the variance, and rho is the tuning parameter.
///
/// # Errors
///
/// Returns an error if:
/// * `alpha` is not in the open interval (0, 1)
/// * `rho` is not strictly positive (if provided)
///
/// # References
///
/// Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E. H., & Ramdas, A. (2024).
/// Time-uniform central limit theory and asymptotic confidence sequences.
/// *The Annals of Statistics*, 52(6), 2380-2407.
/// DOI: [10.1214/24-AOS2408](https://doi.org/10.1214/24-AOS2408)
///
/// # Example
///
/// ```ignore
/// let feedback = vec![
///     InternalCumulativeFeedbackTimeSeriesPoint {
///         period_end: chrono::Utc::now(),
///         variant_name: "control".to_string(),
///         mean: 0.5,
///         variance: 0.25,
///         count: 100,
///     },
/// ];
/// let result = asymp_cs(feedback, 0.05, None)?;
/// // result[0].cs_lower and result[0].cs_upper contain 95% confidence bounds
/// ```
pub fn asymp_cs(
    feedback: Vec<InternalCumulativeFeedbackTimeSeriesPoint>,
    alpha: f32,
    rho: Option<f32>,
) -> Result<Vec<CumulativeFeedbackTimeSeriesPoint>, Error> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("alpha must be in (0, 1), got {alpha}"),
        }));
    }

    // Default value of rho, computed as sqrt( (-2 log(alpha) + log(-2 log(alpha)) + 1) / 100 )
    let rho =
        rho.unwrap_or_else(|| (-2.0 * alpha.ln() + (-2.0 * alpha.ln()).ln() + 1.0).sqrt() / 10.0);

    if rho <= 0.0 {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("rho must be strictly positive, got {rho}"),
        }));
    }
    let rho2 = rho * rho;
    let alpha2 = alpha * alpha;

    Ok(feedback
        .into_iter()
        .map(|f| {
            // If variance is None, we can't compute confidence sequences
            let (cs_lower, cs_upper) = match (f.mean, f.variance) {
                (mean, Some(variance)) => {
                    let count_f32 = f.count as f32;
                    let cv_rho2 = count_f32 * variance * rho2;
                    // Compute margin: sqrt(((n*v*rho^2 + 1) / (n^2 * rho^2)) * ln((n*v*rho^2 + 1) / alpha^2))
                    let margin = ((cv_rho2 + 1.0) / (count_f32 * count_f32 * rho2)
                        * ((cv_rho2 + 1.0) / alpha2).ln())
                    .sqrt();
                    (Some(mean - margin), Some(mean + margin))
                }
                _ => (None, None),
            };

            CumulativeFeedbackTimeSeriesPoint {
                period_end: f.period_end,
                variant_name: f.variant_name,
                mean: f.mean,
                variance: f.variance,
                count: f.count,
                alpha,
                cs_lower,
                cs_upper,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_point(
        mean: f32,
        variance: f32,
        count: u64,
    ) -> InternalCumulativeFeedbackTimeSeriesPoint {
        InternalCumulativeFeedbackTimeSeriesPoint {
            period_end: Utc::now(),
            variant_name: "test_variant".to_string(),
            mean,
            variance: Some(variance),
            count,
        }
    }

    #[test]
    fn test_basic_functionality() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].mean, 0.5);
        assert_eq!(result[0].variance, Some(0.25));
        assert_eq!(result[0].count, 100);
        assert_eq!(result[0].alpha, 0.05);
        assert!(result[0].cs_lower.unwrap() < result[0].mean);
        assert!(result[0].cs_upper.unwrap() > result[0].mean);
    }

    #[test]
    fn test_empty_input() {
        let feedback = vec![];
        let result = asymp_cs(feedback, 0.05, None).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_single_data_point() {
        let feedback = vec![create_test_point(0.8, 0.16, 50)];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        assert_eq!(result.len(), 1);
        // Verify bounds are symmetric
        let mean = result[0].mean;
        let cs_lower = result[0].cs_lower.unwrap();
        let cs_upper = result[0].cs_upper.unwrap();
        let margin = cs_upper - mean;
        assert!((mean - cs_lower - margin).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_data_points() {
        let feedback = vec![
            create_test_point(0.3, 0.21, 10),
            create_test_point(0.5, 0.25, 50),
            create_test_point(0.7, 0.21, 100),
        ];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        assert_eq!(result.len(), 3);
        // Verify all points have confidence sequences
        for point in result {
            assert!(point.cs_lower.unwrap() < point.mean);
            assert!(point.cs_upper.unwrap() > point.mean);
            assert_eq!(point.alpha, 0.05);
        }
    }

    #[test]
    fn test_confidence_sequences_narrow_with_more_data() {
        let feedback = vec![
            create_test_point(0.5, 0.25, 10),
            create_test_point(0.5, 0.25, 100),
            create_test_point(0.5, 0.25, 1000),
        ];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        let width_10 = result[0].cs_upper.unwrap() - result[0].cs_lower.unwrap();
        let width_100 = result[1].cs_upper.unwrap() - result[1].cs_lower.unwrap();
        let width_1000 = result[2].cs_upper.unwrap() - result[2].cs_lower.unwrap();

        // Widths should decrease with more data
        assert!(width_10 > width_100);
        assert!(width_100 > width_1000);
    }

    #[test]
    fn test_higher_variance_wider_intervals() {
        let feedback = vec![
            create_test_point(0.5, 0.1, 100),
            create_test_point(0.5, 0.25, 100),
            create_test_point(0.5, 0.5, 100),
        ];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        let width_low_var = result[0].cs_upper.unwrap() - result[0].cs_lower.unwrap();
        let width_med_var = result[1].cs_upper.unwrap() - result[1].cs_lower.unwrap();
        let width_high_var = result[2].cs_upper.unwrap() - result[2].cs_lower.unwrap();

        // Widths should increase with variance
        assert!(width_low_var < width_med_var);
        assert!(width_med_var < width_high_var);
    }

    #[test]
    fn test_smaller_alpha_wider_intervals() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];

        let result_95 = asymp_cs(feedback.clone(), 0.05, None).unwrap();
        let result_99 = asymp_cs(feedback, 0.01, None).unwrap();

        let width_95 = result_95[0].cs_upper.unwrap() - result_95[0].cs_lower.unwrap();
        let width_99 = result_99[0].cs_upper.unwrap() - result_99[0].cs_lower.unwrap();

        // 99% confidence should be wider than 95%
        assert!(width_99 > width_95);
    }

    #[test]
    fn test_custom_rho() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];

        let result_default = asymp_cs(feedback.clone(), 0.05, None).unwrap();
        let result_custom = asymp_cs(feedback, 0.05, Some(0.5)).unwrap();

        // Different rho values should give different bounds
        assert_ne!(
            result_default[0].cs_lower.unwrap(),
            result_custom[0].cs_lower.unwrap()
        );
        assert_ne!(
            result_default[0].cs_upper.unwrap(),
            result_custom[0].cs_upper.unwrap()
        );
    }

    #[test]
    fn test_bounds_symmetry() {
        let feedback = vec![
            create_test_point(0.2, 0.16, 50),
            create_test_point(0.5, 0.25, 100),
            create_test_point(0.8, 0.16, 150),
        ];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        // Verify bounds are symmetric around the mean
        for point in result {
            let mean = point.mean;
            let cs_lower = point.cs_lower.unwrap();
            let cs_upper = point.cs_upper.unwrap();
            let lower_margin = mean - cs_lower;
            let upper_margin = cs_upper - mean;
            assert!((lower_margin - upper_margin).abs() < 1e-5);
        }
    }

    #[test]
    fn test_zero_variance() {
        let feedback = vec![create_test_point(0.5, 0.0, 100)];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        // Should still produce valid (narrow) bounds
        assert!(result[0].cs_lower.unwrap() < result[0].mean);
        assert!(result[0].cs_upper.unwrap() > result[0].mean);
        // With zero variance, bounds should be very tight
        let width = result[0].cs_upper.unwrap() - result[0].cs_lower.unwrap();
        assert!(width < 0.2);
    }

    #[test]
    fn test_preserves_input_fields() {
        let feedback = vec![InternalCumulativeFeedbackTimeSeriesPoint {
            period_end: Utc::now(),
            variant_name: "variant_a".to_string(),
            mean: 0.42,
            variance: Some(0.24),
            count: 123,
        }];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        assert_eq!(result[0].variant_name, "variant_a");
        assert_eq!(result[0].mean, 0.42);
        assert_eq!(result[0].variance, Some(0.24));
        assert_eq!(result[0].count, 123);
    }

    // Error cases
    #[test]
    fn test_alpha_zero() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 0.0, None);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("alpha must be in (0, 1)"));
    }

    #[test]
    fn test_alpha_one() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 1.0, None);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("alpha must be in (0, 1)"));
    }

    #[test]
    fn test_alpha_negative() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, -0.1, None);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("alpha must be in (0, 1)"));
    }

    #[test]
    fn test_alpha_greater_than_one() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 1.5, None);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("alpha must be in (0, 1)"));
    }

    #[test]
    fn test_rho_zero() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 0.05, Some(0.0));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("rho must be strictly positive"));
    }

    #[test]
    fn test_rho_negative() {
        let feedback = vec![create_test_point(0.5, 0.25, 100)];
        let result = asymp_cs(feedback, 0.05, Some(-0.5));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("rho must be strictly positive"));
    }

    #[test]
    fn test_margin_calculation_correctness() {
        // Test the actual margin calculation with known values
        let mean = 0.5_f32;
        let variance = 0.25_f32;
        let count = 100_u64;
        let alpha = 0.05_f32;
        let rho = 0.296_f32;

        let feedback = vec![create_test_point(mean, variance, count)];
        let result = asymp_cs(feedback, alpha, Some(rho)).unwrap();

        // Manually compute expected margin
        let count_f32 = count as f32;
        let rho2 = rho * rho;
        let alpha2 = alpha * alpha;
        let cv_rho2 = count_f32 * variance * rho2;
        let expected_margin = ((cv_rho2 + 1.0) / (count_f32 * count_f32 * rho2)
            * ((cv_rho2 + 1.0) / alpha2).ln())
        .sqrt();

        let actual_margin = result[0].cs_upper.unwrap() - result[0].mean;
        assert!((actual_margin - expected_margin).abs() < 1e-5);
    }

    #[test]
    fn test_large_count() {
        let feedback = vec![create_test_point(0.5, 0.25, 1_000_000)];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        // With very large count, bounds should be very tight
        let width = result[0].cs_upper.unwrap() - result[0].cs_lower.unwrap();
        assert!(width < 0.01);
    }

    #[test]
    fn test_small_count() {
        let feedback = vec![create_test_point(0.5, 0.25, 2)];
        let result = asymp_cs(feedback, 0.05, None).unwrap();

        // With small count, bounds should be wide
        let width = result[0].cs_upper.unwrap() - result[0].cs_lower.unwrap();
        assert!(width > 0.1);
    }
}
