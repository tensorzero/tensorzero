use crate::{
    db::{FeedbackTimeSeriesPoint, InternalFeedbackTimeSeriesPoint},
    error::{Error, ErrorDetails},
};

/// Asymptotic confidence sequences for the average conditional mean of
/// possibly time-varying means, under martingale dependence.
///
/// rho is a tuning parameter that determines the "intrinsic time".
/// The default value of 0.296 is based on...
pub fn asymp_cs(
    feedback: Vec<InternalFeedbackTimeSeriesPoint>,
    alpha: f32,
    rho: Option<f32>,
) -> Result<Vec<FeedbackTimeSeriesPoint>, Error> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("alpha must be in (0, 1), got {alpha}"),
        }));
    }

    // Use a reasonable default value of rho
    let rho = rho.unwrap_or(0.296);

    if rho <= 0.0 {
        return Err(Error::new(ErrorDetails::InvalidRequest {
            message: format!("rho must be strictly positive, got {rho}"),
        }));
    }
    let rho2 = rho * rho;
    let alpha2 = alpha * alpha;

    Ok(feedback
        .iter()
        .map(|f| {
            let count_f32 = f.count as f32;
            let cv_rho2 = count_f32 * f.variance * rho2;
            // Compute margin: sqrt(((n*v*rho^2 + 1) / (n^2 * rho^2)) * ln((n*v*rho^2 + 1) / alpha^2))
            let margin = ((cv_rho2 + 1.0) / (count_f32 * count_f32 * rho2)
                * ((cv_rho2 + 1.0) / alpha2).ln())
            .sqrt();
            FeedbackTimeSeriesPoint {
                period_end: f.period_end,
                variant_name: f.variant_name.clone(),
                mean: f.mean,
                variance: f.variance,
                count: f.count,
                alpha,
                cs_lower: f.mean - margin,
                cs_upper: f.mean + margin,
            }
        })
        .collect())
}
