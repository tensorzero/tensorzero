/// Asymptotic confidence sequences for the average conditional mean of
/// possibly time-varying means, under martingale dependence.
pub fn asymp_cs(
    means: Vec<f64>,
    variances: Vec<f64>,
    counts: Vec<u32>,
    alpha: f64,
    rho: f64,
) -> (Vec<f64>, Vec<f64>) {
    let rho2 = rho * rho;
    let alpha2 = alpha * alpha;

    // Compute lower and upper confidence bounds
    let (cs_lower, cs_upper): (Vec<f64>, Vec<f64>) = means
        .iter()
        .zip(variances.iter())
        .zip(counts.iter())
        .map(|((mean, variance), count)| {
            let count_f64 = *count as f64;
            // Factor out common term: count * variance * rho^2
            let cv_rho2 = count_f64 * variance * rho2;

            // Compute margin: sqrt(((n*v*rho^2 + 1) / (n^2 * rho^2)) * ln((n*v*rho^2 + 1) / alpha^2))
            let margin = ((cv_rho2 + 1.0) / (count_f64 * count_f64 * rho2)
                * ((cv_rho2 + 1.0) / alpha2).ln())
            .sqrt();

            (mean - margin, mean + margin)
        })
        .unzip();

    (cs_lower, cs_upper)
}
