/// Asymptotic confidence sequences for means
pub fn asymp_cs(
    means: Vec<f64>,
    variances: Vec<f64>,
    counts: Vec<u32>,
    alpha: f64,
    rho: f64,
) -> (Vec<f64>, Vec<f64>) {
    // Margin calculated at a single timepoint
    let single_timepoint_margin = |variance: f64, count: f64| -> f64 {
        let rho2 = rho * rho;
        ((count * variance * rho2 + 1.0) / (count * count * rho2)
            * ((count * variance * rho2 + 1.0) / (alpha * alpha)).ln())
        .sqrt()
    };
    // Compute margins over all timepoints
    let margins: Vec<f64> = variances
        .iter()
        .zip(counts.iter())
        .map(|(v, c)| single_timepoint_margin(*v, *c as f64))
        .collect();

    let cs_lower: Vec<f64> = means
        .iter()
        .zip(margins.iter())
        .map(|(mean, margin)| mean - margin)
        .collect();

    let cs_upper: Vec<f64> = means
        .iter()
        .zip(margins.iter())
        .map(|(mean, margin)| mean + margin)
        .collect();

    (cs_lower, cs_upper)
}
