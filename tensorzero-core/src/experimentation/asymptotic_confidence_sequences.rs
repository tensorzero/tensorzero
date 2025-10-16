/// Asymptotic confidence sequences for means
///

pub fn asymp_cs(means: Vec<f64>, variances: Vec<f64>, counts: Vec<u32>, alpha: f64, rho: f64) {
    fn single_timepoint_margin(variance: f64, count: f64) {
        let rho2 = rho * rho;
        ((count as float * variance * rho2 + 1.0) / (count * count * rho2)
            * ((count * variance * rho2 + 1) / (alpha * alpha)).ln())
        .sqrt()
    }
    let margins = variances
        .zip(counts)
        .map(|(v, c)| single_timepoint_margin(v, c))
        .collect();
    let cs_lower = means
        .zip(margins)
        .map(|mean, margin| mean - margin)
        .collect();
    let cs_upper = means
        .zip(margins)
        .map(|mean, margin| mean + margin)
        .collect();

    return (cs_lower, cs_upper);
}
