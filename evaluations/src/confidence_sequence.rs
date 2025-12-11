use itertools;

const DEFAULT_M_RESOLUTION: usize = 1001;

pub struct WealthProcesses {
    pub m_values: Option<Vec<f64>>, // None = linspace(0, 1, resolution)
    pub resolution: Option<usize>,  // None = DEFAULT_M_RESOLUTION
    pub wealth_upper: Vec<f64>,     // K_t^+ at each m
    pub wealth_lower: Vec<f64>,     // K_t^- at each m
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

pub struct MeanBettingConfidenceSequence {
    pub name: String, // A variant or evaluator name
    pub mean_regularized: f64,
    pub variance_regularized: f64,
    pub count: u64,
    pub mean_est: f64,
    pub cs_lower: f64,
    pub cs_upper: f64,
    pub alpha: f32,
    pub wealth: WealthProcesses,
}

// combo_type: "max" or "convex"
// hedge weight for the process K_t+; for K_t- the hedge weight will be 1 - hedge_weight. Must be convex weights.
pub fn update_betting_cs(
    prev_results: MeanBettingConfidenceSequence,
    new_observations: Vec<f64>,
    combo_type: String,
    hedge_weight_upper: f32,
) -> MeanBettingConfidenceSequence {
    let n = new_observations.len();
    let prev_count = prev_results.count;

    // Times go from prev_count+1 to prev_count+n (inclusive)
    // These represent the "time" index for each new observation
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
        .map(|(&t, prev_variance)| {
            let num = 2.0 * (2.0 / prev_results.alpha as f64).ln();
            let denom = prev_variance * (t as f64) * ((t as f64) + 1.0).ln();
            (num / denom).sqrt()
        })
        .collect();

    // Update wealth processes for each candidate mean m
    let (new_wealth_upper, new_wealth_lower): (Vec<f64>, Vec<f64>) = prev_results
        .wealth
        .m_values_iter()
        .zip(prev_results.wealth.wealth_upper.iter())
        .zip(prev_results.wealth.wealth_lower.iter())
        .map(|((m, &prev_upper), &prev_lower)| {
            let (prod_upper, prod_lower) = bets.iter().zip(new_observations.iter()).fold(
                (1.0, 1.0),
                |(acc_upper, acc_lower), (&bet, &x)| {
                    (
                        acc_upper * (1.0 + bet * (x - m)),
                        acc_lower * (1.0 - bet * (x - m)),
                    )
                },
            );
            (prev_upper * prod_upper, prev_lower * prod_lower)
        })
        .unzip();

    // TODO: Compute new confidence bounds from wealth processes

    prev_results // Placeholder return
}
