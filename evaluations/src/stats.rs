use std::{collections::HashMap, io::Write};

use anyhow::Result;
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_core::client::InferenceResponse;
use tensorzero_core::{endpoints::datasets::Datapoint, evaluations::EvaluatorConfig};
use tracing::{debug, info, instrument};
use uuid::Uuid;

use crate::{evaluators, OutputFormat};

pub struct EvaluationStats {
    pub output_format: OutputFormat,
    pub evaluation_infos: Vec<EvaluationInfo>,
    pub evaluation_errors: Vec<EvaluationError>,
    pub progress_bar: Option<ProgressBar>,
}

impl EvaluationStats {
    pub fn new(output_format: OutputFormat, dataset_len: usize) -> Self {
        let progress_bar = match output_format {
            OutputFormat::Jsonl => None,
            OutputFormat::Pretty => Some(ProgressBar::new(dataset_len as u64)),
        };
        debug!(
            output_format = ?output_format,
            dataset_len = dataset_len,
            "Initialized evaluation stats tracker"
        );
        Self {
            output_format,
            evaluation_infos: Vec::new(),
            evaluation_errors: Vec::new(),
            progress_bar,
        }
    }

    pub fn push(
        &mut self,
        evaluation_update: EvaluationUpdate,
        writer: &mut impl Write,
    ) -> Result<()> {
        match self.output_format {
            OutputFormat::Jsonl => {
                let json = match &evaluation_update {
                    EvaluationUpdate::RunInfo(run_info) => serde_json::to_string(run_info)?,
                    other => serde_json::to_string(other)?,
                };
                writeln!(writer, "{json}")?;
            }
            OutputFormat::Pretty => match &evaluation_update {
                EvaluationUpdate::RunInfo(run_info) => {
                    writeln!(writer, "Run ID: {}", run_info.evaluation_run_id)?;
                    writeln!(writer, "Number of datapoints: {}", run_info.num_datapoints)?;
                }
                EvaluationUpdate::Success(_) | EvaluationUpdate::Error(_) => {
                    if let Some(progress_bar) = &mut self.progress_bar {
                        progress_bar.inc(1);
                    }
                }
            },
        }
        match evaluation_update {
            EvaluationUpdate::Success(evaluation_info) => {
                self.evaluation_infos.push(evaluation_info);
            }
            EvaluationUpdate::Error(evaluation_error) => {
                self.evaluation_errors.push(evaluation_error);
            }
            EvaluationUpdate::RunInfo(_) => {
                // No data to store
            }
        }

        Ok(())
    }

    /// Computes the mean and stderr for each of the evaluations observed
    #[instrument(skip_all, fields(evaluation_infos_count = self.evaluation_infos.len(), evaluation_errors_count = self.evaluation_errors.len()))]
    pub fn compute_stats(
        &self,
        evaluators: &HashMap<String, EvaluatorConfig>,
    ) -> HashMap<String, EvaluatorStats> {
        info!("Computing evaluation statistics");
        let mut per_evaluator_stats: HashMap<String, PerEvaluatorStats> = evaluators
            .iter()
            .map(|(key, config)| (key.clone(), PerEvaluatorStats::new(config.is_bernoulli())))
            .collect();
        debug!(evaluators = ?evaluators.keys().collect::<Vec<_>>(), "Initialized data collectors for evaluators");

        // Collect evaluation inference data using PerEvaluatorStats
        debug!("Processing evaluation results into statistics");
        for evaluation_info in &self.evaluation_infos {
            for (evaluation_name, evaluation_result) in &evaluation_info.evaluations {
                match evaluation_result {
                    Some(Value::Number(n)) => {
                        if let Some(stats) = per_evaluator_stats.get_mut(evaluation_name) {
                            if let Some(num) = n.as_f64() {
                                stats.push(num as f32);
                            }
                        } else {
                            tracing::error!(
                                evaluator_name = %evaluation_name,
                                "Received evaluation result for unknown evaluator"
                            );
                        }
                    }
                    Some(Value::Bool(b)) => {
                        if let Some(stats) = per_evaluator_stats.get_mut(evaluation_name) {
                            stats.push(if *b { 1.0 } else { 0.0 });
                        } else {
                            tracing::error!(
                                evaluator_name = %evaluation_name,
                                "Received evaluation result for unknown evaluator"
                            );
                        }
                    }
                    _ => {}
                }
            }
        }

        // Convert PerEvaluatorStats to EvaluatorStats
        debug!("Computing final statistics");
        let stats: HashMap<String, EvaluatorStats> = per_evaluator_stats
            .into_iter()
            .map(|(evaluator_name, per_eval_stats)| {
                let eval_stats = per_eval_stats.to_evaluator_stats();
                debug!(
                    evaluator_name = %evaluator_name,
                    count = eval_stats.count,
                    mean = eval_stats.mean,
                    stderr = eval_stats.stderr,
                    "Computed statistics for evaluator"
                );
                (evaluator_name, eval_stats)
            })
            .collect();
        info!(
            computed_stats = stats.len(),
            "Statistics computation completed"
        );
        stats
    }
}

// We allow large enum variants because
// we expect the Success case way more often so it's ok to pay the cost
// of a large enum here.
#[expect(clippy::large_enum_variant)]
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum EvaluationUpdate {
    // RunInfo is used for internal channel communication only and receives special
    // serialization handling in EvaluationStats::push (serializes the inner RunInfo
    // directly, not the enum wrapper). The #[serde(skip)] prevents accidental
    // serialization of the enum variant itself.
    #[serde(skip)]
    RunInfo(crate::RunInfo),
    Success(EvaluationInfo),
    Error(EvaluationError),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvaluationInfo {
    pub datapoint: Datapoint,
    pub response: InferenceResponse,
    pub evaluations: HashMap<String, Option<Value>>,
    pub evaluator_errors: HashMap<String, String>,
}

impl EvaluationInfo {
    pub fn new(
        datapoint: Datapoint,
        response: InferenceResponse,
        evaluation_result: evaluators::EvaluationResult,
    ) -> Self {
        let mut evaluations = HashMap::new();
        let mut evaluator_errors = HashMap::new();
        for (evaluation_name, evaluation_result) in evaluation_result {
            match evaluation_result {
                Ok(evaluation_result) => {
                    evaluations.insert(evaluation_name, evaluation_result);
                }
                Err(e) => {
                    evaluator_errors.insert(evaluation_name, e.to_string());
                }
            }
        }
        Self {
            datapoint,
            response,
            evaluations,
            evaluator_errors,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvaluationError {
    pub datapoint_id: Uuid,
    pub message: String,
}

/// Statistics computed about a particular evaluator
/// We anticipate extending this over time
#[derive(Clone, Debug, Serialize)]
pub struct EvaluatorStats {
    pub mean: f32,
    pub stderr: f32,
    pub count: usize,
}

impl std::fmt::Display for EvaluatorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.2} ± {:.2} (n={})",
            self.mean, self.stderr, self.count
        )
    }
}

pub fn mean(data: &[f32]) -> Option<f32> {
    let sum = data.iter().sum::<f32>();
    let count = data.len();

    match count {
        positive if positive > 0 => Some(sum / count as f32),
        _ => None,
    }
}

pub fn std_deviation(data: &[f32]) -> Option<f32> {
    match (mean(data), data.len()) {
        (Some(data_mean), count) if count > 0 => {
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
        _ => None,
    }
}

/// Compute 95% Wilson confidence interval for Bernoulli data
pub fn wilson_confint(data: &[f32]) -> Option<(f64, f64)> {
    match (mean(data), data.len()) {
        (Some(data_mean), count) if count > 0 => {
            let data_mean = data_mean as f64;
            let count_f64 = count as f64;
            let z: f64 = 1.96; // Standard normal quantile for 95% CI
            let z_squared: f64 = z.powi(2);
            let scale: f64 = 1.0 / (1.0 + z_squared / count_f64);
            let center: f64 = data_mean + z_squared / (2.0 * count_f64);
            let margin = z / (2.0 * count_f64)
                * (4.0 * count_f64 * data_mean * (1.0 - data_mean) + z_squared).sqrt();
            let ci_lower = (center - margin) * scale;
            let ci_upper = (center + margin) * scale;

            // Clamp to [0, 1] to handle floating-point precision errors
            let ci_lower = ci_lower.max(0.0);
            let ci_upper = ci_upper.min(1.0);

            Some((ci_lower, ci_upper))
        }
        _ => None,
    }
}

/// Tracks statistics for a single evaluator during adaptive evaluation
/// Used for computing stopping conditions based on confidence intervals
#[derive(Default)]
pub struct PerEvaluatorStats {
    values: Vec<f32>,
    is_bernoulli: bool,
}

impl PerEvaluatorStats {
    pub fn new(is_bernoulli: bool) -> Self {
        Self {
            values: Vec::new(),
            is_bernoulli,
        }
    }

    pub fn push(&mut self, value: f32) {
        self.values.push(value);
    }

    pub fn count(&self) -> usize {
        self.values.len()
    }

    pub fn mean(&self) -> Option<f32> {
        mean(&self.values)
    }

    pub fn stderr(&self) -> Option<f32> {
        if self.values.len() < 2 {
            return None;
        }

        std_deviation(&self.values).map(|std_dev| std_dev / (self.values.len() as f32).sqrt())
    }

    /// Returns the 95% confidence interval (CI) half-width
    /// For Bernoulli evaluators: Uses Wilson CI (max distance from mean to bounds, since CI is asymmetric)
    /// For Float evaluators: Uses half-width of the Wald CI (1.96 * stderr, since CI is symmetric)
    pub fn ci_half_width(&self) -> Option<f32> {
        if self.is_bernoulli {
            let mean = self.mean()?;
            wilson_confint(&self.values).map(|(ci_lower, ci_upper)| {
                (mean as f64 - ci_lower).max(ci_upper - mean as f64) as f32
            })
        } else {
            self.stderr().map(|se| 1.96 * se)
        }
    }

    /// Converts to an EvaluatorStats snapshot for output/serialization
    pub fn to_evaluator_stats(&self) -> EvaluatorStats {
        EvaluatorStats {
            mean: self.mean().unwrap_or(0.0),
            stderr: self.stderr().unwrap_or(0.0),
            count: self.count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{wilson_confint, PerEvaluatorStats};

    #[test]
    fn test_wilson_confint_all_zeros() {
        // All 0s should still produce non-zero width interval
        let data = vec![0.0; 10];
        let result = wilson_confint(&data);
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
    fn test_wilson_confint_all_ones() {
        // All 1s should still produce non-zero width interval
        let data = vec![1.0; 10];
        let result = wilson_confint(&data);
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
    fn test_wilson_confint_half_half() {
        // 50/50 split should have symmetric interval around 0.5
        let data = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let result = wilson_confint(&data);
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
    fn test_wilson_confint_single_value() {
        // Single observation should still work
        let data = vec![1.0];
        let result = wilson_confint(&data);
        assert!(result.is_some());
        let (lower, upper) = result.unwrap();

        // Should be bounded in [0, 1]
        assert!(lower >= 0.0);
        assert!(upper <= 1.0);
        assert!(upper > lower);
    }

    #[test]
    fn test_wilson_confint_empty_data() {
        // Empty data should return None
        let data: Vec<f32> = vec![];
        let result = wilson_confint(&data);
        assert!(result.is_none());
    }

    #[test]
    fn test_wilson_confint_known_values() {
        // Test against known values
        let data = vec![0.0; 10];
        let mut test_data = data;
        test_data.extend(vec![1.0; 10]);

        let result = wilson_confint(&test_data);
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

        let result = wilson_confint(&data);
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
    fn test_wilson_confint_versus_wald_extreme() {
        // Wilson should handle extreme proportions better than Wald
        // With p=0.05 (1 success in 20), Wald can give lower < 0
        let mut data = vec![0.0; 19];
        data.push(1.0);

        let result = wilson_confint(&data);
        assert!(result.is_some());
        let (lower, _upper) = result.unwrap();

        // Wilson should keep bounds within [0, 1]
        assert!(
            lower >= 0.0,
            "Wilson lower bound should be >= 0, got {lower}",
        );
    }

    #[test]
    fn test_per_evaluator_stats_basic() {
        let mut stats = PerEvaluatorStats::new(false);
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.mean(), None);
        assert_eq!(stats.stderr(), None);
        assert_eq!(stats.ci_half_width(), None);

        // Add a single value
        stats.push(1.0);
        assert_eq!(stats.count(), 1);
        assert_eq!(stats.mean(), Some(1.0));
        assert_eq!(stats.stderr(), None); // Need at least 2 values
        assert_eq!(stats.ci_half_width(), None);
    }

    #[test]
    fn test_per_evaluator_stats_mean_and_stderr() {
        let mut stats = PerEvaluatorStats::new(false);

        // Add values: [1.0, 2.0, 3.0, 4.0, 5.0]
        // Mean = 3.0
        // Variance = ((3-1)^2 + (3-2)^2 + (3-3)^2 + (3-4)^2 + (3-5)^2) / 5 = (4+1+0+1+4)/5 = 2.0
        // StdDev = sqrt(2.0) = 1.414...
        // Stderr = 1.414.../sqrt(5) = 0.632...
        for value in [1.0, 2.0, 3.0, 4.0, 5.0] {
            stats.push(value);
        }

        assert_eq!(stats.count(), 5);
        assert_eq!(stats.mean(), Some(3.0));

        let stderr = stats.stderr().unwrap();
        assert!((stderr - 0.632).abs() < 0.01); // Approximately 0.632
    }

    #[test]
    fn test_per_evaluator_stats_ci_half_width() {
        let mut stats = PerEvaluatorStats::new(false);

        // Add values with known statistics
        for value in [1.0, 2.0, 3.0, 4.0, 5.0] {
            stats.push(value);
        }

        let ci_half_width = stats.ci_half_width().unwrap();
        let stderr = stats.stderr().unwrap();

        // CI half-width should be 1.96 * stderr
        assert!((ci_half_width - 1.96 * stderr).abs() < 0.001);
    }

    #[test]
    fn test_per_evaluator_stats_to_evaluator_stats() {
        let mut stats = PerEvaluatorStats::new(false);
        for value in [1.0, 2.0, 3.0, 4.0, 5.0] {
            stats.push(value);
        }

        let evaluator_stats = stats.to_evaluator_stats();
        assert_eq!(evaluator_stats.count, 5);
        assert_eq!(evaluator_stats.mean, 3.0);
        assert!((evaluator_stats.stderr - 0.632).abs() < 0.01);
    }

    #[test]
    fn test_per_evaluator_stats_empty_conversion() {
        let stats = PerEvaluatorStats::new(false);
        let evaluator_stats = stats.to_evaluator_stats();

        // Empty stats should have defaults
        assert_eq!(evaluator_stats.count, 0);
        assert_eq!(evaluator_stats.mean, 0.0);
        assert_eq!(evaluator_stats.stderr, 0.0);
    }
}
