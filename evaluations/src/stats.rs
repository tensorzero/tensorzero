use std::{collections::HashMap, io::Write};

use anyhow::Result;
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_core::client::InferenceResponse;
use tensorzero_core::statistics_util::{mean, std_deviation, wilson_confint_from_data};
use tensorzero_core::{endpoints::datasets::Datapoint, evaluations::EvaluatorConfig};
use tracing::{debug, info, instrument};
use uuid::Uuid;

use crate::{OutputFormat, evaluators};

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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
            wilson_confint_from_data(&self.values).map(|(ci_lower, ci_upper)| {
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
    use super::PerEvaluatorStats;

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
