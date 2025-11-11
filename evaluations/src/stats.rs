use std::{collections::HashMap, io::Write};

use anyhow::Result;
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero_core::client::InferenceResponse;
use tensorzero_core::{endpoints::datasets::StoredDatapoint, evaluations::EvaluatorConfig};
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
        let mut data: HashMap<String, Vec<f32>> = evaluators
            .keys()
            .map(|key| (key.clone(), Vec::new()))
            .collect();
        debug!(evaluators = ?evaluators.keys().collect::<Vec<_>>(), "Initialized data collectors for evaluators");
        // Collect evaluation inference data into vectors by evaluation (all as floats)
        debug!("Processing evaluation results into statistics");
        for evaluation_info in &self.evaluation_infos {
            for (evaluation_name, evaluation_result) in &evaluation_info.evaluations {
                match evaluation_result {
                    Some(Value::Number(n)) => {
                        if let Some(data_vec) = data.get_mut(evaluation_name) {
                            if let Some(num) = n.as_f64() {
                                data_vec.push(num as f32);
                            }
                        }
                    }
                    Some(Value::Bool(b)) => {
                        if let Some(data_vec) = data.get_mut(evaluation_name) {
                            data_vec.push(if *b { 1.0 } else { 0.0 });
                        }
                    }
                    _ => {}
                }
            }
        }

        // Compute stats
        debug!("Computing final statistics");
        let mut stats = HashMap::new();
        for (evaluator_name, data_vec) in data.into_iter() {
            let count = data_vec.len();
            let mean = mean(&data_vec).unwrap_or(0.0);
            let stderr = match std_deviation(&data_vec) {
                Some(std_dev) if !data_vec.is_empty() => std_dev / (data_vec.len() as f32).sqrt(),
                _ => 0.0,
            };
            debug!(
                evaluator_name = %evaluator_name,
                count = count,
                mean = mean,
                stderr = stderr,
                "Computed statistics for evaluator"
            );
            stats.insert(
                evaluator_name.clone(),
                EvaluatorStats {
                    mean,
                    stderr,
                    count,
                },
            );
        }
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
    pub datapoint: StoredDatapoint,
    pub response: InferenceResponse,
    pub evaluations: HashMap<String, Option<Value>>,
    pub evaluator_errors: HashMap<String, String>,
}

impl EvaluationInfo {
    pub fn new(
        datapoint: StoredDatapoint,
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
