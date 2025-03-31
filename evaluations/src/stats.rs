use std::{collections::HashMap, io::Write};

use anyhow::Result;
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero::InferenceResponse;
use tensorzero_internal::{endpoints::datasets::Datapoint, evaluations::EvaluatorConfig};
use uuid::Uuid;

use crate::{evaluators, OutputFormat};

pub(crate) struct EvalStats {
    pub output_format: OutputFormat,
    pub eval_infos: Vec<EvalInfo>,
    pub eval_errors: Vec<EvalError>,
    pub progress_bar: Option<ProgressBar>,
}

impl EvalStats {
    pub(crate) fn new(output_format: OutputFormat, dataset_len: usize) -> Self {
        let progress_bar = match output_format {
            OutputFormat::Jsonl => None,
            OutputFormat::HumanReadable => Some(ProgressBar::new(dataset_len as u64)),
        };
        Self {
            output_format,
            eval_infos: Vec::new(),
            eval_errors: Vec::new(),
            progress_bar,
        }
    }

    pub(crate) fn push(&mut self, eval_update: EvalUpdate, writer: &mut impl Write) -> Result<()> {
        match self.output_format {
            OutputFormat::Jsonl => {
                writeln!(writer, "{}", serde_json::to_string(&eval_update)?)?;
            }
            OutputFormat::HumanReadable => {
                if let Some(progress_bar) = &mut self.progress_bar {
                    progress_bar.inc(1);
                }
            }
        }
        match eval_update {
            EvalUpdate::Success(eval_info) => self.eval_infos.push(eval_info),
            EvalUpdate::Error(eval_error) => {
                self.eval_errors.push(eval_error);
            }
        }
        Ok(())
    }

    /// Computes the mean and stderr for each of the evals observed
    pub(crate) fn compute_stats(
        &self,
        evaluators: &HashMap<String, EvaluatorConfig>,
    ) -> HashMap<String, EvaluatorStats> {
        let mut data: HashMap<String, Vec<f32>> = evaluators
            .keys()
            .map(|key| (key.clone(), Vec::new()))
            .collect();
        // Collect eval inference data into vectors by eval (all as floats)
        for eval_info in self.eval_infos.iter() {
            for (eval_name, eval_result) in eval_info.evaluations.iter() {
                match eval_result {
                    Some(Value::Number(n)) => {
                        if let Some(data_vec) = data.get_mut(eval_name) {
                            if let Some(num) = n.as_f64() {
                                data_vec.push(num as f32);
                            }
                        }
                    }
                    Some(Value::Bool(b)) => {
                        if let Some(data_vec) = data.get_mut(eval_name) {
                            data_vec.push(if *b { 1.0 } else { 0.0 });
                        }
                    }
                    _ => {}
                }
            }
        }

        // Compute stats
        let mut stats = HashMap::new();
        for (evaluator_name, data_vec) in data.into_iter() {
            let mean = mean(&data_vec).unwrap_or(0.0);
            let stderr = match std_deviation(&data_vec) {
                Some(std_dev) if !data_vec.is_empty() => std_dev / (data_vec.len() as f32).sqrt(),
                _ => 0.0,
            };
            stats.insert(evaluator_name.clone(), EvaluatorStats { mean, stderr });
        }
        stats
    }
}

// We allow large enum variants because
// we expect the Success case way more often so it's ok to pay the cost
// of a large enum here.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum EvalUpdate {
    Success(EvalInfo),
    Error(EvalError),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EvalInfo {
    pub datapoint: Datapoint,
    pub response: InferenceResponse,
    pub evaluations: HashMap<String, Option<Value>>,
    pub evaluator_errors: HashMap<String, String>,
}

impl EvalInfo {
    pub fn new(
        datapoint: Datapoint,
        response: InferenceResponse,
        eval_result: evaluators::EvalResult,
    ) -> Self {
        let mut evaluations = HashMap::new();
        let mut evaluator_errors = HashMap::new();
        for (eval_name, eval_result) in eval_result {
            match eval_result {
                Ok(eval_result) => {
                    evaluations.insert(eval_name, eval_result);
                }
                Err(e) => {
                    evaluator_errors.insert(eval_name, e.to_string());
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

#[derive(Debug, Deserialize, Serialize)]
pub struct EvalError {
    pub datapoint_id: Uuid,
    pub message: String,
}

/// Statistics computed about a particular evaluator
/// We anticipate extending this over time
pub struct EvaluatorStats {
    pub mean: f32,
    pub stderr: f32,
}

impl std::fmt::Display for EvaluatorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2} ± {:.2}", self.mean, self.stderr)
    }
}

fn mean(data: &[f32]) -> Option<f32> {
    let sum = data.iter().sum::<f32>();
    let count = data.len();

    match count {
        positive if positive > 0 => Some(sum / count as f32),
        _ => None,
    }
}

fn std_deviation(data: &[f32]) -> Option<f32> {
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
