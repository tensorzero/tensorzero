#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::deserialize_from_pyobj;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_derive::TensorZeroDeserialize;
use tensorzero_stored_config::{
    StoredOpenAIGrader, StoredOpenAIModelGraderInput, StoredOpenAIRFTConfig,
    StoredOpenAIRFTResponseFormat, StoredOpenAIRFTRole, StoredOpenAISimilarityMetric,
    StoredOpenAIStringCheckOp, StoredRFTJsonSchemaInfo,
};
use url::Url;

use crate::{
    endpoints::openai_compatible::types::chat_completions::JsonSchemaInfo,
    providers::openai::grader::OpenAIGrader,
};

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "RFTJsonSchemaInfoOption"))]
#[serde(untagged)]
pub enum RFTJsonSchemaInfoOption {
    JsonSchema(JsonSchemaInfo),
}

impl std::fmt::Display for RFTJsonSchemaInfoOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Response format configuration for OpenAI Reinforcement Fine-Tuning (RFT).
///
/// When a response format is specified, the model being fine-tuned will produce
/// structured outputs that conform to the provided JSON schema during RFT sampling.
/// These structured outputs will be populated in the `output_json` field of the
/// Sample namespace.
///
/// If no response format is specified but the model is instructed (e.g., via prompts)
/// to produce structured outputs, those outputs will be returned as raw JSON strings
/// in the `output_text` field of the Sample namespace instead.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, JsonSchema, PartialEq, Serialize, TensorZeroDeserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "OpenAIRFTResponseFormat"))]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum OpenAIRFTResponseFormat {
    JsonSchema {
        json_schema: RFTJsonSchemaInfoOption,
    },
}

impl std::fmt::Display for OpenAIRFTResponseFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Initialized OpenAI RFT Config (per-job settings only).
/// Provider-level settings (credentials) come from
/// `provider_types.openai` defaults in the gateway config.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct OpenAIRFTConfig {
    pub model: String,
    pub grader: OpenAIGrader,
    pub response_format: Option<OpenAIRFTResponseFormat>,
    pub batch_size: Option<usize>,
    pub compute_multiplier: Option<f64>,
    pub eval_interval: Option<usize>,
    pub eval_samples: Option<usize>,
    pub learning_rate_multiplier: Option<f64>,
    pub n_epochs: Option<usize>,
    pub reasoning_effort: Option<String>,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
}

/// Uninitialized OpenAI RFT Config (per-job settings only).
/// Provider-level settings (credentials) come from
/// `provider_types.openai` defaults in the gateway config.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "OpenAIRFTConfig"))]
pub struct UninitializedOpenAIRFTConfig {
    pub model: String,
    pub grader: OpenAIGrader,
    pub response_format: Option<OpenAIRFTResponseFormat>,
    pub batch_size: Option<usize>,
    pub compute_multiplier: Option<f64>,
    pub eval_interval: Option<usize>,
    pub eval_samples: Option<usize>,
    pub learning_rate_multiplier: Option<f64>,
    pub n_epochs: Option<usize>,
    pub reasoning_effort: Option<String>,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
}

impl std::fmt::Display for UninitializedOpenAIRFTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl From<StoredOpenAIStringCheckOp> for crate::providers::openai::grader::OpenAIStringCheckOp {
    fn from(stored: StoredOpenAIStringCheckOp) -> Self {
        match stored {
            StoredOpenAIStringCheckOp::Eq => Self::Eq,
            StoredOpenAIStringCheckOp::Ne => Self::Ne,
            StoredOpenAIStringCheckOp::Like => Self::Like,
            StoredOpenAIStringCheckOp::Ilike => Self::Ilike,
        }
    }
}

impl From<StoredOpenAISimilarityMetric>
    for crate::providers::openai::grader::OpenAISimilarityMetric
{
    fn from(stored: StoredOpenAISimilarityMetric) -> Self {
        match stored {
            StoredOpenAISimilarityMetric::FuzzyMatch => Self::FuzzyMatch,
            StoredOpenAISimilarityMetric::Bleu => Self::Bleu,
            StoredOpenAISimilarityMetric::Gleu => Self::Gleu,
            StoredOpenAISimilarityMetric::Meteor => Self::Meteor,
            StoredOpenAISimilarityMetric::Rouge1 => Self::Rouge1,
            StoredOpenAISimilarityMetric::Rouge2 => Self::Rouge2,
            StoredOpenAISimilarityMetric::Rouge3 => Self::Rouge3,
            StoredOpenAISimilarityMetric::Rouge4 => Self::Rouge4,
            StoredOpenAISimilarityMetric::Rouge5 => Self::Rouge5,
            StoredOpenAISimilarityMetric::RougeL => Self::RougeL,
        }
    }
}

impl From<StoredOpenAIRFTRole> for crate::providers::openai::grader::OpenAIRFTRole {
    fn from(stored: StoredOpenAIRFTRole) -> Self {
        match stored {
            StoredOpenAIRFTRole::Developer => Self::Developer,
            StoredOpenAIRFTRole::User => Self::User,
        }
    }
}

impl From<StoredOpenAIModelGraderInput>
    for crate::providers::openai::grader::OpenAIModelGraderInput
{
    fn from(stored: StoredOpenAIModelGraderInput) -> Self {
        Self {
            role: stored.role.into(),
            content: stored.content,
        }
    }
}

impl From<StoredOpenAIGrader> for OpenAIGrader {
    fn from(stored: StoredOpenAIGrader) -> Self {
        match stored {
            StoredOpenAIGrader::StringCheck {
                name,
                operation,
                input,
                reference,
            } => Self::StringCheck {
                name,
                operation: operation.into(),
                input,
                reference,
            },
            StoredOpenAIGrader::TextSimilarity {
                name,
                evaluation_metric,
                input,
                reference,
            } => Self::TextSimilarity {
                name,
                evaluation_metric: evaluation_metric.into(),
                input,
                reference,
            },
            StoredOpenAIGrader::ScoreModel {
                name,
                model,
                input,
                range,
            } => Self::ScoreModel {
                name,
                model,
                input: input.into_iter().map(Into::into).collect(),
                range,
            },
            StoredOpenAIGrader::LabelModel {
                name,
                model,
                labels,
                passing_labels,
                input,
            } => Self::LabelModel {
                name,
                model,
                labels,
                passing_labels,
                input: input.into_iter().map(Into::into).collect(),
            },
            StoredOpenAIGrader::Python {
                name,
                source,
                image_tag,
            } => Self::Python {
                name,
                source,
                image_tag,
            },
            StoredOpenAIGrader::Multi {
                calculate_output,
                graders,
                name,
            } => Self::Multi {
                calculate_output,
                graders: graders
                    .into_iter()
                    .map(|(k, v)| (k, Box::new((*v).into())))
                    .collect(),
                name,
            },
        }
    }
}

impl From<StoredRFTJsonSchemaInfo> for JsonSchemaInfo {
    fn from(stored: StoredRFTJsonSchemaInfo) -> Self {
        Self {
            name: stored.name,
            description: stored.description,
            schema: stored.schema,
            strict: stored.strict.unwrap_or_default(),
        }
    }
}

impl From<StoredOpenAIRFTResponseFormat> for OpenAIRFTResponseFormat {
    fn from(stored: StoredOpenAIRFTResponseFormat) -> Self {
        match stored {
            StoredOpenAIRFTResponseFormat::JsonSchema { json_schema } => Self::JsonSchema {
                json_schema: RFTJsonSchemaInfoOption::JsonSchema(json_schema.into()),
            },
        }
    }
}

impl From<StoredOpenAIRFTConfig> for UninitializedOpenAIRFTConfig {
    fn from(stored: StoredOpenAIRFTConfig) -> Self {
        UninitializedOpenAIRFTConfig {
            model: stored.model,
            grader: stored.grader.into(),
            response_format: stored.response_format.map(Into::into),
            batch_size: stored.batch_size,
            compute_multiplier: stored.compute_multiplier,
            eval_interval: stored.eval_interval,
            eval_samples: stored.eval_samples,
            learning_rate_multiplier: stored.learning_rate_multiplier,
            n_epochs: stored.n_epochs,
            reasoning_effort: stored.reasoning_effort,
            seed: stored.seed,
            suffix: stored.suffix,
        }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UninitializedOpenAIRFTConfig {
    /// Initialize the OpenAIRFTConfig.
    ///
    /// Provider-level settings (credentials) are configured in the gateway config at
    /// `[provider_types.openai.defaults]`.
    ///
    /// :param model: The model to use for the reinforcement fine-tuning job (required).
    /// :param grader: The grader to use for the reinforcement fine-tuning job (required).
    /// :param response_format: The response format to use for the reinforcement fine-tuning job.
    /// :param batch_size: The batch size to use for the reinforcement fine-tuning job.
    /// :param compute_multiplier: The compute multiplier to use for the reinforcement fine-tuning job.
    /// :param eval_interval: The eval interval to use for the fine-tuning job.
    /// :param eval_samples: The eval samples to use for the fine-tuning job.
    /// :param learning_rate_multiplier: The learning rate multiplier to use for the fine-tuning job.
    /// :param n_epochs: The number of epochs to use for the fine-tuning job.
    /// :param reasoning_effort: The reasoning effort to use for the fine-tuning job.
    /// :param seed: The seed to use for the fine-tuning job.
    /// :param suffix: The suffix to use for the fine-tuning job (this is for naming in OpenAI).
    #[new]
    #[expect(clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, grader, response_format=None, batch_size=None, compute_multiplier=None, eval_interval=None, eval_samples=None, learning_rate_multiplier=None, n_epochs=None, reasoning_effort=None, seed=None, suffix=None))]
    pub fn new(
        py: Python,
        model: String,
        grader: &Bound<'_, PyAny>,
        response_format: Option<&Bound<'_, PyAny>>,
        batch_size: Option<usize>,
        compute_multiplier: Option<f64>,
        eval_interval: Option<usize>,
        eval_samples: Option<usize>,
        learning_rate_multiplier: Option<f64>,
        n_epochs: Option<usize>,
        reasoning_effort: Option<String>,
        seed: Option<u64>,
        suffix: Option<String>,
    ) -> PyResult<Self> {
        // Deserialize the grader from Python dict to Rust OpenAIGrader
        let grader: OpenAIGrader = if let Ok(grader) = grader.extract::<OpenAIGrader>() {
            // If it's already a Grader object, use it directly
            grader
        } else {
            // Otherwise, try to deserialize from a Python dict
            deserialize_from_pyobj(py, grader)?
        };

        // Deserialize the response_format from Python dict to Rust OpenAIRFTResponseFormat
        let response_format: Option<OpenAIRFTResponseFormat> = if let Some(rf) = response_format {
            if let Ok(response_format) = rf.extract::<OpenAIRFTResponseFormat>() {
                // If it's already a ResponseFormat object, use it directly
                Some(response_format)
            } else {
                // Otherwise, try to deserialize from a Python dict
                Some(deserialize_from_pyobj(py, rf)?)
            }
        } else {
            None
        };

        Ok(Self {
            model,
            grader,
            response_format,
            batch_size,
            compute_multiplier,
            eval_interval,
            eval_samples,
            learning_rate_multiplier,
            n_epochs,
            reasoning_effort,
            seed,
            suffix,
        })
    }

    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, grader, response_format=None, batch_size=None, compute_multiplier=None, eval_interval=None, eval_samples=None, learning_rate_multiplier=None, n_epochs=None, reasoning_effort=None, seed=None, suffix=None))]
    fn __init__(
        this: Py<Self>,
        model: String,
        grader: OpenAIGrader,
        response_format: Option<OpenAIRFTResponseFormat>,
        batch_size: Option<usize>,
        compute_multiplier: Option<f64>,
        eval_interval: Option<usize>,
        eval_samples: Option<usize>,
        learning_rate_multiplier: Option<f64>,
        n_epochs: Option<usize>,
        reasoning_effort: Option<String>,
        seed: Option<u64>,
        suffix: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedOpenAIRFTConfig {
    pub fn load(self) -> OpenAIRFTConfig {
        OpenAIRFTConfig {
            model: self.model,
            grader: self.grader,
            response_format: self.response_format,
            batch_size: self.batch_size,
            compute_multiplier: self.compute_multiplier,
            eval_interval: self.eval_interval,
            eval_samples: self.eval_samples,
            learning_rate_multiplier: self.learning_rate_multiplier,
            n_epochs: self.n_epochs,
            reasoning_effort: self.reasoning_effort,
            suffix: self.suffix,
            seed: self.seed,
        }
    }
}

/// Minimal job handle for OpenAI RFT.
/// All configuration needed for polling comes from provider_types at poll time.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct OpenAIRFTJobHandle {
    pub job_id: String,
    /// A url to a human-readable page for the job.
    pub job_url: Url,
    pub job_api_url: Url,
}

impl std::fmt::Display for OpenAIRFTJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}
