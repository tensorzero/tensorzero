#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::deserialize_from_pyobj;
use crate::providers::openai::grader::{
    OpenAIModelGraderInput, OpenAIRFTRole, OpenAISimilarityMetric, OpenAIStringCheckOp,
};
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

// --- Reverse conversions: core -> stored ---

impl From<OpenAIStringCheckOp> for StoredOpenAIStringCheckOp {
    fn from(op: OpenAIStringCheckOp) -> Self {
        match op {
            OpenAIStringCheckOp::Eq => Self::Eq,
            OpenAIStringCheckOp::Ne => Self::Ne,
            OpenAIStringCheckOp::Like => Self::Like,
            OpenAIStringCheckOp::Ilike => Self::Ilike,
        }
    }
}

impl From<OpenAISimilarityMetric> for StoredOpenAISimilarityMetric {
    fn from(metric: OpenAISimilarityMetric) -> Self {
        match metric {
            OpenAISimilarityMetric::FuzzyMatch => Self::FuzzyMatch,
            OpenAISimilarityMetric::Bleu => Self::Bleu,
            OpenAISimilarityMetric::Gleu => Self::Gleu,
            OpenAISimilarityMetric::Meteor => Self::Meteor,
            OpenAISimilarityMetric::Rouge1 => Self::Rouge1,
            OpenAISimilarityMetric::Rouge2 => Self::Rouge2,
            OpenAISimilarityMetric::Rouge3 => Self::Rouge3,
            OpenAISimilarityMetric::Rouge4 => Self::Rouge4,
            OpenAISimilarityMetric::Rouge5 => Self::Rouge5,
            OpenAISimilarityMetric::RougeL => Self::RougeL,
        }
    }
}

impl From<OpenAIRFTRole> for StoredOpenAIRFTRole {
    fn from(role: OpenAIRFTRole) -> Self {
        match role {
            OpenAIRFTRole::Developer => Self::Developer,
            OpenAIRFTRole::User => Self::User,
        }
    }
}

impl From<OpenAIModelGraderInput> for StoredOpenAIModelGraderInput {
    fn from(input: OpenAIModelGraderInput) -> Self {
        Self {
            role: input.role.into(),
            content: input.content,
        }
    }
}

impl From<OpenAIGrader> for StoredOpenAIGrader {
    fn from(grader: OpenAIGrader) -> Self {
        match grader {
            OpenAIGrader::StringCheck {
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
            OpenAIGrader::TextSimilarity {
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
            OpenAIGrader::ScoreModel {
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
            OpenAIGrader::LabelModel {
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
            OpenAIGrader::Python {
                name,
                source,
                image_tag,
            } => Self::Python {
                name,
                source,
                image_tag,
            },
            OpenAIGrader::Multi {
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

impl From<JsonSchemaInfo> for StoredRFTJsonSchemaInfo {
    fn from(info: JsonSchemaInfo) -> Self {
        Self {
            name: info.name,
            description: info.description,
            schema: info.schema,
            strict: Some(info.strict),
        }
    }
}

impl From<OpenAIRFTResponseFormat> for StoredOpenAIRFTResponseFormat {
    fn from(format: OpenAIRFTResponseFormat) -> Self {
        match format {
            OpenAIRFTResponseFormat::JsonSchema { json_schema } => {
                let RFTJsonSchemaInfoOption::JsonSchema(info) = json_schema;
                Self::JsonSchema {
                    json_schema: info.into(),
                }
            }
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

impl From<UninitializedOpenAIRFTConfig> for StoredOpenAIRFTConfig {
    fn from(config: UninitializedOpenAIRFTConfig) -> Self {
        StoredOpenAIRFTConfig {
            model: config.model,
            grader: config.grader.into(),
            response_format: config.response_format.map(Into::into),
            batch_size: config.batch_size,
            compute_multiplier: config.compute_multiplier,
            eval_interval: config.eval_interval,
            eval_samples: config.eval_samples,
            learning_rate_multiplier: config.learning_rate_multiplier,
            n_epochs: config.n_epochs,
            reasoning_effort: config.reasoning_effort,
            seed: config.seed,
            suffix: config.suffix,
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

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;
    use std::collections::HashMap;

    fn sample_string_check_grader() -> OpenAIGrader {
        OpenAIGrader::StringCheck {
            name: "exact-match".to_string(),
            operation: OpenAIStringCheckOp::Eq,
            input: "{{output}}".to_string(),
            reference: "{{reference}}".to_string(),
        }
    }

    fn sample_score_model_grader() -> OpenAIGrader {
        OpenAIGrader::ScoreModel {
            name: "score-model".to_string(),
            model: "gpt-4o".to_string(),
            input: vec![OpenAIModelGraderInput {
                role: OpenAIRFTRole::Developer,
                content: "rate this".to_string(),
            }],
            range: Some([0.0, 1.0]),
        }
    }

    #[gtest]
    fn test_openai_string_check_op_round_trip() {
        for original in [
            OpenAIStringCheckOp::Eq,
            OpenAIStringCheckOp::Ne,
            OpenAIStringCheckOp::Like,
            OpenAIStringCheckOp::Ilike,
        ] {
            let stored: StoredOpenAIStringCheckOp = original.clone().into();
            let restored: OpenAIStringCheckOp = stored.into();
            expect_that!(restored, eq(&original));
        }
    }

    #[gtest]
    fn test_openai_similarity_metric_round_trip() {
        for original in [
            OpenAISimilarityMetric::FuzzyMatch,
            OpenAISimilarityMetric::Bleu,
            OpenAISimilarityMetric::Gleu,
            OpenAISimilarityMetric::Meteor,
            OpenAISimilarityMetric::Rouge1,
            OpenAISimilarityMetric::Rouge2,
            OpenAISimilarityMetric::Rouge3,
            OpenAISimilarityMetric::Rouge4,
            OpenAISimilarityMetric::Rouge5,
            OpenAISimilarityMetric::RougeL,
        ] {
            let stored: StoredOpenAISimilarityMetric = original.clone().into();
            let restored: OpenAISimilarityMetric = stored.into();
            expect_that!(restored, eq(&original));
        }
    }

    #[gtest]
    fn test_openai_rft_role_round_trip() {
        for original in [OpenAIRFTRole::Developer, OpenAIRFTRole::User] {
            let stored: StoredOpenAIRFTRole = original.into();
            let restored: OpenAIRFTRole = stored.into();
            expect_that!(restored, eq(original));
        }
    }

    #[gtest]
    fn test_openai_grader_string_check_round_trip() {
        let original = sample_string_check_grader();
        let stored: StoredOpenAIGrader = original.clone().into();
        let restored: OpenAIGrader = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_openai_grader_text_similarity_round_trip() {
        let original = OpenAIGrader::TextSimilarity {
            name: "text-sim".to_string(),
            evaluation_metric: OpenAISimilarityMetric::Bleu,
            input: "{{output}}".to_string(),
            reference: "{{reference}}".to_string(),
        };
        let stored: StoredOpenAIGrader = original.clone().into();
        let restored: OpenAIGrader = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_openai_grader_score_model_round_trip() {
        let original = sample_score_model_grader();
        let stored: StoredOpenAIGrader = original.clone().into();
        let restored: OpenAIGrader = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_openai_grader_label_model_round_trip() {
        let original = OpenAIGrader::LabelModel {
            name: "label-model".to_string(),
            model: "gpt-4o".to_string(),
            labels: vec!["good".to_string(), "bad".to_string()],
            passing_labels: vec!["good".to_string()],
            input: vec![OpenAIModelGraderInput {
                role: OpenAIRFTRole::User,
                content: "label this".to_string(),
            }],
        };
        let stored: StoredOpenAIGrader = original.clone().into();
        let restored: OpenAIGrader = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_openai_grader_python_round_trip() {
        let original = OpenAIGrader::Python {
            name: "py-grader".to_string(),
            source: "def grade(): return 1.0".to_string(),
            image_tag: Some("latest".to_string()),
        };
        let stored: StoredOpenAIGrader = original.clone().into();
        let restored: OpenAIGrader = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_openai_grader_multi_round_trip() {
        let mut graders = HashMap::new();
        graders.insert("a".to_string(), Box::new(sample_string_check_grader()));
        graders.insert("b".to_string(), Box::new(sample_score_model_grader()));
        let original = OpenAIGrader::Multi {
            calculate_output: "a + b".to_string(),
            graders,
            name: "combined".to_string(),
        };
        let stored: StoredOpenAIGrader = original.clone().into();
        let restored: OpenAIGrader = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_openai_rft_response_format_round_trip() {
        let original = OpenAIRFTResponseFormat::JsonSchema {
            json_schema: RFTJsonSchemaInfoOption::JsonSchema(JsonSchemaInfo {
                name: "schema".to_string(),
                description: Some("desc".to_string()),
                schema: Some(serde_json::json!({"type": "object"})),
                strict: true,
            }),
        };
        let stored: StoredOpenAIRFTResponseFormat = original.clone().into();
        let restored: OpenAIRFTResponseFormat = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_openai_rft_config_round_trip_full() {
        let original = UninitializedOpenAIRFTConfig {
            model: "gpt-4o-mini".to_string(),
            grader: sample_string_check_grader(),
            response_format: Some(OpenAIRFTResponseFormat::JsonSchema {
                json_schema: RFTJsonSchemaInfoOption::JsonSchema(JsonSchemaInfo {
                    name: "schema".to_string(),
                    description: None,
                    schema: Some(serde_json::json!({"type": "object"})),
                    strict: true,
                }),
            }),
            batch_size: Some(8),
            compute_multiplier: Some(1.5),
            eval_interval: Some(10),
            eval_samples: Some(100),
            learning_rate_multiplier: Some(0.5),
            n_epochs: Some(3),
            reasoning_effort: Some("low".to_string()),
            seed: Some(42),
            suffix: Some("rft-tune".to_string()),
        };
        let stored: StoredOpenAIRFTConfig = original.clone().into();
        let restored: UninitializedOpenAIRFTConfig = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_openai_rft_config_round_trip_minimal() {
        let original = UninitializedOpenAIRFTConfig {
            model: "gpt-4o-mini".to_string(),
            grader: sample_string_check_grader(),
            response_format: None,
            batch_size: None,
            compute_multiplier: None,
            eval_interval: None,
            eval_samples: None,
            learning_rate_multiplier: None,
            n_epochs: None,
            reasoning_effort: None,
            seed: None,
            suffix: None,
        };
        let stored: StoredOpenAIRFTConfig = original.clone().into();
        let restored: UninitializedOpenAIRFTConfig = stored.into();
        expect_that!(restored, eq(&original));
    }
}
