#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::deserialize_from_pyobj;
use crate::model_table::{OpenAIKind, ProviderKind, ProviderTypeDefaultCredentials};
#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::{
    endpoints::openai_compatible::types::chat_completions::JsonSchemaInfo,
    error::Error,
    model::CredentialLocationWithFallback,
    providers::openai::{grader::OpenAIGrader, OpenAICredentials},
};

#[cfg(feature = "pyo3")]
use crate::model::CredentialLocation;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
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

#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
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
    #[serde(skip)]
    pub credentials: OpenAICredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocationWithFallback>,
    pub api_base: Option<Url>,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
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
    #[serde(skip)]
    pub credentials: Option<CredentialLocationWithFallback>,
    pub api_base: Option<Url>,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
}

impl std::fmt::Display for UninitializedOpenAIRFTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UninitializedOpenAIRFTConfig {
    // We allow too many arguments since it is a Python constructor
    /// NOTE: This signature currently does not work:
    /// print(OpenAIRFTConfig.__init__.__text_signature__)
    /// prints out signature:
    /// ($self, /, *args, **kwargs)
    #[expect(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, model, grader, response_format=None, batch_size=None, compute_multiplier=None, eval_interval=None, eval_samples=None, learning_rate_multiplier=None, n_epochs=None, reasoning_effort=None, credentials=None, api_base=None, seed=None, suffix=None))]
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
        credentials: Option<String>,
        api_base: Option<String>,
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

        // Use Deserialize to convert the string to a CredentialLocationWithFallback
        let credentials = credentials.map(|s| {
            serde_json::from_str(&s).unwrap_or(CredentialLocationWithFallback::Single(
                CredentialLocation::Env(s),
            ))
        });
        let api_base = api_base
            .map(|s| {
                Url::parse(&s)
                    .map_err(|e| PyErr::new::<PyValueError, std::string::String>(e.to_string()))
            })
            .transpose()?;
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
            credentials,
            api_base,
            seed,
            suffix,
        })
    }

    /// Initialize the OpenAISFTConfig. All parameters are optional except for `model`.
    ///
    /// :param model: The model to use for the reinforcement fine-tuning job.
    /// :param grader: The grader to use for the reinforcement fine-tuning job.
    /// :param response_format: The response format to use for the reinforcement fine-tuning job.
    /// :param batch_size: The batch size to use for the reinforcement fine-tuning job.
    /// :param compute_multiplier: The compute multiplier to use for the reinforcement fine-tuning job.
    /// :param eval_interval: The eval interval to use for the fine-tuning job.
    /// :param eval_samples: The eval samples to use for the fine-tuning job.
    /// :param batch_size: The batch size to use for the fine-tuning job.
    /// :param learning_rate_multiplier: The learning rate multiplier to use for the fine-tuning job.
    /// :param n_epochs: The number of epochs to use for the fine-tuning job.
    /// :param credentials: The credentials to use for the fine-tuning job. This should be a string like "env::OPENAI_API_KEY". See docs for more details.
    /// :param api_base: The base URL to use for the fine-tuning job. This is primarily used for testing.
    /// :param seed: The seed to use for the fine-tuning job.
    /// :param suffix: The suffix to use for the fine-tuning job (this is for naming in OpenAI).
    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, grader, response_format=None, batch_size=None, compute_multiplier=None, eval_interval=None, eval_samples=None, learning_rate_multiplier=None, n_epochs=None, reasoning_effort=None, credentials=None, api_base=None, seed=None, suffix=None))]
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
        credentials: Option<String>,
        api_base: Option<String>,
        seed: Option<u64>,
        suffix: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedOpenAIRFTConfig {
    pub async fn load(
        self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OpenAIRFTConfig, Error> {
        Ok(OpenAIRFTConfig {
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
            credentials: OpenAIKind
                .get_defaulted_credential(self.credentials.as_ref(), default_credentials)
                .await?,
            credential_location: self.credentials,
            api_base: self.api_base,
            suffix: self.suffix,
            seed: self.seed,
        })
    }
}

#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct OpenAIRFTJobHandle {
    pub job_id: String,
    /// A url to a human-readable page for the job.
    pub job_url: Url,
    pub job_api_url: Url,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocationWithFallback>,
}

impl std::fmt::Display for OpenAIRFTJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}
