#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

/// Initialized OpenAI SFT Config (per-job settings only).
/// Provider-level settings (credentials) come from
/// `provider_types.openai` defaults in the gateway config.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct OpenAISFTConfig {
    pub model: String,
    pub batch_size: Option<usize>,
    pub learning_rate_multiplier: Option<f64>,
    pub n_epochs: Option<usize>,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
}

/// Uninitialized OpenAI SFT Config (per-job settings only).
/// Provider-level settings (credentials) come from
/// `provider_types.openai` defaults in the gateway config.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "OpenAISFTConfig"))]
pub struct UninitializedOpenAISFTConfig {
    pub model: String,
    pub batch_size: Option<usize>,
    pub learning_rate_multiplier: Option<f64>,
    pub n_epochs: Option<usize>,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
}

impl std::fmt::Display for UninitializedOpenAISFTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UninitializedOpenAISFTConfig {
    /// Initialize the OpenAISFTConfig.
    ///
    /// Provider-level settings (credentials) are configured in the gateway config at
    /// `[provider_types.openai.defaults]`.
    ///
    /// :param model: The model to use for the fine-tuning job (required).
    /// :param batch_size: The batch size to use for the fine-tuning job.
    /// :param learning_rate_multiplier: The learning rate multiplier to use for the fine-tuning job.
    /// :param n_epochs: The number of epochs to use for the fine-tuning job.
    /// :param seed: The seed to use for the fine-tuning job.
    /// :param suffix: The suffix to use for the fine-tuning job (this is for naming in OpenAI).
    #[new]
    #[pyo3(signature = (*, model, batch_size=None, learning_rate_multiplier=None, n_epochs=None, seed=None, suffix=None))]
    pub fn new(
        model: String,
        batch_size: Option<usize>,
        learning_rate_multiplier: Option<f64>,
        n_epochs: Option<usize>,
        seed: Option<u64>,
        suffix: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            model,
            batch_size,
            learning_rate_multiplier,
            n_epochs,
            seed,
            suffix,
        })
    }

    #[expect(unused_variables)]
    #[pyo3(signature = (*, model, batch_size=None, learning_rate_multiplier=None, n_epochs=None, seed=None, suffix=None))]
    fn __init__(
        this: Py<Self>,
        model: String,
        batch_size: Option<usize>,
        learning_rate_multiplier: Option<f64>,
        n_epochs: Option<usize>,
        seed: Option<u64>,
        suffix: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedOpenAISFTConfig {
    pub fn load(self) -> OpenAISFTConfig {
        OpenAISFTConfig {
            model: self.model,
            batch_size: self.batch_size,
            learning_rate_multiplier: self.learning_rate_multiplier,
            n_epochs: self.n_epochs,
            seed: self.seed,
            suffix: self.suffix,
        }
    }
}

/// Minimal job handle for OpenAI SFT.
/// All configuration needed for polling comes from provider_types at poll time.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct OpenAISFTJobHandle {
    pub job_id: String,
    /// A url to a human-readable page for the job.
    pub job_url: Url,
    pub job_api_url: Url,
}

impl std::fmt::Display for OpenAISFTJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}
