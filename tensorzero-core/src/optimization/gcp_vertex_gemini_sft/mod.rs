#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

/// Initialized GCP Vertex Gemini SFT Config (per-job settings only).
/// Provider-level settings (project_id, region, bucket_name, credentials, etc.)
/// come from `provider_types.gcp_vertex_gemini.sft` in the gateway config.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct GCPVertexGeminiSFTConfig {
    pub model: String,
    pub learning_rate_multiplier: Option<f64>,
    pub adapter_size: Option<usize>,
    pub n_epochs: Option<usize>,
    pub export_last_checkpoint_only: Option<bool>,
    pub seed: Option<u64>,
    pub tuned_model_display_name: Option<String>,
}

/// Uninitialized GCP Vertex Gemini SFT Config (per-job settings only).
/// Provider-level settings (project_id, region, bucket_name, credentials, etc.)
/// come from `provider_types.gcp_vertex_gemini.sft` in the gateway config.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "GCPVertexGeminiSFTConfig"))]
pub struct UninitializedGCPVertexGeminiSFTConfig {
    pub model: String,
    pub learning_rate_multiplier: Option<f64>,
    pub adapter_size: Option<usize>,
    pub n_epochs: Option<usize>,
    pub export_last_checkpoint_only: Option<bool>,
    pub seed: Option<u64>,
    pub tuned_model_display_name: Option<String>,
}

impl std::fmt::Display for UninitializedGCPVertexGeminiSFTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UninitializedGCPVertexGeminiSFTConfig {
    /// Initialize the GCPVertexGeminiSFTConfig.
    ///
    /// Provider-level settings (project_id, region, bucket_name, credentials, service_account,
    /// kms_key_name, bucket_path_prefix) are configured in the gateway config at
    /// `[provider_types.gcp_vertex_gemini.sft]`.
    ///
    /// :param model: The model to use for the fine-tuning job (required).
    /// :param learning_rate_multiplier: The learning rate multiplier for the fine-tuning job.
    /// :param adapter_size: The adapter size for the fine-tuning job.
    /// :param n_epochs: The number of epochs for the fine-tuning job.
    /// :param export_last_checkpoint_only: Whether to export the last checkpoint only.
    /// :param seed: The seed for the fine-tuning job.
    /// :param tuned_model_display_name: The display name for the tuned model.
    #[new]
    #[pyo3(signature = (*, model, learning_rate_multiplier=None, adapter_size=None, n_epochs=None, export_last_checkpoint_only=None, seed=None, tuned_model_display_name=None))]
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        _py: Python<'_>,
        model: String,
        learning_rate_multiplier: Option<f64>,
        adapter_size: Option<usize>,
        n_epochs: Option<usize>,
        export_last_checkpoint_only: Option<bool>,
        seed: Option<u64>,
        tuned_model_display_name: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            model,
            learning_rate_multiplier,
            adapter_size,
            n_epochs,
            export_last_checkpoint_only,
            seed,
            tuned_model_display_name,
        })
    }

    #[expect(unused_variables)]
    #[pyo3(signature = (*, model, learning_rate_multiplier=None, adapter_size=None, n_epochs=None, export_last_checkpoint_only=None, seed=None, tuned_model_display_name=None))]
    #[expect(clippy::too_many_arguments)]
    fn __init__(
        this: Py<Self>,
        model: String,
        learning_rate_multiplier: Option<f64>,
        adapter_size: Option<usize>,
        n_epochs: Option<usize>,
        export_last_checkpoint_only: Option<bool>,
        seed: Option<u64>,
        tuned_model_display_name: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedGCPVertexGeminiSFTConfig {
    pub fn load(self) -> GCPVertexGeminiSFTConfig {
        GCPVertexGeminiSFTConfig {
            model: self.model,
            learning_rate_multiplier: self.learning_rate_multiplier,
            adapter_size: self.adapter_size,
            n_epochs: self.n_epochs,
            export_last_checkpoint_only: self.export_last_checkpoint_only,
            seed: self.seed,
            tuned_model_display_name: self.tuned_model_display_name,
        }
    }
}

/// Minimal job handle for GCP Vertex Gemini SFT.
/// All configuration needed for polling comes from provider_types at poll time.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct GCPVertexGeminiSFTJobHandle {
    pub job_url: Url,
    /// The API resource name (e.g., projects/{project}/locations/{region}/tuningJobs/{job_id})
    pub job_name: String,
}

impl std::fmt::Display for GCPVertexGeminiSFTJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}
