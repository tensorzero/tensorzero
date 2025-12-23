#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::{
    error::Error,
    model::CredentialLocationWithFallback,
    model_table::{GCPVertexGeminiKind, ProviderTypeDefaultCredentials},
    providers::gcp_vertex_gemini::GCPVertexCredentials,
};

#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct GCPVertexGeminiSFTConfig {
    pub model: String,
    pub bucket_name: String,
    pub project_id: String,
    pub region: String,
    pub learning_rate_multiplier: Option<f64>,
    pub adapter_size: Option<usize>,
    pub n_epochs: Option<usize>,
    pub export_last_checkpoint_only: Option<bool>,
    #[serde(skip)]
    pub credentials: GCPVertexCredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocationWithFallback>,
    pub seed: Option<u64>,
    pub api_base: Option<Url>,
    pub service_account: Option<String>,
    pub kms_key_name: Option<String>,
    pub tuned_model_display_name: Option<String>,
    pub bucket_path_prefix: Option<String>, // e.g., "fine-tuning/data" (optional)
}

#[derive(ts_rs::TS, Clone, Debug, Default, Deserialize, Serialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "GCPVertexGeminiSFTConfig"))]
pub struct UninitializedGCPVertexGeminiSFTConfig {
    pub model: String,
    pub bucket_name: String,
    pub project_id: String,
    pub region: String,
    pub learning_rate_multiplier: Option<f64>,
    pub adapter_size: Option<usize>,
    pub n_epochs: Option<usize>,
    pub export_last_checkpoint_only: Option<bool>,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credentials: Option<CredentialLocationWithFallback>,
    pub api_base: Option<Url>,
    pub seed: Option<u64>,
    pub service_account: Option<String>,
    pub kms_key_name: Option<String>,
    pub tuned_model_display_name: Option<String>,
    pub bucket_path_prefix: Option<String>, // e.g., "fine-tuning/data" (optional)
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
    #[expect(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, model, bucket_name, project_id, region, learning_rate_multiplier=None, adapter_size=None, n_epochs=None, export_last_checkpoint_only=None, credentials=None, api_base=None, seed=None, service_account=None, kms_key_name=None, tuned_model_display_name=None, bucket_path_prefix=None))]
    pub fn new(
        _py: Python<'_>,
        model: String,
        bucket_name: String,
        project_id: String,
        region: String,
        learning_rate_multiplier: Option<f64>,
        adapter_size: Option<usize>,
        n_epochs: Option<usize>,
        export_last_checkpoint_only: Option<bool>,
        credentials: Option<String>,
        api_base: Option<String>,
        seed: Option<u64>,
        service_account: Option<String>,
        kms_key_name: Option<String>,
        tuned_model_display_name: Option<String>,
        bucket_path_prefix: Option<String>,
    ) -> PyResult<Self> {
        // Use Deserialize to convert the string to a CredentialLocation
        let credentials = credentials
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Invalid credentials JSON: {e}")))?;
        let api_base = api_base
            .map(|s| {
                Url::parse(&s)
                    .map_err(|e| PyErr::new::<PyValueError, std::string::String>(e.to_string()))
            })
            .transpose()?;
        Ok(Self {
            model,
            bucket_name,
            project_id,
            region,
            learning_rate_multiplier,
            adapter_size,
            n_epochs,
            export_last_checkpoint_only,
            credentials,
            api_base,
            seed,
            service_account,
            kms_key_name,
            tuned_model_display_name,
            bucket_path_prefix,
        })
    }

    /// Initialize the GCPVertexGeminiSFTConfig. All parameters are optional except for `model`, `bucket_name`, `project_id`, and `region`.
    ///
    /// :param model: The model to use for the fine-tuning job.
    /// :param bucket_name: The GCS bucket name to store training data.
    /// :param project_id: The GCP project ID where the fine-tuning job will run.
    /// :param region: The GCP region where the fine-tuning job will run.
    /// :param learning_rate_multiplier: The learning rate multiplier to use for the fine-tuning job.
    /// :param adapter_size: The adapter size to use for the fine-tuning job.
    /// :param n_epochs: The number of epochs to use for the fine-tuning job.
    /// :param export_last_checkpoint_only: Whether to export the last checkpoint only.
    /// :param credentials: The credentials to use for the fine-tuning job. This should be a string like `env::GCP_VERTEX_CREDENTIALS_PATH`. See docs for more details.
    /// :param api_base: The base URL to use for the fine-tuning job. This is primarily used for testing.
    /// :param seed: The seed to use for the fine-tuning job.
    /// :param service_account: The service account to use for the fine-tuning job.
    /// :param kms_key_name: The KMS key name to use for the fine-tuning job.
    /// :param tuned_model_display_name: The display name to use for the fine-tuning job.
    /// :param bucket_path_prefix: The bucket path prefix to use for the fine-tuning job.
    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, bucket_name, project_id, region, learning_rate_multiplier=None, adapter_size=None, n_epochs=None, export_last_checkpoint_only=None, credentials=None, api_base=None, seed=None, service_account=None, kms_key_name=None, tuned_model_display_name=None, bucket_path_prefix=None))]
    fn __init__(
        this: Py<Self>,
        model: String,
        bucket_name: String,
        project_id: String,
        region: String,
        learning_rate_multiplier: Option<f64>,
        adapter_size: Option<usize>,
        n_epochs: Option<usize>,
        export_last_checkpoint_only: Option<bool>,
        credentials: Option<String>,
        api_base: Option<String>,
        seed: Option<u64>,
        service_account: Option<String>,
        kms_key_name: Option<String>,
        tuned_model_display_name: Option<String>,
        bucket_path_prefix: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedGCPVertexGeminiSFTConfig {
    pub async fn load(
        self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<GCPVertexGeminiSFTConfig, Error> {
        Ok(GCPVertexGeminiSFTConfig {
            model: self.model,
            bucket_name: self.bucket_name,
            project_id: self.project_id,
            region: self.region,
            learning_rate_multiplier: self.learning_rate_multiplier,
            adapter_size: self.adapter_size,
            n_epochs: self.n_epochs,
            export_last_checkpoint_only: self.export_last_checkpoint_only,
            credentials: GCPVertexGeminiKind
                .get_defaulted_credential(self.credentials.as_ref(), default_credentials)
                .await?,
            credential_location: self.credentials,
            api_base: self.api_base,
            seed: self.seed,
            service_account: self.service_account,
            kms_key_name: self.kms_key_name,
            tuned_model_display_name: self.tuned_model_display_name,
            bucket_path_prefix: self.bucket_path_prefix,
        })
    }
}

#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct GCPVertexGeminiSFTJobHandle {
    pub job_url: Url,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocationWithFallback>,
    pub region: String,
    pub project_id: String,
}

impl std::fmt::Display for GCPVertexGeminiSFTJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}
