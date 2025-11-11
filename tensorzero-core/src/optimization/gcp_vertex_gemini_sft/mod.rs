use futures::{future::try_join_all, try_join};
#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use url::Url;
use uuid::Uuid;

use crate::{
    config::Config,
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    http::TensorzeroHttpClient,
    model::CredentialLocationWithFallback,
    model_table::{GCPVertexGeminiKind, ProviderTypeDefaultCredentials},
    optimization::{JobHandle, OptimizationJobInfo, Optimizer},
    providers::gcp_vertex_gemini::{
        location_subdomain_prefix,
        optimization::{
            convert_to_optimizer_status, EncryptionSpec, GCPVertexGeminiFineTuningJob,
            GCPVertexGeminiFineTuningRequest, SupervisedHyperparameters, SupervisedTuningSpec,
        },
        upload_rows_to_gcp_object_store, GCPVertexCredentials, GCPVertexGeminiSupervisedRow,
        PROVIDER_TYPE,
    },
    stored_inference::RenderedSample,
};

pub fn gcp_vertex_gemini_base_url(project_id: &str, region: &str) -> Result<Url, url::ParseError> {
    let subdomain_prefix = location_subdomain_prefix(region);
    Url::parse(&format!(
        "https://{subdomain_prefix}aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/tuningJobs"
    ))
}

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

impl Optimizer for GCPVertexGeminiSFTConfig {
    type Handle = GCPVertexGeminiSFTJobHandle;

    async fn launch(
        &self,
        client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
        _clickhouse_connection_info: &ClickHouseConnectionInfo,
        _config: Arc<Config>,
    ) -> Result<Self::Handle, Error> {
        let train_examples = train_examples
            .into_iter()
            .map(RenderedSample::into_lazy_rendered_sample)
            .collect::<Vec<_>>();
        let val_examples = val_examples.map(|examples| {
            examples
                .into_iter()
                .map(RenderedSample::into_lazy_rendered_sample)
                .collect::<Vec<_>>()
        });
        // TODO(#2642): improve error handling here so we know what index of example failed
        let train_rows: Vec<GCPVertexGeminiSupervisedRow> = try_join_all(
            train_examples
                .iter()
                .map(GCPVertexGeminiSupervisedRow::from_rendered_sample),
        )
        .await?;

        let val_rows = if let Some(examples) = val_examples.as_ref() {
            Some(
                try_join_all(
                    examples
                        .iter()
                        .map(GCPVertexGeminiSupervisedRow::from_rendered_sample),
                )
                .await?,
            )
        } else {
            None
        };

        let train_filename = format!("train_{}.jsonl", Uuid::now_v7()); // or use job ID
        let train_gs_url = match &self.bucket_path_prefix {
            Some(prefix) => format!("gs://{}/{}/{}", self.bucket_name, prefix, train_filename),
            None => format!("gs://{}/{}", self.bucket_name, train_filename),
        };
        // Run uploads concurrently
        let train_fut = upload_rows_to_gcp_object_store(
            &train_rows,
            &train_gs_url,
            &self.credentials,
            credentials,
        );

        let val_gs_url = if let Some(val_rows) = &val_rows {
            let val_filename = format!("val_{}.jsonl", Uuid::now_v7());
            let val_url = match &self.bucket_path_prefix {
                Some(prefix) => format!("gs://{}/{}/{}", self.bucket_name, prefix, val_filename),
                None => format!("gs://{}/{}", self.bucket_name, val_filename),
            };

            let val_fut =
                upload_rows_to_gcp_object_store(val_rows, &val_url, &self.credentials, credentials);

            // Run both futures concurrently
            try_join!(train_fut, val_fut)?;
            Some(val_url)
        } else {
            // Just run the training file upload
            train_fut.await?;
            None
        };

        let supervised_tuning_spec = SupervisedTuningSpec {
            training_dataset_uri: train_gs_url,
            validation_dataset_uri: val_gs_url,
            hyper_parameters: Some(SupervisedHyperparameters {
                epoch_count: self.n_epochs,
                adapter_size: self.adapter_size,
                learning_rate_multiplier: self.learning_rate_multiplier,
            }),
            export_last_checkpoint_only: self.export_last_checkpoint_only,
        };

        let encryption_spec = self.kms_key_name.as_ref().map(|kms_key| EncryptionSpec {
            kms_key_name: Some(kms_key.clone()),
        });

        let body = GCPVertexGeminiFineTuningRequest {
            base_model: self.model.clone(),
            supervised_tuning_spec,
            tuned_model_display_name: self.tuned_model_display_name.clone(),
            service_account: self.service_account.clone(),
            encryption_spec,
        };

        let url = gcp_vertex_gemini_base_url(&self.project_id, &self.region).map_err(|e| {
            Error::new(ErrorDetails::InvalidBaseUrl {
                message: e.to_string(),
            })
        })?;

        let auth_headers = self
            .credentials
            .get_auth_headers(
                &format!(
                    "https://{}aiplatform.googleapis.com/",
                    location_subdomain_prefix(&self.region)
                ),
                credentials,
            )
            .await
            .map_err(|e| e.log())?;

        let request = client.post(url).headers(auth_headers);
        let res = request.json(&body).send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error sending request to GCP Vertex Gemini: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: None,
            })
        })?;

        let raw_response = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error reading response from GCP Vertex Gemini: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: None,
            })
        })?;
        let job: GCPVertexGeminiFineTuningJob =
            serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
        let subdomain_prefix = location_subdomain_prefix(&self.region);
        let job_url = Url::parse(&format!(
            "https://{subdomain_prefix}aiplatform.googleapis.com/v1/{}",
            job.name
        ))
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to parse job URL: {e}"),
            })
        })?;

        Ok(GCPVertexGeminiSFTJobHandle {
            job_url,
            credential_location: self.credential_location.clone(),
            region: self.region.clone(),
            project_id: self.project_id.clone(),
        })
    }
}

impl JobHandle for GCPVertexGeminiSFTJobHandle {
    async fn poll(
        &self,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OptimizationJobInfo, Error> {
        let gcp_credentials = crate::model_table::GCPVertexGeminiKind
            .get_defaulted_credential(self.credential_location.as_ref(), default_credentials)
            .await?;

        let auth_headers = gcp_credentials
            .get_auth_headers(
                &format!(
                    "https://{}aiplatform.googleapis.com/",
                    location_subdomain_prefix(&self.region)
                ),
                credentials,
            )
            .await
            .map_err(|e| e.log())?;

        // Use the stored job_url directly (it was already constructed with the helper)
        let res = client
            .get(self.job_url.clone())
            .headers(auth_headers)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!(
                        "Error sending request to GCP Vertex Gemini: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: None,
                })
            })?;

        let raw_response = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error reading response from GCP Vertex Gemini: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;

        let job: GCPVertexGeminiFineTuningJob =
            serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: None,
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

        convert_to_optimizer_status(
            job,
            self.region.clone(),
            self.project_id.clone(),
            self.credential_location.clone(),
        )
    }
}
