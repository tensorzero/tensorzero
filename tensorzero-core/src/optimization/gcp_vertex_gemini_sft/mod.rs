use serde::{Deserialize, Serialize};
use url::Url;

use crate::{
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    model::{build_creds_caching_default, CredentialLocation},
    optimization::{JobHandle, Optimizer, OptimizerStatus},
    providers::gcp_vertex_gemini::{
        default_api_key_location,
        optimization::{
            convert_to_optimizer_status, EncryptionSpec, GCPVertexGeminiFineTuningJob,
            GCPVertexGeminiFineTuningRequest, SupervisedHyperparameters, SupervisedTuningSpec,
        },
        upload_rows_to_gcp_object_store, GCPVertexCredentials, GCPVertexGeminiSupervisedRow,
        DEFAULT_CREDENTIALS, PROVIDER_TYPE,
    },
    stored_inference::RenderedSample,
};

pub fn gcp_vertex_gemini_url(
    project_id: &str,
    region: &str,
    job_id: Option<&str>,
) -> Result<Url, url::ParseError> {
    match job_id {
        Some(id) => Url::parse(&format!(
            "https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/tuningJobs/{id}"
        )),
        None => Url::parse(&format!(
            "https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/tuningJobs"
        )),
    }
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GCPVertexGeminiSFTConfig {
    pub model: String,
    pub learning_rate_multiplier: Option<f64>,
    pub adapter_size: Option<usize>,
    pub n_epochs: Option<usize>,
    pub export_last_checkpoint_only: Option<bool>,
    #[serde(skip)]
    pub credentials: GCPVertexCredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocation>,
    pub seed: Option<u64>,
    pub api_base: Option<Url>,
    pub service_account: Option<String>,
    pub kms_key_name: Option<String>,
    pub tuned_model_display_name: Option<String>,
    pub bucket_name: String,
    pub bucket_path_prefix: Option<String>, // e.g., "fine-tuning/data" (optional)
    pub project_id: String,
    pub region: String,
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize)]
#[cfg_attr(test, ts(export))]
pub struct UninitializedGCPVertexGeminiSFTConfig {
    pub model: String,
    pub learning_rate_multiplier: Option<f64>,
    pub adapter_size: Option<usize>,
    pub n_epochs: Option<usize>,
    pub export_last_checkpoint_only: Option<bool>,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credentials: Option<CredentialLocation>,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocation>,
    pub api_base: Option<Url>,
    pub seed: Option<u64>,
    pub service_account: Option<String>,
    pub kms_key_name: Option<String>,
    pub tuned_model_display_name: Option<String>,
    pub bucket_name: String,
    pub bucket_path_prefix: Option<String>, // e.g., "fine-tuning/data" (optional)
    pub project_id: String,
    pub region: String,
}

impl UninitializedGCPVertexGeminiSFTConfig {
    pub fn load(self) -> Result<GCPVertexGeminiSFTConfig, Error> {
        Ok(GCPVertexGeminiSFTConfig {
            model: self.model,
            api_base: self.api_base,
            learning_rate_multiplier: self.learning_rate_multiplier,
            adapter_size: self.adapter_size,
            n_epochs: self.n_epochs,
            export_last_checkpoint_only: self.export_last_checkpoint_only,
            credentials: build_creds_caching_default(
                self.credentials.clone(),
                default_api_key_location(),
                PROVIDER_TYPE,
                &DEFAULT_CREDENTIALS,
            )?,
            credential_location: self.credential_location,
            seed: self.seed,
            service_account: self.service_account,
            kms_key_name: self.kms_key_name,
            tuned_model_display_name: self.tuned_model_display_name,
            bucket_name: self.bucket_name,
            bucket_path_prefix: self.bucket_path_prefix,
            project_id: self.project_id,
            region: self.region,
        })
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(test, ts(export))]
pub struct GCPVertexGeminiSFTJobHandle {
    pub job_id: String,
    pub job_url: Url,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocation>,
    pub region: String,
    pub project_id: String,
}

impl Optimizer for GCPVertexGeminiSFTConfig {
    type Handle = GCPVertexGeminiSFTJobHandle;

    async fn launch(
        &self,
        client: &reqwest::Client,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
    ) -> Result<Self::Handle, Error> {
        // TODO(#2642): improve error handling here so we know what index of example failed
        let train_rows: Vec<GCPVertexGeminiSupervisedRow> = train_examples
            .iter()
            .map(GCPVertexGeminiSupervisedRow::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        let val_rows: Option<Vec<GCPVertexGeminiSupervisedRow>> = val_examples
            .as_ref()
            .map(|examples| {
                examples
                    .iter()
                    .map(GCPVertexGeminiSupervisedRow::try_from)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;

        let train_filename = format!("train_{}.jsonl", uuid::Uuid::now_v7()); // or use job ID
        let train_gs_url = match &self.bucket_path_prefix {
            Some(prefix) => format!("gs://{}/{}/{}", self.bucket_name, prefix, train_filename),
            None => format!("gs://{}/{}", self.bucket_name, train_filename),
        };
        upload_rows_to_gcp_object_store(&train_rows, &train_gs_url, &self.credentials, credentials)
            .await?;

        // Upload validation data if provided
        let val_gs_url = if let Some(val_rows) = &val_rows {
            let val_filename = format!("val_{}.jsonl", uuid::Uuid::now_v7());
            let val_url = match &self.bucket_path_prefix {
                Some(prefix) => format!("gs://{}/{}/{}", self.bucket_name, prefix, val_filename),
                None => format!("gs://{}/{}", self.bucket_name, val_filename),
            };

            upload_rows_to_gcp_object_store(val_rows, &val_url, &self.credentials, credentials)
                .await?;
            Some(val_url)
        } else {
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

        let encryption_spec = EncryptionSpec {
            kms_key_name: self.kms_key_name.clone(),
        };

        let body = GCPVertexGeminiFineTuningRequest {
            base_model: self.model.clone(),
            supervised_tuning_spec,
            tuned_model_display_name: self.tuned_model_display_name.clone(),
            service_account: self.service_account.clone(),
            encryption_spec: Some(encryption_spec),
        };

        let url = gcp_vertex_gemini_url(&self.project_id, &self.region, None).map_err(|e| {
            Error::new(ErrorDetails::InvalidBaseUrl {
                message: e.to_string(),
            })
        })?;

        let auth_headers = self
            .credentials
            .get_auth_headers(
                &format!("https://{}-aiplatform.googleapis.com/", self.region),
                credentials,
            )
            .await?;

        let request = client.post(url).headers(auth_headers);
        let res = request.json(&body).send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error sending request to OpenAI: {}",
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
        let job_id = job
            .name
            .split('/')
            .next_back()
            .ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "Invalid job name format in response".to_string(),
                    raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?
            .to_string();
        let job_url = gcp_vertex_gemini_url(&self.project_id, &self.region, Some(&job_id))
            .map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to parse job URL: {e}"),
                })
            })?;
        Ok(GCPVertexGeminiSFTJobHandle {
            job_id,
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
        client: &reqwest::Client,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizerStatus, Error> {
        let gcp_credentials = build_creds_caching_default(
            self.credential_location.clone(),
            default_api_key_location(),
            PROVIDER_TYPE,
            &DEFAULT_CREDENTIALS,
        )?;

        let auth_headers = gcp_credentials
            .get_auth_headers(
                &format!("https://{}-aiplatform.googleapis.com/", self.region),
                credentials,
            )
            .await?;

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
            self.credential_location
                .clone()
                .unwrap_or(default_api_key_location()),
        )
    }
}
