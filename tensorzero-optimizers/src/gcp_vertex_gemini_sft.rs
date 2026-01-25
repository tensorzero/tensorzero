//! GCP Vertex Gemini Supervised Fine-Tuning (SFT) optimizer implementation

use futures::{future::try_join_all, try_join};
use std::sync::Arc;
use url::Url;
use uuid::Uuid;

use tensorzero_core::{
    config::{Config, provider_types::GCPSFTConfig, provider_types::ProviderTypesConfig},
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    http::TensorzeroHttpClient,
    model_table::{GCPVertexGeminiKind, ProviderTypeDefaultCredentials},
    optimization::{
        OptimizationJobInfo,
        gcp_vertex_gemini_sft::{GCPVertexGeminiSFTConfig, GCPVertexGeminiSFTJobHandle},
    },
    providers::gcp_vertex_gemini::{
        GCPVertexGeminiSupervisedRow, PROVIDER_TYPE, location_subdomain_prefix,
        optimization::{
            EncryptionSpec, GCPVertexGeminiFineTuningJob, GCPVertexGeminiFineTuningRequest,
            SupervisedHyperparameters, SupervisedTuningSpec, convert_to_optimizer_status,
        },
        upload_rows_to_gcp_object_store,
    },
    stored_inference::RenderedSample,
    utils::mock::{get_mock_provider_api_base, is_mock_mode},
};

use crate::{JobHandle, Optimizer};

pub fn gcp_vertex_gemini_base_url(project_id: &str, region: &str) -> Result<Url, url::ParseError> {
    let subdomain_prefix = location_subdomain_prefix(region);
    Url::parse(&format!(
        "https://{subdomain_prefix}aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/tuningJobs"
    ))
}

fn get_sft_config(provider_types: &ProviderTypesConfig) -> Result<&GCPSFTConfig, Error> {
    provider_types
        .gcp_vertex_gemini
        .sft
        .as_ref()
        .ok_or_else(|| {
            Error::new(ErrorDetails::InvalidRequest {
                message: "GCP Vertex Gemini SFT requires `[provider_types.gcp_vertex_gemini.sft]` configuration section".to_string(),
            })
        })
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
        config: Arc<Config>,
    ) -> Result<Self::Handle, Error> {
        // Get provider-level config
        let sft_config = get_sft_config(&config.provider_types)?;

        // Check if we're in mock mode (TENSORZERO_INTERNAL_MOCK_PROVIDER_API is set)
        let mock_mode = is_mock_mode();

        // Get credentials from provider defaults (only needed in real mode)
        let gcp_credentials = if mock_mode {
            None
        } else {
            Some(
                GCPVertexGeminiKind
                    .get_defaulted_credential(None, &config.models.default_credentials)
                    .await?,
            )
        };

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

        let train_filename = format!("train_{}.jsonl", Uuid::now_v7());
        let train_gs_url = match &sft_config.bucket_path_prefix {
            Some(prefix) => format!(
                "gs://{}/{}/{}",
                sft_config.bucket_name, prefix, train_filename
            ),
            None => format!("gs://{}/{}", sft_config.bucket_name, train_filename),
        };

        // Upload to GCS (skip in mock mode - mock server ignores gs:// URLs)
        let val_gs_url = if let Some(gcp_creds) = &gcp_credentials {
            let train_fut =
                upload_rows_to_gcp_object_store(&train_rows, &train_gs_url, gcp_creds, credentials);

            if let Some(val_rows) = &val_rows {
                let val_filename = format!("val_{}.jsonl", Uuid::now_v7());
                let val_url = match &sft_config.bucket_path_prefix {
                    Some(prefix) => format!(
                        "gs://{}/{}/{}",
                        sft_config.bucket_name, prefix, val_filename
                    ),
                    None => format!("gs://{}/{}", sft_config.bucket_name, val_filename),
                };

                let val_fut =
                    upload_rows_to_gcp_object_store(val_rows, &val_url, gcp_creds, credentials);

                // Run both futures concurrently
                try_join!(train_fut, val_fut)?;
                Some(val_url)
            } else {
                // Just run the training file upload
                train_fut.await?;
                None
            }
        } else {
            // Mock mode: skip uploads, use placeholder URL for validation if needed
            val_rows
                .as_ref()
                .map(|_| format!("gs://{}/mock_validation.jsonl", sft_config.bucket_name))
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

        let encryption_spec = sft_config
            .kms_key_name
            .as_ref()
            .map(|kms_key| EncryptionSpec {
                kms_key_name: Some(kms_key.clone()),
            });

        let body = GCPVertexGeminiFineTuningRequest {
            base_model: self.model.clone(),
            supervised_tuning_spec,
            tuned_model_display_name: self.tuned_model_display_name.clone(),
            service_account: sft_config.service_account.clone(),
            encryption_spec,
        };

        // Build URL - use mock API base override for testing if available
        let url = if let Some(api_base) = get_mock_provider_api_base("gcp_vertex_gemini/") {
            api_base
                .join(&format!(
                    "v1/projects/{}/locations/{}/tuningJobs",
                    sft_config.project_id, sft_config.region
                ))
                .map_err(|e| {
                    Error::new(ErrorDetails::InvalidBaseUrl {
                        message: e.to_string(),
                    })
                })?
        } else {
            gcp_vertex_gemini_base_url(&sft_config.project_id, &sft_config.region).map_err(|e| {
                Error::new(ErrorDetails::InvalidBaseUrl {
                    message: e.to_string(),
                })
            })?
        };

        let request = if let Some(gcp_creds) = &gcp_credentials {
            let auth_headers = gcp_creds
                .get_auth_headers(
                    &format!(
                        "https://{}aiplatform.googleapis.com/",
                        location_subdomain_prefix(&sft_config.region)
                    ),
                    credentials,
                )
                .await
                .map_err(|e| e.log())?;
            client.post(url).headers(auth_headers)
        } else {
            // Mock mode: no auth headers needed
            client.post(url)
        };
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

        // Extract job ID from job.name (format: projects/{project}/locations/{region}/tuningJobs/{job_id})
        let job_id = job.name.rsplit('/').next().ok_or_else(|| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to extract job ID from job name: {}", job.name),
            })
        })?;
        let job_url = Url::parse(&format!(
            "https://console.cloud.google.com/vertex-ai/tuning/locations/{}/tuningJob/{}/monitor?project={}",
            sft_config.region, job_id, sft_config.project_id
        ))
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!("Failed to parse job URL: {e}"),
            })
        })?;

        Ok(GCPVertexGeminiSFTJobHandle {
            job_url,
            job_name: job.name,
        })
    }
}

impl JobHandle for GCPVertexGeminiSFTJobHandle {
    async fn poll(
        &self,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        default_credentials: &ProviderTypeDefaultCredentials,
        provider_types: &ProviderTypesConfig,
    ) -> Result<OptimizationJobInfo, Error> {
        // Get provider-level config
        let sft_config = get_sft_config(provider_types)?;

        // Check if we're in mock mode (TENSORZERO_INTERNAL_MOCK_PROVIDER_API is set)
        let mock_mode = is_mock_mode();

        // Construct the API URL from job_name
        let api_url = if let Some(api_base) = get_mock_provider_api_base("gcp_vertex_gemini/") {
            api_base
                .join(&format!("v1/{}", self.job_name))
                .map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!("Failed to parse API URL: {e}"),
                    })
                })?
        } else {
            Url::parse(&format!(
                "https://{}aiplatform.googleapis.com/v1/{}",
                location_subdomain_prefix(&sft_config.region),
                self.job_name
            ))
            .map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to parse API URL: {e}"),
                })
            })?
        };

        let request = if mock_mode {
            // Mock mode: no auth headers needed
            client.get(api_url)
        } else {
            let gcp_credentials = GCPVertexGeminiKind
                .get_defaulted_credential(None, default_credentials)
                .await?;

            let auth_headers = gcp_credentials
                .get_auth_headers(
                    &format!(
                        "https://{}aiplatform.googleapis.com/",
                        location_subdomain_prefix(&sft_config.region)
                    ),
                    credentials,
                )
                .await
                .map_err(|e| e.log())?;

            client.get(api_url).headers(auth_headers)
        };

        let res = request.send().await.map_err(|e| {
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

        convert_to_optimizer_status(job, sft_config)
    }
}
