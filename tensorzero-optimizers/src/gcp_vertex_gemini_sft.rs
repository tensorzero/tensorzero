//! GCP Vertex Gemini Supervised Fine-Tuning (SFT) optimizer implementation

use async_trait::async_trait;
use futures::{future::try_join_all, try_join};
use std::sync::Arc;
use url::Url;
use uuid::Uuid;

use tensorzero_core::{
    config::{snapshot::SnapshotHash, Config},
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    http::TensorzeroHttpClient,
    model_table::{GCPVertexGeminiKind, ProviderTypeDefaultCredentials},
    optimization::{
        gcp_vertex_gemini_sft::{GCPVertexGeminiSFTConfig, GCPVertexGeminiSFTJobHandle},
        OptimizationJobInfo,
    },
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

use crate::{JobHandle, Optimizer};

pub fn gcp_vertex_gemini_base_url(project_id: &str, region: &str) -> Result<Url, url::ParseError> {
    let subdomain_prefix = location_subdomain_prefix(region);
    Url::parse(&format!(
        "https://{subdomain_prefix}aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/tuningJobs"
    ))
}

#[async_trait]
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

#[async_trait]
impl JobHandle for GCPVertexGeminiSFTJobHandle {
    async fn poll(
        &self,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OptimizationJobInfo, Error> {
        let gcp_credentials: GCPVertexCredentials = GCPVertexGeminiKind
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
