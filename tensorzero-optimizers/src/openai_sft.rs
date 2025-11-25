//! OpenAI Supervised Fine-Tuning (SFT) optimizer implementation

use async_trait::async_trait;
use futures::future::try_join_all;
use secrecy::ExposeSecret;
use std::sync::Arc;
use tokio::try_join;
use url::Url;

use tensorzero_core::{
    config::Config,
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    http::TensorzeroHttpClient,
    model_table::{OpenAIKind, ProviderKind, ProviderTypeDefaultCredentials},
    optimization::{
        openai_sft::{OpenAISFTConfig, OpenAISFTJobHandle},
        OptimizationJobInfo,
    },
    providers::openai::{
        upload_openai_file, OpenAICredentials, OPENAI_DEFAULT_BASE_URL, PROVIDER_TYPE,
    },
    stored_inference::RenderedSample,
};

use crate::{
    openai::{
        convert_to_optimizer_status, OpenAIFineTuningJob, OpenAIFineTuningMethod,
        OpenAIFineTuningRequest, OpenAISupervisedRow, Supervised, SupervisedHyperparameters,
    },
    JobHandle, Optimizer,
};

const OPENAI_FINE_TUNE_PURPOSE: &str = "fine-tune";

#[async_trait]
impl Optimizer for OpenAISFTConfig {
    type Handle = OpenAISFTJobHandle;

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
        let train_rows: Vec<OpenAISupervisedRow> = try_join_all(
            train_examples
                .iter()
                .map(OpenAISupervisedRow::from_rendered_sample),
        )
        .await?;

        let val_rows = if let Some(examples) = val_examples.as_ref() {
            Some(
                try_join_all(
                    examples
                        .iter()
                        .map(OpenAISupervisedRow::from_rendered_sample),
                )
                .await?,
            )
        } else {
            None
        };

        let api_key = self
            .credentials
            .get_api_key(credentials)
            .map_err(|e| e.log())?;

        // Run these concurrently
        let api_base = self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL);

        let train_fut = upload_openai_file(
            &train_rows,
            client,
            api_key,
            api_base,
            OPENAI_FINE_TUNE_PURPOSE.to_string(),
        );

        let (train_file_id, val_file_id) = if let Some(val_rows) = val_rows.as_ref() {
            let val_fut = upload_openai_file(
                val_rows,
                client,
                api_key,
                api_base,
                OPENAI_FINE_TUNE_PURPOSE.to_string(),
            );

            // Run both futures concurrently
            let (train_id, val_id) = try_join!(train_fut, val_fut)?;
            (train_id, Some(val_id))
        } else {
            // Just run the training file upload
            let train_file_id = train_fut.await?;
            (train_file_id, None)
        };

        let body = OpenAIFineTuningRequest {
            model: self.model.clone(),
            training_file: train_file_id,
            validation_file: val_file_id,
            method: OpenAIFineTuningMethod::Supervised {
                supervised: Supervised {
                    hyperparameters: Some(SupervisedHyperparameters {
                        batch_size: self.batch_size,
                        learning_rate_multiplier: self.learning_rate_multiplier,
                        n_epochs: self.n_epochs,
                    }),
                },
            },
            seed: self.seed,
            suffix: self.suffix.clone(),
            metadata: None,
        };

        let url = get_fine_tuning_url(
            self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL),
            None,
        )?;
        let mut request = client.post(url);
        if let Some(api_key) = api_key {
            request = request.bearer_auth(api_key.expose_secret());
        }
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
                    "Error sending request to OpenAI: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: None,
            })
        })?;
        let job: OpenAIFineTuningJob = serde_json::from_str(&raw_response).map_err(|e| {
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
        let job_api_url = get_fine_tuning_url(
            self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL),
            Some(&job.id),
        )?;
        Ok(OpenAISFTJobHandle {
            job_id: job.id.clone(),
            job_url: format!("https://platform.openai.com/finetune/{}", job.id)
                .parse()
                .map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!(
                            "Failed to construct job url: {e}. {IMPOSSIBLE_ERROR_MESSAGE}"
                        ),
                    })
                })?,
            job_api_url,
            credential_location: self.credential_location.clone(),
        })
    }
}

#[async_trait]
impl JobHandle for OpenAISFTJobHandle {
    async fn poll(
        &self,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OptimizationJobInfo, Error> {
        let openai_credentials: OpenAICredentials = OpenAIKind
            .get_defaulted_credential(self.credential_location.as_ref(), default_credentials)
            .await?;
        let mut request = client.get(self.job_api_url.clone());
        let api_key = openai_credentials
            .get_api_key(credentials)
            .map_err(|e| e.log())?;
        if let Some(api_key) = api_key {
            request = request.bearer_auth(api_key.expose_secret());
        }
        let res = request.send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error sending request to OpenAI: {}",
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
                    "Error sending request to OpenAI: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
        let job: OpenAIFineTuningJob = serde_json::from_str(&raw_response).map_err(|e| {
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
        convert_to_optimizer_status(job)
    }
}

fn get_fine_tuning_url(base_url: &Url, job_id: Option<&str>) -> Result<Url, Error> {
    let mut url = base_url.clone();
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    let path = if let Some(id) = job_id {
        format!("fine_tuning/jobs/{id}")
    } else {
        "fine_tuning/jobs".to_string()
    };
    url.join(&path).map_err(|e| {
        Error::new(ErrorDetails::InvalidBaseUrl {
            message: e.to_string(),
        })
    })
}
