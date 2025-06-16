use secrecy::ExposeSecret;
use serde::Deserialize;
use url::Url;

use crate::{
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    inference::providers::openai::{
        default_api_key_location, upload_openai_file, OpenAICredentials, DEFAULT_CREDENTIALS,
        OPENAI_DEFAULT_BASE_URL, PROVIDER_TYPE,
    },
    model::{build_creds_caching_default, CredentialLocation},
    optimization::{
        providers::openai::{
            OpenAIFineTuningJob, OpenAIFineTuningMethod, OpenAIFineTuningRequest,
            OpenAISupervisedRow, SupervisedHyperparameters,
        },
        Optimizer, OptimizerStatus,
    },
    stored_inference::RenderedStoredInference,
};

// TODO: consolidate providers into src/providers/openai/inference.rs and so on
// so that internal types and logic can be privately shared

#[derive(Debug, Clone)]
pub struct OpenAISFTConfig {
    model: String,
    batch_size: Option<usize>,
    learning_rate_multiplier: Option<f64>,
    n_epochs: Option<usize>,
    credentials: OpenAICredentials,
    seed: Option<u64>,
    suffix: Option<String>,
    api_base: Option<Url>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UninitializedOpenAISFTConfig {
    model: String,
    batch_size: Option<usize>,
    learning_rate_multiplier: Option<f64>,
    n_epochs: Option<usize>,
    credentials: Option<CredentialLocation>,
    api_base: Option<Url>,
}

impl UninitializedOpenAISFTConfig {
    pub fn load(self) -> Result<OpenAISFTConfig, Error> {
        Ok(OpenAISFTConfig {
            model: self.model,
            api_base: self.api_base,
            batch_size: self.batch_size,
            learning_rate_multiplier: self.learning_rate_multiplier,
            n_epochs: self.n_epochs,
            credentials: build_creds_caching_default(
                self.credentials,
                default_api_key_location(),
                PROVIDER_TYPE,
                &DEFAULT_CREDENTIALS,
            )?,
            seed: self.seed,
            suffix: self.suffix,
        })
    }
}

pub struct OpenAISFTJobHandle {
    pub job_id: String,
}

impl Optimizer for OpenAISFTConfig {
    type JobHandle = OpenAISFTJobHandle;

    async fn launch(
        &self,
        client: &reqwest::Client,
        train_examples: Vec<RenderedStoredInference>,
        val_examples: Option<Vec<RenderedStoredInference>>,
        credentials: &InferenceCredentials,
    ) -> Result<Self::JobHandle, Error> {
        let train_rows: Vec<OpenAISupervisedRow> = train_examples
            .into_iter()
            .map(|example| example.into())
            .collect();

        let val_rows: Option<Vec<OpenAISupervisedRow>> = val_examples
            .map(|examples| examples.into_iter().map(|example| example.into()).collect());

        let api_key = self.credentials.get_api_key(credentials)?;

        // TODO: run these concurrently
        let train_file_id = upload_openai_file(
            &train_rows,
            client,
            api_key,
            self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL),
            "sft".to_string(),
        )
        .await?;
        let val_file_id = if let Some(val_rows) = val_rows {
            Some(
                upload_openai_file(
                    &val_rows,
                    client,
                    api_key,
                    self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL),
                    "sft".to_string(),
                )
                .await?,
            )
        } else {
            None
        };

        let body = OpenAIFineTuningRequest {
            model: self.model.clone(),
            training_file: train_file_id,
            validation_file: val_file_id,
            method: OpenAIFineTuningMethod::Supervised {
                hyperparameters: Some(SupervisedHyperparameters {
                    batch_size: self.batch_size,
                    learning_rate_multiplier: self.learning_rate_multiplier,
                    n_epochs: self.n_epochs,
                }),
            },
            seed: self.seed,
            suffix: self.suffix,
            metadata: None,
        };

        let url = get_fine_tuning_url(self.api_base.as_ref().unwrap_or(&OPENAI_DEFAULT_BASE_URL))?;
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
        Ok(OpenAISFTJobHandle { job_id: job.id })
    }

    async fn poll(
        &self,
        client: &reqwest::Client,
        job_handle: Self::JobHandle,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizerStatus, Error> {
        todo!()
    }
}

fn get_fine_tuning_url(base_url: &Url) -> Result<Url, Error> {
    let mut url = base_url.clone();
    if !url.path().ends_with('/') {
        url.set_path(&format!("{}/", url.path()));
    }
    url.join("fine_tuning").map_err(|e| {
        Error::new(ErrorDetails::InvalidBaseUrl {
            message: e.to_string(),
        })
    })
}
