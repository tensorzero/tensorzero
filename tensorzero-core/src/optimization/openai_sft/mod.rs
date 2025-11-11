use crate::{
    error::IMPOSSIBLE_ERROR_MESSAGE,
    http::TensorzeroHttpClient,
    model_table::{OpenAIKind, ProviderKind, ProviderTypeDefaultCredentials},
};
use futures::future::try_join_all;
#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::try_join;
use url::Url;

use crate::{
    config::Config,
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    model::CredentialLocationWithFallback,
    optimization::{JobHandle, OptimizationJobInfo, Optimizer},
    providers::openai::{
        optimization::{
            convert_to_optimizer_status, OpenAIFineTuningJob, OpenAIFineTuningMethod,
            OpenAIFineTuningRequest, OpenAISupervisedRow, Supervised, SupervisedHyperparameters,
        },
        upload_openai_file, OpenAICredentials, OPENAI_DEFAULT_BASE_URL, PROVIDER_TYPE,
    },
    stored_inference::RenderedSample,
};

const OPENAI_FINE_TUNE_PURPOSE: &str = "fine-tune";

#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OpenAISFTConfig {
    pub model: String,
    pub batch_size: Option<usize>,
    pub learning_rate_multiplier: Option<f64>,
    pub n_epochs: Option<usize>,
    #[serde(skip)]
    pub credentials: OpenAICredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocationWithFallback>,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
    pub api_base: Option<Url>,
}

#[derive(ts_rs::TS, Clone, Debug, Default, Deserialize, Serialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "OpenAISFTConfig"))]
pub struct UninitializedOpenAISFTConfig {
    pub model: String,
    pub batch_size: Option<usize>,
    pub learning_rate_multiplier: Option<f64>,
    pub n_epochs: Option<usize>,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credentials: Option<CredentialLocationWithFallback>,
    pub api_base: Option<Url>,
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
    // We allow too many arguments since it is a Python constructor
    /// NOTE: This signature currently does not work:
    /// print(OpenAISFTConfig.__init__.__text_signature__)
    /// prints out signature:
    /// ($self, /, *args, **kwargs)
    /// Same is true for FireworksSFTConfig
    #[expect(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, model, batch_size=None, learning_rate_multiplier=None, n_epochs=None, credentials=None, api_base=None, seed=None, suffix=None))]
    pub fn new(
        model: String,
        batch_size: Option<usize>,
        learning_rate_multiplier: Option<f64>,
        n_epochs: Option<usize>,
        credentials: Option<String>,
        api_base: Option<String>,
        seed: Option<u64>,
        suffix: Option<String>,
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
            batch_size,
            learning_rate_multiplier,
            n_epochs,
            credentials,
            api_base,
            seed,
            suffix,
        })
    }

    /// Initialize the OpenAISFTConfig. All parameters are optional except for `model`.
    ///
    /// :param model: The model to use for the fine-tuning job.
    /// :param batch_size: The batch size to use for the fine-tuning job.
    /// :param learning_rate_multiplier: The learning rate multiplier to use for the fine-tuning job.
    /// :param n_epochs: The number of epochs to use for the fine-tuning job.
    /// :param credentials: The credentials to use for the fine-tuning job. This should be a string like `env::OPENAI_API_KEY`. See docs for more details.
    /// :param api_base: The base URL to use for the fine-tuning job. This is primarily used for testing.
    /// :param seed: The seed to use for the fine-tuning job.
    /// :param suffix: The suffix to use for the fine-tuning job (this is for naming in OpenAI).
    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, batch_size=None, learning_rate_multiplier=None, n_epochs=None, credentials=None, api_base=None, seed=None, suffix=None))]
    fn __init__(
        this: Py<Self>,
        model: String,
        batch_size: Option<usize>,
        learning_rate_multiplier: Option<f64>,
        n_epochs: Option<usize>,
        credentials: Option<String>,
        api_base: Option<String>,
        seed: Option<u64>,
        suffix: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedOpenAISFTConfig {
    pub async fn load(
        self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OpenAISFTConfig, Error> {
        Ok(OpenAISFTConfig {
            model: self.model,
            api_base: self.api_base,
            batch_size: self.batch_size,
            learning_rate_multiplier: self.learning_rate_multiplier,
            n_epochs: self.n_epochs,
            credentials: OpenAIKind
                .get_defaulted_credential(self.credentials.as_ref(), default_credentials)
                .await?,
            credential_location: self.credentials,
            seed: self.seed,
            suffix: self.suffix,
        })
    }
}

#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct OpenAISFTJobHandle {
    pub job_id: String,
    /// A url to a human-readable page for the job.
    pub job_url: Url,
    pub job_api_url: Url,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocationWithFallback>,
}

impl std::fmt::Display for OpenAISFTJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

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
