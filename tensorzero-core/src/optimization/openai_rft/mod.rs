#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::deserialize_from_pyobj;
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
            OpenAIFineTuningRequest, OpenAIGrader, OpenAIRFTResponseFormat, OpenAIReinforcementRow,
            Reinforcement, ReinforcementHyperparameters,
        },
        upload_openai_file, OpenAICredentials, OPENAI_DEFAULT_BASE_URL, PROVIDER_TYPE,
    },
    stored_inference::RenderedSample,
};

#[cfg(feature = "pyo3")]
use crate::model::CredentialLocation;

const OPENAI_FINE_TUNE_PURPOSE: &str = "fine-tune";

#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OpenAIRFTConfig {
    pub model: String,
    pub grader: OpenAIGrader,
    pub response_format: Option<OpenAIRFTResponseFormat>,
    pub batch_size: Option<usize>,
    pub compute_multiplier: Option<f64>,
    pub eval_interval: Option<usize>,
    pub eval_samples: Option<usize>,
    pub learning_rate_multiplier: Option<f64>,
    pub n_epochs: Option<usize>,
    pub reasoning_effort: Option<String>,
    #[serde(skip)]
    pub credentials: OpenAICredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocationWithFallback>,
    pub api_base: Option<Url>,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "OpenAIRFTConfig"))]
pub struct UninitializedOpenAIRFTConfig {
    pub model: String,
    pub grader: OpenAIGrader,
    pub response_format: Option<OpenAIRFTResponseFormat>,
    pub batch_size: Option<usize>,
    pub compute_multiplier: Option<f64>,
    pub eval_interval: Option<usize>,
    pub eval_samples: Option<usize>,
    pub learning_rate_multiplier: Option<f64>,
    pub n_epochs: Option<usize>,
    pub reasoning_effort: Option<String>,
    #[serde(skip)]
    pub credentials: Option<CredentialLocationWithFallback>,
    pub api_base: Option<Url>,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
}

impl std::fmt::Display for UninitializedOpenAIRFTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UninitializedOpenAIRFTConfig {
    // We allow too many arguments since it is a Python constructor
    /// NOTE: This signature currently does not work:
    /// print(OpenAIRFTConfig.__init__.__text_signature__)
    /// prints out signature:
    /// ($self, /, *args, **kwargs)
    #[expect(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, model, grader, response_format=None, batch_size=None, compute_multiplier=None, eval_interval=None, eval_samples=None, learning_rate_multiplier=None, n_epochs=None, reasoning_effort=None, credentials=None, api_base=None, seed=None, suffix=None))]
    pub fn new(
        py: Python,
        model: String,
        grader: &Bound<'_, PyAny>,
        response_format: Option<&Bound<'_, PyAny>>,
        batch_size: Option<usize>,
        compute_multiplier: Option<f64>,
        eval_interval: Option<usize>,
        eval_samples: Option<usize>,
        learning_rate_multiplier: Option<f64>,
        n_epochs: Option<usize>,
        reasoning_effort: Option<String>,
        credentials: Option<String>,
        api_base: Option<String>,
        seed: Option<u64>,
        suffix: Option<String>,
    ) -> PyResult<Self> {
        // Deserialize the grader from Python dict to Rust OpenAIGrader
        let grader: OpenAIGrader = if let Ok(grader) = grader.extract::<OpenAIGrader>() {
            // If it's already a Grader object, use it directly
            grader
        } else {
            // Otherwise, try to deserialize from a Python dict
            deserialize_from_pyobj(py, grader)?
        };

        // Deserialize the response_format from Python dict to Rust OpenAIRFTResponseFormat
        let response_format: Option<OpenAIRFTResponseFormat> = if let Some(rf) = response_format {
            if let Ok(response_format) = rf.extract::<OpenAIRFTResponseFormat>() {
                // If it's already a ResponseFormat object, use it directly
                Some(response_format)
            } else {
                // Otherwise, try to deserialize from a Python dict
                Some(deserialize_from_pyobj(py, rf)?)
            }
        } else {
            None
        };

        // Use Deserialize to convert the string to a CredentialLocationWithFallback
        let credentials = credentials.map(|s| {
            serde_json::from_str(&s).unwrap_or(CredentialLocationWithFallback::Single(
                CredentialLocation::Env(s),
            ))
        });
        let api_base = api_base
            .map(|s| {
                Url::parse(&s)
                    .map_err(|e| PyErr::new::<PyValueError, std::string::String>(e.to_string()))
            })
            .transpose()?;
        Ok(Self {
            model,
            grader,
            response_format,
            batch_size,
            compute_multiplier,
            eval_interval,
            eval_samples,
            learning_rate_multiplier,
            n_epochs,
            reasoning_effort,
            credentials,
            api_base,
            seed,
            suffix,
        })
    }

    /// Initialize the OpenAISFTConfig. All parameters are optional except for `model`.
    ///
    /// :param model: The model to use for the reinforcement fine-tuning job.
    /// :param grader: The grader to use for the reinforcement fine-tuning job.
    /// :param response_format: The response format to use for the reinforcement fine-tuning job.
    /// :param batch_size: The batch size to use for the reinforcement fine-tuning job.
    /// :param compute_multiplier: The compute multiplier to use for the reinforcement fine-tuning job.
    /// :param eval_interval: The eval interval to use for the fine-tuning job.
    /// :param eval_samples: The eval samples to use for the fine-tuning job.
    /// :param batch_size: The batch size to use for the fine-tuning job.
    /// :param learning_rate_multiplier: The learning rate multiplier to use for the fine-tuning job.
    /// :param n_epochs: The number of epochs to use for the fine-tuning job.
    /// :param credentials: The credentials to use for the fine-tuning job. This should be a string like "env::OPENAI_API_KEY". See docs for more details.
    /// :param api_base: The base URL to use for the fine-tuning job. This is primarily used for testing.
    /// :param seed: The seed to use for the fine-tuning job.
    /// :param suffix: The suffix to use for the fine-tuning job (this is for naming in OpenAI).
    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, grader, response_format=None, batch_size=None, compute_multiplier=None, eval_interval=None, eval_samples=None, learning_rate_multiplier=None, n_epochs=None, reasoning_effort=None, credentials=None, api_base=None, seed=None, suffix=None))]
    fn __init__(
        this: Py<Self>,
        model: String,
        grader: OpenAIGrader,
        response_format: Option<OpenAIRFTResponseFormat>,
        batch_size: Option<usize>,
        compute_multiplier: Option<f64>,
        eval_interval: Option<usize>,
        eval_samples: Option<usize>,
        learning_rate_multiplier: Option<f64>,
        n_epochs: Option<usize>,
        reasoning_effort: Option<String>,
        credentials: Option<String>,
        api_base: Option<String>,
        seed: Option<u64>,
        suffix: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedOpenAIRFTConfig {
    pub async fn load(
        self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OpenAIRFTConfig, Error> {
        Ok(OpenAIRFTConfig {
            model: self.model,
            grader: self.grader,
            response_format: self.response_format,
            batch_size: self.batch_size,
            compute_multiplier: self.compute_multiplier,
            eval_interval: self.eval_interval,
            eval_samples: self.eval_samples,
            learning_rate_multiplier: self.learning_rate_multiplier,
            n_epochs: self.n_epochs,
            reasoning_effort: self.reasoning_effort,
            credentials: OpenAIKind
                .get_defaulted_credential(self.credentials.as_ref(), default_credentials)
                .await?,
            credential_location: self.credentials,
            api_base: self.api_base,
            suffix: self.suffix,
            seed: self.seed,
        })
    }
}

#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct OpenAIRFTJobHandle {
    pub job_id: String,
    /// A url to a human-readable page for the job.
    pub job_url: Url,
    pub job_api_url: Url,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocationWithFallback>,
}

impl std::fmt::Display for OpenAIRFTJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl Optimizer for OpenAIRFTConfig {
    type Handle = OpenAIRFTJobHandle;

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
        let train_rows: Vec<OpenAIReinforcementRow> = try_join_all(
            train_examples
                .iter()
                .map(OpenAIReinforcementRow::from_rendered_sample),
        )
        .await?;

        let val_rows = if let Some(examples) = val_examples.as_ref() {
            Some(
                try_join_all(
                    examples
                        .iter()
                        .map(OpenAIReinforcementRow::from_rendered_sample),
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
            method: OpenAIFineTuningMethod::Reinforcement {
                reinforcement: Reinforcement {
                    grader: Box::new(self.grader.clone()),
                    hyperparameters: Some(ReinforcementHyperparameters {
                        batch_size: self.batch_size,
                        compute_multiplier: self.compute_multiplier,
                        eval_interval: self.eval_interval,
                        eval_samples: self.eval_samples,
                        learning_rate_multiplier: self.learning_rate_multiplier,
                        n_epochs: self.n_epochs,
                        reasoning_effort: self.reasoning_effort.clone(),
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
                    "Error parsing OpenAI fine-tuning job response as JSON. Parse error: {}. Raw response: {}",
                    DisplayOrDebugGateway::new(e),
                    raw_response
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
        Ok(OpenAIRFTJobHandle {
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

impl JobHandle for OpenAIRFTJobHandle {
    async fn poll(
        &self,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OptimizationJobInfo, Error> {
        let openai_credentials: OpenAICredentials = OpenAIKind
            .get_defaulted_credential(None, default_credentials)
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
                    "Error parsing OpenAI fine-tuning job response as JSON. Parse error: {}. Raw response: {}",
                    DisplayOrDebugGateway::new(e),
                    raw_response
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
