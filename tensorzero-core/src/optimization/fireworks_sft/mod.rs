//! Fireworks SFT implementation. The overall flow is:
//! 1. `FireworksSFTConfig.launch` creates and uploads training/validation datasets to Fireworks
//! 2. We kick off a SFT job in Fireworks, and produce a `FireworksSFTJobHandle` with the job ID
//! 3. `FireworksSFTJobHandle.poll` performs the following checks (without maintaining any additional state):
//!    - If the job is still running, we return a `Pending` status
//!    - If the job has failed, we return a `Failed` status
//!    - If the job has completed, we look for an existing 'default' deployment for the model.
//!      If it exists, we return its status - otherwise, we start a new serverless deployment.
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Display;
use std::sync::Arc;
use std::time::Duration;

use futures::future::try_join_all;
use futures::try_join;
use http::StatusCode;
#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use reqwest::multipart::{Form, Part};
use secrecy::ExposeSecret;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use url::Url;
use uuid::Uuid;

use crate::config::{Config, TimeoutsConfig};
use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use crate::http::TensorzeroHttpClient;
use crate::model::UninitializedModelConfig;
use crate::model::UninitializedModelProvider;
use crate::model::UninitializedProviderConfig;
use crate::model_table::FireworksKind;
use crate::model_table::ProviderKind;
use crate::model_table::ProviderTypeDefaultCredentials;
use crate::optimization::JobHandle;
use crate::optimization::OptimizationJobInfo;
use crate::optimization::Optimizer;
use crate::optimization::OptimizerOutput;
use crate::providers::fireworks::prepare_fireworks_messages;
use crate::providers::fireworks::FIREWORKS_API_BASE;
use crate::providers::helpers::UrlParseErrExt;
use crate::providers::openai::tensorzero_to_openai_assistant_message;
use crate::providers::openai::OpenAIMessagesConfig;
use crate::stored_inference::LazyRenderedSample;
use crate::stored_inference::RenderedSample;
use crate::{
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails},
    inference::types::ContentBlock,
    model::CredentialLocationWithFallback,
    providers::{
        fireworks::{FireworksCredentials, FireworksTool, PROVIDER_TYPE},
        openai::OpenAIRequestMessage,
    },
};
use std::io::Write;

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FireworksFineTuningRequest {
    pub base_model: String,
    pub dataset: String,
    pub evaluation_dataset: Option<String>,
    pub early_stop: Option<bool>,
    pub epochs: Option<usize>,
    pub learning_rate: Option<f64>,
    pub max_context_length: Option<usize>,
    pub lora_rank: Option<usize>,
    pub batch_size: Option<usize>,
    pub display_name: Option<String>,
    pub output_model: Option<String>,
    pub warm_start_from: Option<String>,
    pub is_turbo: Option<bool>,
    pub eval_auto_carveout: Option<bool>,
    pub nodes: Option<usize>,
    pub mtp_enabled: Option<bool>,
    pub mtp_num_draft_tokens: Option<usize>,
    pub mtp_freeze_base_model: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct FireworksSupervisedRow<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<FireworksTool<'a>>,
}

impl<'a> FireworksSupervisedRow<'a> {
    pub async fn from_rendered_sample(inference: &'a LazyRenderedSample) -> Result<Self, Error> {
        if inference
            .tool_params
            .parallel_tool_calls
            .unwrap_or_default()
        {
            return Err(Error::new(ErrorDetails::InvalidRenderedStoredInference {
                message: "Parallel tool calls are not supported for Fireworks".to_string(),
            }));
        }
        let tools = inference
            .tool_params
            .additional_tools
            .as_ref()
            .map(|tools| tools.iter().map(Into::into).collect())
            .unwrap_or_default();
        let mut messages = prepare_fireworks_messages(
            inference.system_input.as_deref(),
            &inference.messages,
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                // For now, this isn't configurable in SFT (we should never need to resolve a file URL here)
                fetch_and_encode_input_files_before_inference: true,
            },
        )
        .await?;

        let Some(output) = &inference.output else {
            return Err(Error::new(ErrorDetails::InvalidRenderedStoredInference {
                message: "No output in inference".to_string(),
            }));
        };
        if output.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRenderedStoredInference {
                message: "No output in inference".to_string(),
            }));
        }
        let output_content_blocks: Vec<ContentBlock> =
            output.iter().map(|c| c.clone().into()).collect::<Vec<_>>();
        let final_assistant_message = tensorzero_to_openai_assistant_message(
            Cow::Owned(output_content_blocks),
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                // For now, this isn't configurable in SFT (we should never need to resolve a file URL here)
                fetch_and_encode_input_files_before_inference: true,
            },
        )
        .await?;
        messages.push(final_assistant_message);
        Ok(Self { messages, tools })
    }
}

#[derive(Debug, Clone, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct FireworksSFTConfig {
    pub model: String,
    pub early_stop: Option<bool>,
    pub epochs: Option<usize>,
    pub learning_rate: Option<f64>,
    pub max_context_length: Option<usize>,
    pub lora_rank: Option<usize>,
    pub batch_size: Option<usize>,
    pub display_name: Option<String>,
    pub output_model: Option<String>,
    pub warm_start_from: Option<String>,
    pub is_turbo: Option<bool>,
    pub eval_auto_carveout: Option<bool>,
    pub nodes: Option<usize>,
    pub mtp_enabled: Option<bool>,
    pub mtp_num_draft_tokens: Option<usize>,
    pub mtp_freeze_base_model: Option<bool>,
    #[serde(skip)]
    pub credentials: FireworksCredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocationWithFallback>,
    pub account_id: String,
    pub api_base: Url,
}

#[derive(ts_rs::TS, Clone, Debug, Default, Deserialize, Serialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "FireworksSFTConfig"))]
pub struct UninitializedFireworksSFTConfig {
    pub model: String,
    pub early_stop: Option<bool>,
    pub epochs: Option<usize>,
    pub learning_rate: Option<f64>,
    pub max_context_length: Option<usize>,
    pub lora_rank: Option<usize>,
    pub batch_size: Option<usize>,
    pub display_name: Option<String>,
    pub output_model: Option<String>,
    pub warm_start_from: Option<String>,
    pub is_turbo: Option<bool>,
    pub eval_auto_carveout: Option<bool>,
    pub nodes: Option<usize>,
    pub mtp_enabled: Option<bool>,
    pub mtp_num_draft_tokens: Option<usize>,
    pub mtp_freeze_base_model: Option<bool>,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credentials: Option<CredentialLocationWithFallback>,
    pub account_id: String,
    pub api_base: Option<Url>,
}

impl std::fmt::Display for UninitializedFireworksSFTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UninitializedFireworksSFTConfig {
    #[expect(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, model, early_stop=None, epochs=None, learning_rate=None, max_context_length=None, lora_rank=None, batch_size=None, display_name=None, output_model=None, warm_start_from=None, is_turbo=None, eval_auto_carveout=None, nodes=None, mtp_enabled=None, mtp_num_draft_tokens=None, mtp_freeze_base_model=None, credentials=None, account_id, api_base=None))]
    pub fn new(
        model: String,
        early_stop: Option<bool>,
        epochs: Option<usize>,
        learning_rate: Option<f64>,
        max_context_length: Option<usize>,
        lora_rank: Option<usize>,
        batch_size: Option<usize>,
        display_name: Option<String>,
        output_model: Option<String>,
        warm_start_from: Option<String>,
        is_turbo: Option<bool>,
        eval_auto_carveout: Option<bool>,
        nodes: Option<usize>,
        mtp_enabled: Option<bool>,
        mtp_num_draft_tokens: Option<usize>,
        mtp_freeze_base_model: Option<bool>,
        credentials: Option<String>,
        account_id: String,
        api_base: Option<String>,
    ) -> PyResult<Self> {
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
            early_stop,
            epochs,
            learning_rate,
            max_context_length,
            lora_rank,
            batch_size,
            display_name,
            output_model,
            warm_start_from,
            is_turbo,
            eval_auto_carveout,
            nodes,
            mtp_enabled,
            mtp_num_draft_tokens,
            mtp_freeze_base_model,
            credentials,
            account_id,
            api_base,
        })
    }

    /// Initialize the FireworksSFTConfig. All parameters are optional except for `model` and `account_id`.
    ///
    /// :param model: The model to use for the fine-tuning job.
    /// :param early_stop: Whether to early stop the fine-tuning job.
    /// :param epochs: The number of epochs to use for the fine-tuning job.
    /// :param learning_rate: The learning rate to use for the fine-tuning job.
    /// :param max_context_length: The maximum context length to use for the fine-tuning job.
    /// :param lora_rank: The rank of the LoRA matrix to use for the fine-tuning job.
    /// :param batch_size: The batch size to use for the fine-tuning job (tokens).
    /// :param display_name: The display name for the fine-tuning job.
    /// :param output_model: The model ID to be assigned to the resulting fine-tuned model. If not specified, the job ID will be used.
    /// :param warm_start_from: The PEFT addon model in Fireworks format to be fine-tuned from. Only one of 'model' or 'warm_start_from' should be specified.
    /// :param is_turbo: Whether to run the fine-tuning job in turbo mode.
    /// :param eval_auto_carveout: Whether to auto-carve the dataset for eval.
    /// :param nodes: The number of nodes to use for the fine-tuning job.
    /// :param mtp_enabled: Whether to enable MTP (Multi-Token Prediction).
    /// :param mtp_num_draft_tokens: The number of draft tokens for MTP.
    /// :param mtp_freeze_base_model: Whether to freeze the base model for MTP.
    /// :param credentials: The credentials to use for the fine-tuning job. This should be a string like `env::FIREWORKS_API_KEY`. See docs for more details.
    /// :param account_id: The account ID to use for the fine-tuning job.
    /// :param api_base: The base URL to use for the fine-tuning job. This is primarily used for testing.
    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, early_stop=None, epochs=None, learning_rate=None, max_context_length=None, lora_rank=None, batch_size=None, display_name=None, output_model=None, warm_start_from=None, is_turbo=None, eval_auto_carveout=None, nodes=None, mtp_enabled=None, mtp_num_draft_tokens=None, mtp_freeze_base_model=None, credentials=None, account_id, api_base=None))]
    fn __init__(
        this: Py<Self>,
        model: String,
        early_stop: Option<bool>,
        epochs: Option<usize>,
        learning_rate: Option<f64>,
        max_context_length: Option<usize>,
        lora_rank: Option<usize>,
        batch_size: Option<usize>,
        display_name: Option<String>,
        output_model: Option<String>,
        warm_start_from: Option<String>,
        is_turbo: Option<bool>,
        eval_auto_carveout: Option<bool>,
        nodes: Option<usize>,
        mtp_enabled: Option<bool>,
        mtp_num_draft_tokens: Option<usize>,
        mtp_freeze_base_model: Option<bool>,
        credentials: Option<String>,
        account_id: String,
        api_base: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedFireworksSFTConfig {
    pub async fn load(
        self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<FireworksSFTConfig, Error> {
        Ok(FireworksSFTConfig {
            model: self.model,
            early_stop: self.early_stop,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            max_context_length: self.max_context_length,
            lora_rank: self.lora_rank,
            batch_size: self.batch_size,
            display_name: self.display_name,
            output_model: self.output_model,
            warm_start_from: self.warm_start_from,
            is_turbo: self.is_turbo,
            eval_auto_carveout: self.eval_auto_carveout,
            nodes: self.nodes,
            mtp_enabled: self.mtp_enabled,
            mtp_num_draft_tokens: self.mtp_num_draft_tokens,
            mtp_freeze_base_model: self.mtp_freeze_base_model,
            api_base: self.api_base.unwrap_or_else(|| FIREWORKS_API_BASE.clone()),
            account_id: self.account_id,
            credentials: FireworksKind
                .get_defaulted_credential(self.credentials.as_ref(), default_credentials)
                .await?,
            credential_location: self.credentials,
        })
    }
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FireworksDatasetResponse {
    state: String,
}

impl FireworksSFTConfig {
    /// Polls the status of an existing dataset.
    /// Returns `true` if the dataset is in the `READY` state.
    async fn poll_dataset_read(
        &self,
        client: &TensorzeroHttpClient,
        api_key: &SecretString,
        dataset_id: &str,
    ) -> Result<bool, Error> {
        let res = client
            .get(
                self.api_base
                    .join(&format!(
                        "v1/accounts/{}/datasets/{}",
                        self.account_id, dataset_id
                    ))
                    .convert_parse_error()?,
            )
            .bearer_auth(api_key.expose_secret())
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!(
                        "Error checking dataset status: {}",
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
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
                message: format!(
                    "Error checking dataset status: {}",
                    DisplayOrDebugGateway::new(e)
                ),
            })
        })?;
        let response: FireworksDatasetResponse =
            serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    raw_request: None,
                    raw_response: Some(raw_response.clone()),
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
        Ok(response.state == "READY")
    }

    /// Produces a new dataset with a randomly-generated id, with the provided 'items' uploaded to it.
    /// Returns the Fireworks path to the dataset (e.g. `account/{account_id}/datasets/{dataset_id}`)
    async fn create_and_upload_dataset(
        &self,
        client: &TensorzeroHttpClient,
        api_key: &SecretString,
        items: &[FireworksSupervisedRow<'_>],
    ) -> Result<String, Error> {
        let dataset_id = format!("t0-{}", Uuid::now_v7());
        let dataset_path = format!("accounts/{}/datasets/{}", self.account_id, dataset_id);
        let res = client
            .post(
                self.api_base
                    .join(&format!("v1/accounts/{}/datasets", self.account_id))
                    .convert_parse_error()?,
            )
            .bearer_auth(api_key.expose_secret())
            .json(&serde_json::json!({
                "datasetId": dataset_id,
                "dataset": {
                    "displayName": dataset_id,
                    "format": "CHAT",
                    "exampleCount": items.len(),
                },
            }))
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!("Error creating dataset: {}", DisplayOrDebugGateway::new(e)),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: None,
                })
            })?;
        let status = res.status();
        let raw_response = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!("Error creating dataset: {}", DisplayOrDebugGateway::new(e)),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
        if !status.is_success() {
            return Err(Error::new(ErrorDetails::InferenceClient {
                status_code: Some(status),
                message: "Error creating dataset".to_string(),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: Some(raw_response),
            }));
        }

        let mut jsonl_data = Vec::new();
        for item in items {
            serde_json::to_writer(&mut jsonl_data, item).map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error writing to JSONL: {}", DisplayOrDebugGateway::new(e)),
                })
            })?;
            jsonl_data.write_all(b"\n").map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error writing to JSONL: {}", DisplayOrDebugGateway::new(e)),
                })
            })?;
        }
        let form = Form::new().part(
            "file",
            Part::bytes(jsonl_data)
                .file_name("dataset.jsonl")
                .mime_str("application/jsonl")
                .map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!(
                            "Error setting MIME type: {}",
                            DisplayOrDebugGateway::new(e)
                        ),
                    })
                })?,
        );
        let request_url = self
            .api_base
            .join(&format!("v1/{dataset_path}:upload"))
            .convert_parse_error()?;
        let request_builder = client
            .post(request_url)
            .bearer_auth(api_key.expose_secret())
            .multipart(form);
        let res = request_builder.send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error sending request to Fireworks: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
        if !res.status().is_success() {
            return Err(Error::new(ErrorDetails::InferenceClient {
                status_code: Some(res.status()),
                message: format!(
                    "Unsuccessful status code when uploading dataset to Fireworks: {}",
                    res.status()
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: res.text().await.ok(),
            }));
        }
        while !self.poll_dataset_read(client, api_key, &dataset_id).await? {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        Ok(dataset_path)
    }
}

impl Optimizer for FireworksSFTConfig {
    type Handle = FireworksSFTJobHandle;

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
        let train_rows: Vec<FireworksSupervisedRow<'_>> = try_join_all(
            train_examples
                .iter()
                .map(FireworksSupervisedRow::from_rendered_sample),
        )
        .await?;

        let val_rows = if let Some(examples) = val_examples.as_ref() {
            Some(
                try_join_all(
                    examples
                        .iter()
                        .map(FireworksSupervisedRow::from_rendered_sample),
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

        let train_fut = self.create_and_upload_dataset(client, api_key, &train_rows);

        let (train_file_id, val_file_id) = if let Some(val_rows) = val_rows.as_ref() {
            let val_fut = self.create_and_upload_dataset(client, api_key, val_rows);

            // Run both futures concurrently
            let (train_id, val_id) = try_join!(train_fut, val_fut)?;
            (train_id, Some(val_id))
        } else {
            // Just run the training file upload
            let train_file_id = train_fut.await?;
            (train_file_id, None)
        };

        let body = FireworksFineTuningRequest {
            base_model: self.model.clone(),
            early_stop: self.early_stop,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            max_context_length: self.max_context_length,
            lora_rank: self.lora_rank,
            batch_size: self.batch_size,
            dataset: train_file_id,
            evaluation_dataset: val_file_id,
            display_name: self.display_name.clone(),
            output_model: self.output_model.clone(),
            warm_start_from: self.warm_start_from.clone(),
            is_turbo: self.is_turbo,
            eval_auto_carveout: self.eval_auto_carveout,
            nodes: self.nodes,
            mtp_enabled: self.mtp_enabled,
            mtp_num_draft_tokens: self.mtp_num_draft_tokens,
            mtp_freeze_base_model: self.mtp_freeze_base_model,
        };

        let request = client
            .post(
                self.api_base
                    .join(&format!(
                        "v1/accounts/{}/supervisedFineTuningJobs",
                        self.account_id
                    ))
                    .convert_parse_error()?,
            )
            .bearer_auth(api_key.expose_secret())
            .json(&body);
        let res = request.json(&body).send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error sending request to Fireworks: {}",
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
                    "Error sending request to Fireworks: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: None,
            })
        })?;
        let job: FireworksFineTuningJobResponse =
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

        // Fireworks job names look like 'accounts/{account_id}/supervisedFineTuningJobs/{job_id}'
        // Extract the job id to construct a dashboard URL
        let job_id = job.name.split("/").last().ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("No job ID in job path: {}", job.name),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        Ok(FireworksSFTJobHandle {
            api_base: self.api_base.clone(),
            account_id: self.account_id.clone(),
            job_url: format!("https://app.fireworks.ai/dashboard/fine-tuning/supervised/{job_id}")
                .parse()
                .convert_parse_error()?,
            job_path: job.name,
            credential_location: self.credential_location.clone(),
        })
    }
}

#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct FireworksSFTJobHandle {
    pub api_base: Url,
    pub account_id: String,
    pub job_url: Url,
    pub job_path: String,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocationWithFallback>,
}

impl std::fmt::Display for FireworksSFTJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[expect(clippy::enum_variant_names)]
enum FireworksFineTuningJobState {
    JobStateUnspecified,
    JobStateCreating,
    JobStateRunning,
    JobStateCompleted,
    JobStateFailed,
    JobStateCancelled,
    JobStateDeleting,
    JobStateWritingResults,
    JobStateValidating,
    JobStateRollout,
    JobStateEvaluation,
    JobStateFailedCleaningUp,
    JobStateDeletingCleaningUp,
    JobStatePolicyUpdate,
    JobStatePending,
}

// Get the 'SCREAMING_SNAKE_CASE' name for the enum value
impl Display for FireworksFineTuningJobState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.serialize(f)
    }
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FireworksFineTuningJobResponse {
    state: FireworksFineTuningJobState,
    status: Option<FireworksFineTuningJobStatus>,
    output_model: Option<String>,
    name: String,
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FireworksFineTuningJobStatus {
    message: String,
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FireworksDeployedModelResponse {
    state: FireworksDeploymentState,
    status: Option<FireworksDeploymentStatus>,
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FireworksDeploymentStatus {
    message: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FireworksDeploymentState {
    StateUnspecified,
    Undeploying,
    Deploying,
    Deployed,
    Updating,
}

impl Display for FireworksDeploymentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.serialize(f)
    }
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FireworksModelResponse {
    deployed_model_refs: Vec<FireworksDeployedModelRef>,
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FireworksDeployedModelRef {
    name: String,
    default: bool,
    state: FireworksDeploymentState,
}

impl FireworksSFTJobHandle {
    async fn get_model(
        &self,
        client: &TensorzeroHttpClient,
        api_key: &SecretString,
        model_path: &str,
    ) -> Result<Option<FireworksModelResponse>, Error> {
        let res = client
            .get(
                self.api_base
                    .join(&format!("v1/{model_path}"))
                    .convert_parse_error()?,
            )
            .bearer_auth(api_key.expose_secret())
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!("Error getting model: {}", DisplayOrDebugGateway::new(e)),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: None,
                })
            })?;
        if res.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let raw_response = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!("Error getting model: {}", DisplayOrDebugGateway::new(e)),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
        let model: FireworksModelResponse = serde_json::from_str(&raw_response).map_err(|e| {
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
        Ok(Some(model))
    }

    async fn deploy_or_poll_model(
        &self,
        client: &TensorzeroHttpClient,
        api_key: &SecretString,
        model_path: &str,
    ) -> Result<FireworksDeploymentState, Error> {
        let model = self.get_model(client, api_key, model_path).await?;
        if let Some(model) = model {
            // We created a default deployment, so assume that this is the one that we want
            // It's possible that the user created a different default deployment while we were polling,
            // but that should be fine - we just care that a default deployment succeeded for our model.
            for deployment in model.deployed_model_refs {
                if deployment.default {
                    return Ok(deployment.state);
                }
            }
        }

        // If we didn't find a default deployment, kick one off.
        // For now, we only support serverless deployments.
        let res = client
            .post(
                self.api_base
                    .join(&format!("v1/accounts/{}/deployedModels", self.account_id))
                    .convert_parse_error()?,
            )
            .bearer_auth(api_key.expose_secret())
            .json(&serde_json::json!({
                "model": model_path,
                "default": true,
                "serverless": true,
                "public": false,
            }))
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorDetails::InferenceClient {
                    status_code: e.status(),
                    message: format!("Error deploying model: {}", DisplayOrDebugGateway::new(e)),
                    provider_type: PROVIDER_TYPE.to_string(),
                    raw_request: None,
                    raw_response: None,
                })
            })?;

        let raw_response = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!("Error deploying model: {}", DisplayOrDebugGateway::new(e)),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
        let response: FireworksDeployedModelResponse = serde_json::from_str(&raw_response)
            .map_err(|e| {
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
        Ok(response.state)
    }
    async fn poll_job(
        &self,
        client: &TensorzeroHttpClient,
        api_key: &SecretString,
    ) -> Result<FireworksFineTuningJobResponse, Error> {
        let request = client
            .get(
                self.api_base
                    .join(&format!("v1/{}", self.job_path))
                    .convert_parse_error()?,
            )
            .bearer_auth(api_key.expose_secret());
        let res = request.send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error sending request to Fireworks: {}",
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
                    "Error sending request to Fireworks: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
        let job: FireworksFineTuningJobResponse =
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
        Ok(job)
    }
}

impl JobHandle for FireworksSFTJobHandle {
    async fn poll(
        &self,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OptimizationJobInfo, Error> {
        let fireworks_credentials: FireworksCredentials = crate::model_table::FireworksKind
            .get_defaulted_credential(self.credential_location.as_ref(), default_credentials)
            .await?;
        let api_key = fireworks_credentials
            .get_api_key(credentials)
            .map_err(|e| e.log())?;
        let job_status = self.poll_job(client, api_key).await?;
        if let FireworksFineTuningJobState::JobStateCompleted = job_status.state {
            // Once the job has completed, start polling the model deployment.
            let model_path = job_status.output_model.ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "No model path in Fireworks JobStateCompleted response".to_string(),
                    raw_request: None,
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            // TODO - start using this as the TensorZero model name
            // once the UI has been refactored to allow separate model names and provider model names
            let _model_id = model_path.split("/").last().ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("No model ID in model path: {model_path}"),
                    raw_request: None,
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            let deployment_state = self
                .deploy_or_poll_model(client, api_key, &model_path)
                .await?;
            match deployment_state {
                FireworksDeploymentState::StateUnspecified
                | FireworksDeploymentState::Deploying
                | FireworksDeploymentState::Undeploying
                | FireworksDeploymentState::Updating => Ok(OptimizationJobInfo::Pending {
                    message: deployment_state.to_string(),
                    estimated_finish: None,
                    trained_tokens: None,
                    error: None,
                }),
                FireworksDeploymentState::Deployed => {
                    let model_provider = UninitializedModelProvider {
                        config: UninitializedProviderConfig::Fireworks {
                            model_name: model_path.clone(),
                            parse_think_blocks: true,
                            api_key_location: None,
                        },
                        extra_headers: None,
                        extra_body: None,
                        timeouts: TimeoutsConfig::default(),
                        discard_unknown_chunks: false,
                    };
                    Ok(OptimizationJobInfo::Completed {
                        output: OptimizerOutput::Model(UninitializedModelConfig {
                            routing: vec![model_path.clone().into()],
                            providers: HashMap::from([(model_path.into(), model_provider)]),
                            timeouts: TimeoutsConfig::default(),
                        }),
                    })
                }
            }
        } else {
            convert_to_optimizer_status(job_status)
        }
    }
}

pub fn convert_to_optimizer_status(
    job: FireworksFineTuningJobResponse,
) -> Result<OptimizationJobInfo, Error> {
    Ok(match job.state {
        FireworksFineTuningJobState::JobStateCreating
        | FireworksFineTuningJobState::JobStatePending
        | FireworksFineTuningJobState::JobStateRunning
        | FireworksFineTuningJobState::JobStateWritingResults
        | FireworksFineTuningJobState::JobStateValidating
        | FireworksFineTuningJobState::JobStateRollout
        | FireworksFineTuningJobState::JobStateEvaluation
        | FireworksFineTuningJobState::JobStateUnspecified
        | FireworksFineTuningJobState::JobStatePolicyUpdate => OptimizationJobInfo::Pending {
            message: job.state.to_string(),
            estimated_finish: None,
            trained_tokens: None,
            error: None,
        },
        FireworksFineTuningJobState::JobStateFailed
        | FireworksFineTuningJobState::JobStateFailedCleaningUp
        | FireworksFineTuningJobState::JobStateCancelled
        | FireworksFineTuningJobState::JobStateDeleting
        | FireworksFineTuningJobState::JobStateDeletingCleaningUp => OptimizationJobInfo::Failed {
            message: job.state.to_string(),
            error: job.status.map(|s| s.message.into()),
        },
        FireworksFineTuningJobState::JobStateCompleted => {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!(
                    "JobStateCompleted should have been handled in poll. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            }))
        }
    })
}
