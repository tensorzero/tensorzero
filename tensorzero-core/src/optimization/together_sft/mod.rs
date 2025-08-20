#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Display;
use std::io::Write;

use crate::config::TimeoutsConfig;
use crate::endpoints::inference::InferenceCredentials;
use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use crate::inference::types::ContentBlock;
use crate::model::{
    build_creds_caching_default, CredentialLocation, UninitializedModelConfig,
    UninitializedModelProvider, UninitializedProviderConfig,
};
use crate::optimization::{JobHandle, OptimizationJobInfo, Optimizer, OptimizerOutput};
use crate::providers::helpers::{TensorZeroRequestBuilderExt, UrlParseErrExt};
use crate::providers::openai::OpenAIRequestMessage;
use crate::providers::openai::{tensorzero_to_openai_assistant_message, OpenAITool};
use crate::providers::together::{
    default_api_key_location, prepare_together_messages, TogetherCredentials, DEFAULT_CREDENTIALS,
    PROVIDER_TYPE, TOGETHER_API_BASE,
};
use crate::stored_inference::RenderedSample;

use reqwest::multipart::{Form, Part};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::try_join;
use url::Url;

use crate::error::{DisplayOrDebugGateway, Error, ErrorDetails};

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct TogetherSFTConfig {
    pub model: String,
    #[serde(skip)]
    pub credentials: TogetherCredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocation>,
    pub api_base: Url,
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct TogetherSFTJobHandle {
    pub api_base: Url,
    pub job_id: String,
    // A url to a human-readable page for the job.
    pub job_url: Url,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocation>,
}

impl std::fmt::Display for TogetherSFTJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
pub struct UninitializedTogetherSFTConfig {
    pub model: String,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credentials: Option<CredentialLocation>,
    pub api_base: Option<Url>,
}

#[derive(Debug, Serialize)]
pub struct TogetherSupervisedRow<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAITool<'a>>,
}

impl<'a> TryFrom<&'a RenderedSample> for TogetherSupervisedRow<'a> {
    type Error = Error;
    fn try_from(inference: &'a RenderedSample) -> Result<Self, Self::Error> {
        let tools = match &inference.tool_params {
            Some(tool_params) => {
                if tool_params.parallel_tool_calls.unwrap_or_default() {
                    return Err(Error::new(ErrorDetails::InvalidRenderedStoredInference {
                        message: "Parallel tool calls are not supported for Together".to_string(),
                    }));
                }
                tool_params.tools_available.iter().map(Into::into).collect()
            }
            None => vec![],
        };
        let mut messages = prepare_together_messages(
            inference.input.system.as_deref(),
            &inference.input.messages,
        )?;

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
            PROVIDER_TYPE,
        )?;
        messages.push(final_assistant_message);
        Ok(Self { messages, tools })
    }
}

impl UninitializedTogetherSFTConfig {
    pub fn load(self) -> Result<TogetherSFTConfig, Error> {
        Ok(TogetherSFTConfig {
            model: self.model,
            api_base: self.api_base.unwrap_or_else(|| TOGETHER_API_BASE.clone()),
            credentials: build_creds_caching_default(
                self.credentials.clone(),
                default_api_key_location(),
                PROVIDER_TYPE,
                &DEFAULT_CREDENTIALS,
            )?,
            credential_location: self.credentials,
        })
    }
}

#[derive(Debug, Deserialize)]
struct TogetherFileResponse {
    id: String,
}

#[derive(Debug, Deserialize)]
pub struct TogetherCreateJobResponse {
    id: String,
}

#[derive(Debug, Deserialize)]
struct TogetherJobResponse {
    status: TogetherJobStatus,
    token_count: Option<u64>,
    model_output_name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TogetherJobStatus {
    Pending,
    Queued,
    Running,
    Compressing,
    Uploading,
    CancelRequested,
    UserError,
    Cancelled,
    Error,
    Completed,
}

impl Display for TogetherJobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.serialize(f)
    }
}

impl TogetherSFTConfig {
    /// Uploads the given rows as a Together file, returning the file ID
    async fn upload_file(
        &self,
        client: &reqwest::Client,
        api_key: &SecretString,
        items: &[TogetherSupervisedRow<'_>],
        purpose: &'static str,
    ) -> Result<String, Error> {
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
        let form = Form::new()
            .part(
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
            )
            .text("purpose", purpose)
            .text("file_name", "dataset.jsonl");
        let res: TogetherFileResponse = client
            .post(self.api_base.join("files/upload").convert_parse_error()?)
            .bearer_auth(api_key.expose_secret())
            .multipart(form)
            .send_and_parse_json(PROVIDER_TYPE)
            .await?;
        Ok(res.id)
    }
}

#[derive(Debug, Serialize)]
struct TogetherCreateJobRequest {
    pub training_file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_file: Option<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_epochs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_checkpoints: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_evals: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_grad_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_decay: Option<f64>,
    // Together claims that this is optional, but errors if it's not provided
    pub batch_size: u32,
    lr_scheduler: TogetherLRScheduler,
    learning_rate: f64,
    training_method: TogetherTrainingMethod,
    training_type: TogetherTrainingType,
}

#[derive(Debug, Serialize)]
struct TogetherLRScheduler {
    lr_scheduler_type: TogetherLRSchedulerType,
    lr_scheduler_args: TogetherLRSchedulerArgs,
}

#[derive(Debug, Serialize)]
struct TogetherTrainingType {
    r#type: String,
    lora_r: Option<u32>,
    lora_alpha: Option<u32>,
    lora_dropout: Option<f64>,
    lora_trainable_modules: String,
}

#[derive(Debug, Serialize)]
struct TogetherTrainingMethod {
    method: String,
    train_on_inputs: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum TogetherLRSchedulerType {
    Linear,
}

#[derive(Debug, Serialize)]
struct TogetherLRSchedulerArgs {
    min_lr_ratio: f64,
}

impl Optimizer for TogetherSFTConfig {
    type Handle = TogetherSFTJobHandle;

    async fn launch(
        &self,
        client: &reqwest::Client,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
    ) -> Result<Self::Handle, Error> {
        // TODO(#2642): improve error handling here so we know what index of example failed
        let train_rows: Vec<TogetherSupervisedRow> = train_examples
            .iter()
            .map(TogetherSupervisedRow::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        let val_rows: Option<Vec<TogetherSupervisedRow>> = val_examples
            .as_ref()
            .map(|examples| {
                examples
                    .iter()
                    .map(TogetherSupervisedRow::try_from)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;

        // Upload the training and validation rows to Together files
        let api_key = self.credentials.get_api_key(credentials)?;
        let train_file_fut = self.upload_file(client, &api_key, &train_rows, "fine-tune");
        let (train_file_id, val_file_id) = if let Some(val_rows) = val_rows.as_ref() {
            // Upload the files in parallel
            let val_fut = self.upload_file(client, &api_key, val_rows, "eval");
            let (train_file_id, val_file_id) = try_join!(train_file_fut, val_fut)?;
            (train_file_id, Some(val_file_id))
        } else {
            let train_file_id = train_file_fut.await?;
            (train_file_id, None)
        };

        let n_evals = if val_file_id.is_some() { 1 } else { 0 };

        let res: TogetherCreateJobResponse = client
            .post(self.api_base.join("fine-tunes").convert_parse_error()?)
            .bearer_auth(api_key.expose_secret())
            .json(&TogetherCreateJobRequest {
                training_file: train_file_id,
                validation_file: val_file_id,
                model: self.model.clone(),
                n_epochs: Some(1),
                n_checkpoints: Some(1),
                n_evals: Some(n_evals),
                learning_rate: 0.00001,
                batch_size: 8,
                lr_scheduler: TogetherLRScheduler {
                    lr_scheduler_type: TogetherLRSchedulerType::Linear,
                    lr_scheduler_args: TogetherLRSchedulerArgs { min_lr_ratio: 0.0 },
                },
                training_method: TogetherTrainingMethod {
                    method: "sft".to_string(),
                    train_on_inputs: "auto".to_string(),
                },
                training_type: TogetherTrainingType {
                    r#type: "Lora".to_string(),
                    lora_r: Some(8),
                    lora_alpha: Some(32),
                    lora_dropout: Some(0.0),
                    lora_trainable_modules: "all-linear".to_string(),
                },
                warmup_ratio: Some(0.0),
                max_grad_norm: Some(1.0),
                weight_decay: Some(0.0),
            })
            .send_and_parse_json(PROVIDER_TYPE)
            .await?;
        Ok(TogetherSFTJobHandle {
            api_base: self.api_base.clone(),
            job_id: res.id.clone(),
            credential_location: self.credential_location.clone(),
            job_url: format!("https://api.together.ai/fine-tuning/{}", res.id)
                .parse()
                .map_err(|e| {
                    Error::new(ErrorDetails::InternalError {
                        message: format!(
                            "Failed to construct job url: {e}. {IMPOSSIBLE_ERROR_MESSAGE}"
                        ),
                    })
                })?,
        })
    }
}

impl JobHandle for TogetherSFTJobHandle {
    async fn poll(
        &self,
        client: &reqwest::Client,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizationJobInfo, Error> {
        let together_credentials = build_creds_caching_default(
            self.credential_location.clone(),
            default_api_key_location(),
            PROVIDER_TYPE,
            &DEFAULT_CREDENTIALS,
        )?;
        let api_key = together_credentials.get_api_key(credentials)?;
        let res: TogetherJobResponse = client
            .get(
                self.api_base
                    .join(&format!("fine-tunes/{}", self.job_id))
                    .convert_parse_error()?,
            )
            .bearer_auth(api_key.expose_secret())
            .send_and_parse_json(PROVIDER_TYPE)
            .await?;
        match res.status {
            TogetherJobStatus::Pending
            | TogetherJobStatus::Queued
            | TogetherJobStatus::Running
            | TogetherJobStatus::Compressing
            | TogetherJobStatus::Uploading => Ok(OptimizationJobInfo::Pending {
                message: res.status.to_string(),
                estimated_finish: None,
                trained_tokens: res.token_count,
                error: None,
            }),
            TogetherJobStatus::CancelRequested
            | TogetherJobStatus::Cancelled
            | TogetherJobStatus::UserError
            | TogetherJobStatus::Error => Ok(OptimizationJobInfo::Failed {
                message: res.status.to_string(),
                error: None,
            }),
            TogetherJobStatus::Completed => {
                let model_name =
                    res.model_output_name
                        .ok_or(Error::new(ErrorDetails::InferenceServer {
                            message: "Missing model_output_name in Together job response"
                                .to_string(),
                            provider_type: PROVIDER_TYPE.to_string(),
                            raw_request: None,
                            raw_response: None,
                        }))?;

                let model_provider = UninitializedModelProvider {
                    config: UninitializedProviderConfig::Together {
                        model_name: model_name.clone(),
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
                        routing: vec![model_name.clone().into()],
                        providers: HashMap::from([(model_name.into(), model_provider)]),
                        timeouts: TimeoutsConfig::default(),
                    }),
                })
            }
        }
    }
}
