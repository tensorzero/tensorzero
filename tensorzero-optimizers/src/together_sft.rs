//! Together Supervised Fine-Tuning (SFT) optimizer implementation

use std::{borrow::Cow, collections::HashMap, fmt::Display, io::Write};

use futures::future::try_join_all;
use reqwest::multipart::{Form, Part};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::try_join;
use url::Url;

use tensorzero_core::{
    config::{
        Config, TimeoutsConfig,
        provider_types::{ProviderTypesConfig, TogetherSFTConfig as TogetherProviderSFTConfig},
    },
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    http::TensorzeroHttpClient,
    inference::types::ContentBlock,
    model::{UninitializedModelConfig, UninitializedModelProvider, UninitializedProviderConfig},
    model_table::{ProviderKind, ProviderTypeDefaultCredentials, TogetherKind},
    optimization::{
        OptimizationJobInfo, OptimizerOutput,
        together_sft::{
            TogetherBatchSize, TogetherLRScheduler, TogetherSFTConfig, TogetherSFTJobHandle,
            TogetherTrainingMethod, TogetherTrainingType,
        },
    },
    providers::{
        helpers::UrlParseErrExt,
        openai::tensorzero_to_openai_assistant_message,
        openai::{OpenAIMessagesConfig, OpenAIRequestMessage, OpenAITool},
        together::prepare_together_messages,
        together::{PROVIDER_TYPE, TOGETHER_API_BASE},
    },
    stored_inference::{LazyRenderedSample, RenderedSample},
    utils::mock::get_mock_provider_api_base,
};

use crate::{JobHandle, Optimizer};

fn get_sft_config(provider_types: &ProviderTypesConfig) -> Option<&TogetherProviderSFTConfig> {
    provider_types.together.sft.as_ref()
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    // Together claims that this is optional, but errors if it's not provided
    pub batch_size: TogetherBatchSize,
    pub lr_scheduler: TogetherLRScheduler,
    pub learning_rate: f64,
    pub training_method: TogetherTrainingMethod,
    pub training_type: TogetherTrainingType,
    // Weights & Biases integration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wandb_api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wandb_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wandb_project_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wandb_name: Option<String>,
    // Advanced options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_checkpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_hf_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_model_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_api_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_output_repo_name: Option<String>,
}

impl Optimizer for TogetherSFTConfig {
    type Handle = TogetherSFTJobHandle;

    async fn launch(
        &self,
        client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
        _clickhouse_connection_info: &ClickHouseConnectionInfo,
        config: Arc<Config>,
    ) -> Result<Self::Handle, Error> {
        // Get optional provider-level configuration
        let sft_config = get_sft_config(&config.provider_types);

        // Get credentials from provider defaults
        let together_credentials = TogetherKind
            .get_defaulted_credential(None, &config.models.default_credentials)
            .await?;
        let api_key = together_credentials
            .get_api_key(credentials)
            .map_err(|e| e.log())?;

        // Use mock API base for testing if set, otherwise default API base
        let api_base =
            get_mock_provider_api_base("together/").unwrap_or_else(|| TOGETHER_API_BASE.clone());

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
        let train_rows: Vec<TogetherSupervisedRow> = try_join_all(
            train_examples
                .iter()
                .map(TogetherSupervisedRow::from_rendered_sample),
        )
        .await?;

        let val_rows = if let Some(examples) = val_examples.as_ref() {
            Some(
                try_join_all(
                    examples
                        .iter()
                        .map(TogetherSupervisedRow::from_rendered_sample),
                )
                .await?,
            )
        } else {
            None
        };
        // Upload the training and validation rows to Together files
        let train_file_fut = upload_file(client, &api_key, &api_base, &train_rows, "fine-tune");
        let (train_file_id, val_file_id) = if let Some(val_rows) = val_rows.as_ref() {
            // Upload the files in parallel
            let val_fut = upload_file(client, &api_key, &api_base, val_rows, "eval");
            let (train_file_id, val_file_id) = try_join!(train_file_fut, val_fut)?;
            (train_file_id, Some(val_file_id))
        } else {
            let train_file_id = train_file_fut.await?;
            (train_file_id, None)
        };

        // Determine n_evals based on whether we have validation data and config
        let n_evals = if val_file_id.is_some() {
            self.n_evals.or(Some(1))
        } else {
            Some(0)
        };

        // Build API configurations with defaults
        let lr_scheduler = self.lr_scheduler.clone();

        let training_method = match &self.training_method {
            TogetherTrainingMethod::Sft { train_on_inputs } => TogetherTrainingMethod::Sft {
                train_on_inputs: train_on_inputs.clone().or_else(|| Some("auto".to_string())),
            },
        };

        let training_type = match &self.training_type {
            TogetherTrainingType::Full {} => TogetherTrainingType::Full {},
            TogetherTrainingType::Lora {
                lora_r,
                lora_alpha,
                lora_dropout,
                lora_trainable_modules,
            } => TogetherTrainingType::Lora {
                lora_r: lora_r.or(Some(8)),
                lora_alpha: lora_alpha.or(Some(16)),
                lora_dropout: lora_dropout.or(Some(0.0)),
                lora_trainable_modules: Some(
                    lora_trainable_modules
                        .clone()
                        .unwrap_or_else(|| "all-linear".to_string()),
                ),
            },
        };

        let res: TogetherCreateJobResponse = client
            .post(api_base.join("fine-tunes").convert_parse_error()?)
            .bearer_auth(api_key.expose_secret())
            .json(&TogetherCreateJobRequest {
                training_file: train_file_id,
                validation_file: val_file_id,
                model: self.model.clone(),
                n_epochs: Some(self.n_epochs),
                n_checkpoints: Some(self.n_checkpoints),
                n_evals,
                learning_rate: self.learning_rate,
                batch_size: self.batch_size.clone(),
                lr_scheduler,
                training_method,
                training_type,
                warmup_ratio: Some(self.warmup_ratio),
                max_grad_norm: Some(self.max_grad_norm),
                weight_decay: Some(self.weight_decay),
                suffix: self.suffix.clone(),
                // Weights & Biases integration - get from provider config if available
                wandb_api_key: sft_config.and_then(|c| c.wandb_api_key.clone()),
                wandb_base_url: sft_config.and_then(|c| c.wandb_base_url.clone()),
                wandb_project_name: sft_config.and_then(|c| c.wandb_project_name.clone()),
                wandb_name: self.wandb_name.clone(),
                // Advanced options
                from_checkpoint: self.from_checkpoint.clone(),
                from_hf_model: self.from_hf_model.clone(),
                hf_model_revision: self.hf_model_revision.clone(),
                hf_api_token: sft_config.and_then(|c| c.hf_api_token.clone()),
                hf_output_repo_name: self.hf_output_repo_name.clone(),
            })
            .send_and_parse_json(PROVIDER_TYPE)
            .await?;
        Ok(TogetherSFTJobHandle {
            job_id: res.id.clone(),
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
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        default_credentials: &ProviderTypeDefaultCredentials,
        _provider_types: &ProviderTypesConfig,
    ) -> Result<OptimizationJobInfo, Error> {
        // Get credentials from provider defaults
        let together_credentials = TogetherKind
            .get_defaulted_credential(None, default_credentials)
            .await?;
        let api_key = together_credentials
            .get_api_key(credentials)
            .map_err(|e| e.log())?;

        // Use mock API base for testing if set, otherwise default API base
        let api_base =
            get_mock_provider_api_base("together/").unwrap_or_else(|| TOGETHER_API_BASE.clone());

        let res: TogetherJobResponse = client
            .get(
                api_base
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
                let model_name = res.model_output_name.ok_or_else(|| {
                    Error::new(ErrorDetails::InferenceServer {
                        message: "Missing model_output_name in Together job response".to_string(),
                        provider_type: PROVIDER_TYPE.to_string(),
                        raw_request: None,
                        raw_response: None,
                    })
                })?;

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
                        skip_relay: None,
                    }),
                })
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TogetherSupervisedRow<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAITool<'a>>,
}

impl<'a> TogetherSupervisedRow<'a> {
    pub async fn from_rendered_sample(inference: &'a LazyRenderedSample) -> Result<Self, Error> {
        if inference
            .tool_params
            .parallel_tool_calls
            .unwrap_or_default()
        {
            return Err(Error::new(ErrorDetails::InvalidRenderedStoredInference {
                message: "Parallel tool calls are not supported for Together".to_string(),
            }));
        }
        let tools = inference
            .tool_params
            .additional_tools
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .filter_map(|dt| match &dt {
                        tensorzero_core::tool::Tool::Function(func) => Some(func.into()),
                        tensorzero_core::tool::Tool::OpenAICustom(_) => None, // Skip custom tools for SFT
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut messages = prepare_together_messages(
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

#[derive(Debug, Deserialize)]
struct TogetherFileResponse {
    id: String,
}

async fn upload_file(
    client: &TensorzeroHttpClient,
    api_key: &SecretString,
    api_base: &Url,
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
        .post(api_base.join("files/upload").convert_parse_error()?)
        .bearer_auth(api_key.expose_secret())
        .multipart(form)
        .send_and_parse_json(PROVIDER_TYPE)
        .await?;
    Ok(res.id)
}
