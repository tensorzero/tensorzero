#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
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
    // Hyperparameters
    pub n_epochs: Option<u32>,
    pub n_checkpoints: Option<u32>,
    pub n_evals: Option<u32>,
    pub batch_size: Option<u32>,
    pub learning_rate: Option<f64>,
    pub warmup_ratio: Option<f64>,
    pub max_grad_norm: Option<f64>,
    pub weight_decay: Option<f64>,
    pub suffix: Option<String>,
    // Learning rate scheduler
    pub lr_scheduler_type: Option<String>, // "linear", "cosine", "constant", etc.
    pub lr_scheduler_min_lr_ratio: Option<f64>,
    // Weights & Biases integration
    pub wandb_api_key: Option<String>,
    pub wandb_base_url: Option<String>,
    pub wandb_project_name: Option<String>,
    pub wandb_name: Option<String>,
    // Training method
    pub train_on_inputs: Option<String>, // "auto", "true", "false" for SFT
    // Training type
    pub training_type: Option<String>, // "Full" or "LoRA"
    // LoRA parameters (only used when training_type is "LoRA")
    pub lora_r: Option<u32>,
    pub lora_alpha: Option<u32>,
    pub lora_dropout: Option<f64>,
    pub lora_trainable_modules: Option<String>,
    // Advanced options
    pub from_checkpoint: Option<String>,
    pub from_hf_model: Option<String>,
    pub hf_model_revision: Option<String>,
    pub hf_api_token: Option<String>,
    pub hf_output_repo_name: Option<String>,
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
#[cfg_attr(feature = "pyo3", pyclass(str, name = "TogetherSFTConfig"))]
pub struct UninitializedTogetherSFTConfig {
    pub model: String,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credentials: Option<CredentialLocation>,
    pub api_base: Option<Url>,
    // Hyperparameters
    pub n_epochs: Option<u32>,
    pub n_checkpoints: Option<u32>,
    pub n_evals: Option<u32>,
    pub batch_size: Option<u32>,
    pub learning_rate: Option<f64>,
    pub warmup_ratio: Option<f64>,
    pub max_grad_norm: Option<f64>,
    pub weight_decay: Option<f64>,
    pub suffix: Option<String>,
    // Learning rate scheduler
    pub lr_scheduler_type: Option<String>,
    pub lr_scheduler_min_lr_ratio: Option<f64>,
    // Weights & Biases integration
    pub wandb_api_key: Option<String>,
    pub wandb_base_url: Option<String>,
    pub wandb_project_name: Option<String>,
    pub wandb_name: Option<String>,
    // Training method
    pub train_on_inputs: Option<String>,
    // Training type
    pub training_type: Option<String>,
    // LoRA parameters
    pub lora_r: Option<u32>,
    pub lora_alpha: Option<u32>,
    pub lora_dropout: Option<f64>,
    pub lora_trainable_modules: Option<String>,
    // Advanced options
    pub from_checkpoint: Option<String>,
    pub from_hf_model: Option<String>,
    pub hf_model_revision: Option<String>,
    pub hf_api_token: Option<String>,
    pub hf_output_repo_name: Option<String>,
}

impl std::fmt::Display for UninitializedTogetherSFTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
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

#[cfg(feature = "pyo3")]
#[pymethods]
impl UninitializedTogetherSFTConfig {
    // We allow too many arguments since it is a Python constructor
    /// NOTE: This signature currently does not work:
    /// print(TogetherSFTConfig.__init__.__text_signature__)
    /// prints out signature:
    /// ($self, /, *args, **kwargs)
    #[expect(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, model, credentials=None, api_base=None, n_epochs=None, n_checkpoints=None, n_evals=None, batch_size=None, learning_rate=None, warmup_ratio=None, max_grad_norm=None, weight_decay=None, suffix=None, lr_scheduler_type=None, lr_scheduler_min_lr_ratio=None, wandb_api_key=None, wandb_base_url=None, wandb_project_name=None, wandb_name=None, train_on_inputs=None, training_type=None, lora_r=None, lora_alpha=None, lora_dropout=None, lora_trainable_modules=None, from_checkpoint=None, from_hf_model=None, hf_model_revision=None, hf_api_token=None, hf_output_repo_name=None))]
    pub fn new(
        model: String,
        credentials: Option<String>,
        api_base: Option<String>,
        n_epochs: Option<u32>,
        n_checkpoints: Option<u32>,
        n_evals: Option<u32>,
        batch_size: Option<u32>,
        learning_rate: Option<f64>,
        warmup_ratio: Option<f64>,
        max_grad_norm: Option<f64>,
        weight_decay: Option<f64>,
        suffix: Option<String>,
        lr_scheduler_type: Option<String>,
        lr_scheduler_min_lr_ratio: Option<f64>,
        wandb_api_key: Option<String>,
        wandb_base_url: Option<String>,
        wandb_project_name: Option<String>,
        wandb_name: Option<String>,
        train_on_inputs: Option<String>,
        training_type: Option<String>,
        lora_r: Option<u32>,
        lora_alpha: Option<u32>,
        lora_dropout: Option<f64>,
        lora_trainable_modules: Option<String>,
        from_checkpoint: Option<String>,
        from_hf_model: Option<String>,
        hf_model_revision: Option<String>,
        hf_api_token: Option<String>,
        hf_output_repo_name: Option<String>,
    ) -> PyResult<Self> {
        // Use Deserialize to convert the string to a CredentialLocation
        let credentials = credentials
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Invalid credentials JSON: {e}")))?
            .or_else(|| Some(default_api_key_location()));
        let api_base = api_base
            .map(|s| {
                Url::parse(&s)
                    .map_err(|e| PyErr::new::<PyValueError, std::string::String>(e.to_string()))
            })
            .transpose()?;
        Ok(Self {
            model,
            credentials,
            api_base,
            n_epochs,
            n_checkpoints,
            n_evals,
            batch_size,
            learning_rate,
            warmup_ratio,
            max_grad_norm,
            weight_decay,
            suffix,
            lr_scheduler_type,
            lr_scheduler_min_lr_ratio,
            wandb_api_key,
            wandb_base_url,
            wandb_project_name,
            wandb_name,
            train_on_inputs,
            training_type,
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_trainable_modules,
            from_checkpoint,
            from_hf_model,
            hf_model_revision,
            hf_api_token,
            hf_output_repo_name,
        })
    }

    /// Initialize the TogetherSFTConfig. All parameters are optional except for `model`.
    ///
    /// :param model: Name of the base model to run fine-tune job on.
    /// :param credentials: The credentials to use for the fine-tuning job. This should be a string like "env::TOGETHER_API_KEY". See docs for more details.
    /// :param api_base: The base URL to use for the fine-tuning job. This is primarily used for testing.
    /// :param n_epochs: Number of complete passes through the training dataset. Default: 1. Higher values may improve results but increase cost and overfitting risk.
    /// :param n_checkpoints: Number of intermediate model versions saved during training. Default: 1.
    /// :param n_evals: Number of evaluations to be run on a given validation set during training. Default: 0.
    /// :param batch_size: Number of training examples processed together. Default: 8. Larger batches use more memory but may train faster.
    /// :param learning_rate: Controls how quickly the model adapts to new information. Default: 0.00001. Too high may cause instability, too low may slow convergence.
    /// :param warmup_ratio: Percent of steps at the start of training to linearly increase learning rate. Default: 0.
    /// :param max_grad_norm: Max gradient norm for gradient clipping. Default: 1. Set to 0 to disable.
    /// :param weight_decay: Regularization parameter for the optimizer. Default: 0.
    /// :param suffix: Suffix that will be added to your fine-tuned model name.
    /// :param lr_scheduler_type: The learning rate scheduler type to use. Options: 'linear', 'cosine'. Default: 'linear'.
    /// :param lr_scheduler_min_lr_ratio: Minimum learning rate ratio for the scheduler. Default: 0.0.
    /// :param wandb_api_key: Weights & Biases API key for experiment tracking.
    /// :param wandb_base_url: Weights & Biases base URL for dedicated instance.
    /// :param wandb_project_name: Weights & Biases project name. Default: 'together'.
    /// :param wandb_name: Weights & Biases run name.
    /// :param train_on_inputs: Whether to train on input tokens for SFT. Options: 'auto', 'true', 'false'. Default: 'auto'.
    /// :param training_type: Type of training to perform. Options: 'Full', 'Lora'. Default: 'Lora'.
    /// :param lora_r: LoRA rank parameter. Default: 8. Only used when training_type is 'Lora'.
    /// :param lora_alpha: LoRA alpha parameter. Default: 32. Only used when training_type is 'Lora'.
    /// :param lora_dropout: LoRA dropout parameter. Default: 0.0. Only used when training_type is 'Lora'.
    /// :param lora_trainable_modules: LoRA trainable modules specification. Default: 'all-linear'. Only used when training_type is 'Lora'.
    /// :param from_checkpoint: Continue training from a previous checkpoint job ID.
    /// :param from_hf_model: Start training from a Hugging Face model repository.
    /// :param hf_model_revision: Specific model version/commit from Hugging Face repository.
    /// :param hf_api_token: Hugging Face API token for authentication.
    /// :param hf_output_repo_name: Hugging Face repository name for uploading the fine-tuned model.
    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, credentials=None, api_base=None, n_epochs=None, n_checkpoints=None, n_evals=None, batch_size=None, learning_rate=None, warmup_ratio=None, max_grad_norm=None, weight_decay=None, suffix=None, lr_scheduler_type=None, lr_scheduler_min_lr_ratio=None, wandb_api_key=None, wandb_base_url=None, wandb_project_name=None, wandb_name=None, train_on_inputs=None, training_type=None, lora_r=None, lora_alpha=None, lora_dropout=None, lora_trainable_modules=None, from_checkpoint=None, from_hf_model=None, hf_model_revision=None, hf_api_token=None, hf_output_repo_name=None))]
    fn __init__(
        this: Py<Self>,
        model: String,
        credentials: Option<String>,
        api_base: Option<String>,
        n_epochs: Option<u32>,
        n_checkpoints: Option<u32>,
        n_evals: Option<u32>,
        batch_size: Option<u32>,
        learning_rate: Option<f64>,
        warmup_ratio: Option<f64>,
        max_grad_norm: Option<f64>,
        weight_decay: Option<f64>,
        suffix: Option<String>,
        lr_scheduler_type: Option<String>,
        lr_scheduler_min_lr_ratio: Option<f64>,
        wandb_api_key: Option<String>,
        wandb_base_url: Option<String>,
        wandb_project_name: Option<String>,
        wandb_name: Option<String>,
        train_on_inputs: Option<String>,
        training_type: Option<String>,
        lora_r: Option<u32>,
        lora_alpha: Option<u32>,
        lora_dropout: Option<f64>,
        lora_trainable_modules: Option<String>,
        from_checkpoint: Option<String>,
        from_hf_model: Option<String>,
        hf_model_revision: Option<String>,
        hf_api_token: Option<String>,
        hf_output_repo_name: Option<String>,
    ) -> Py<Self> {
        this
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
            // Hyperparameters
            n_epochs: self.n_epochs,
            n_checkpoints: self.n_checkpoints,
            n_evals: self.n_evals,
            batch_size: self.batch_size,
            learning_rate: self.learning_rate,
            warmup_ratio: self.warmup_ratio,
            max_grad_norm: self.max_grad_norm,
            weight_decay: self.weight_decay,
            suffix: self.suffix,
            // Learning rate scheduler
            lr_scheduler_type: self.lr_scheduler_type,
            lr_scheduler_min_lr_ratio: self.lr_scheduler_min_lr_ratio,
            // Weights & Biases integration
            wandb_api_key: self.wandb_api_key,
            wandb_base_url: self.wandb_base_url,
            wandb_project_name: self.wandb_project_name,
            wandb_name: self.wandb_name,
            // Training method
            train_on_inputs: self.train_on_inputs,
            // Training type
            training_type: self.training_type,
            // LoRA parameters
            lora_r: self.lora_r,
            lora_alpha: self.lora_alpha,
            lora_dropout: self.lora_dropout,
            lora_trainable_modules: self.lora_trainable_modules,
            // Advanced options
            from_checkpoint: self.from_checkpoint,
            from_hf_model: self.from_hf_model,
            hf_model_revision: self.hf_model_revision,
            hf_api_token: self.hf_api_token,
            hf_output_repo_name: self.hf_output_repo_name,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    // Together claims that this is optional, but errors if it's not provided
    pub batch_size: u32,
    lr_scheduler: TogetherLRScheduler,
    learning_rate: f64,
    training_method: TogetherTrainingMethod,
    training_type: TogetherTrainingType,
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

#[derive(Debug, Serialize)]
struct TogetherLRScheduler {
    lr_scheduler_type: TogetherLRSchedulerType,
    lr_scheduler_args: TogetherLRSchedulerArgs,
}

impl TogetherLRScheduler {
    fn from_config(
        lr_scheduler_type: Option<&str>,
        min_lr_ratio: Option<f64>,
    ) -> Result<Self, Error> {
        let scheduler_type = if let Some(scheduler_type) = lr_scheduler_type {
            TogetherLRSchedulerType::try_from(scheduler_type)?
        } else {
            TogetherLRSchedulerType::Linear
        };

        Ok(TogetherLRScheduler {
            lr_scheduler_type: scheduler_type,
            lr_scheduler_args: TogetherLRSchedulerArgs {
                min_lr_ratio: min_lr_ratio.unwrap_or(0.0),
            },
        })
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum TogetherTrainingType {
    Full,
    Lora {
        #[serde(skip_serializing_if = "Option::is_none")]
        lora_r: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        lora_alpha: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        lora_dropout: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        lora_trainable_modules: Option<String>,
    },
}

impl TogetherTrainingType {
    fn from_config(
        training_type: Option<&str>,
        lora_r: Option<u32>,
        lora_alpha: Option<u32>,
        lora_dropout: Option<f64>,
        lora_trainable_modules: Option<&str>,
    ) -> Result<Self, Error> {
        match training_type {
            Some("full") => Ok(TogetherTrainingType::Full),
            Some("lora") | None => Ok(TogetherTrainingType::Lora {
                lora_r: lora_r.or(Some(8)),
                lora_alpha: lora_alpha.or(Some(32)),
                lora_dropout: lora_dropout.or(Some(0.0)),
                lora_trainable_modules: Some(lora_trainable_modules.unwrap_or("all-linear").to_string()),
            }),
            Some(invalid_type) => Err(Error::new(ErrorDetails::Config {
                message: format!("Invalid training_type '{invalid_type}' for Together provider. Supported types: 'Full', 'Lora'"),
            })),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "method")]
enum TogetherTrainingMethod {
    #[serde(rename = "sft")]
    Sft { train_on_inputs: String },
}

impl TogetherTrainingMethod {
    fn from_config(train_on_inputs: Option<&str>) -> Self {
        TogetherTrainingMethod::Sft {
            train_on_inputs: train_on_inputs.unwrap_or("auto").to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum TogetherLRSchedulerType {
    Linear,
    Cosine,
}

impl TryFrom<&str> for TogetherLRSchedulerType {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "linear" => Ok(TogetherLRSchedulerType::Linear),
            "cosine" => Ok(TogetherLRSchedulerType::Cosine),
            _ => Err(Error::new(ErrorDetails::Config {
                message: format!("Invalid lr_scheduler_type '{value}' for Together provider. Supported types: 'linear', 'cosine'"),
            })),
        }
    }
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

        // Determine n_evals based on whether we have validation data and config
        let n_evals = if val_file_id.is_some() {
            self.n_evals.or(Some(1))
        } else {
            Some(0)
        };

        // Build configurations using clean impl methods
        let lr_scheduler = TogetherLRScheduler::from_config(
            self.lr_scheduler_type.as_deref(),
            self.lr_scheduler_min_lr_ratio,
        )?;

        let training_method = TogetherTrainingMethod::from_config(self.train_on_inputs.as_deref());

        let training_type = TogetherTrainingType::from_config(
            self.training_type.as_deref(),
            self.lora_r,
            self.lora_alpha,
            self.lora_dropout,
            self.lora_trainable_modules.as_deref(),
        )?;

        let res: TogetherCreateJobResponse = client
            .post(self.api_base.join("fine-tunes").convert_parse_error()?)
            .bearer_auth(api_key.expose_secret())
            .json(&TogetherCreateJobRequest {
                training_file: train_file_id,
                validation_file: val_file_id,
                model: self.model.clone(),
                n_epochs: self.n_epochs.or(Some(1)),
                n_checkpoints: self.n_checkpoints.or(Some(1)),
                n_evals,
                learning_rate: self.learning_rate.unwrap_or(0.00001),
                batch_size: self.batch_size.unwrap_or(8),
                lr_scheduler,
                training_method,
                training_type,
                warmup_ratio: self.warmup_ratio.or(Some(0.0)),
                max_grad_norm: self.max_grad_norm.or(Some(1.0)),
                weight_decay: self.weight_decay.or(Some(0.0)),
                suffix: self.suffix.clone(),
                // Weights & Biases integration
                wandb_api_key: self.wandb_api_key.clone(),
                wandb_base_url: self.wandb_base_url.clone(),
                wandb_project_name: self.wandb_project_name.clone(),
                wandb_name: self.wandb_name.clone(),
                // Advanced options
                from_checkpoint: self.from_checkpoint.clone(),
                from_hf_model: self.from_hf_model.clone(),
                hf_model_revision: self.hf_model_revision.clone(),
                hf_api_token: self.hf_api_token.clone(),
                hf_output_repo_name: self.hf_output_repo_name.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_training_type_error() {
        let result = TogetherTrainingType::from_config(
            Some("invalid_type"),
            Some(8),
            Some(32),
            Some(0.0),
            Some("all-linear"),
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Invalid training_type 'invalid_type'"));
        assert!(err.to_string().contains("Supported types: 'Full', 'Lora'"));
    }

    #[test]
    fn test_invalid_lr_scheduler_type_error() {
        let result = TogetherLRScheduler::from_config(Some("invalid_scheduler"), Some(0.1));

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Invalid lr_scheduler_type 'invalid_scheduler'"));
        assert!(err
            .to_string()
            .contains("Supported types: 'linear', 'cosine'"));
    }

    #[test]
    fn test_valid_training_type_full() {
        let result = TogetherTrainingType::from_config(
            Some("full"),
            Some(8),
            Some(32),
            Some(0.0),
            Some("all-linear"),
        );

        assert!(result.is_ok());
        match result.unwrap() {
            TogetherTrainingType::Full => (),
            _ => panic!("Expected Full training type"),
        }
    }

    #[test]
    fn test_valid_training_type_lora() {
        let result = TogetherTrainingType::from_config(
            Some("lora"),
            Some(16),
            Some(64),
            Some(0.1),
            Some("all-linear"),
        );

        assert!(result.is_ok());
        match result.unwrap() {
            TogetherTrainingType::Lora {
                lora_r,
                lora_alpha,
                lora_dropout,
                lora_trainable_modules,
            } => {
                assert_eq!(lora_r, Some(16));
                assert_eq!(lora_alpha, Some(64));
                assert_eq!(lora_dropout, Some(0.1));
                assert_eq!(lora_trainable_modules, Some("all-linear".to_string()));
            }
            _ => panic!("Expected Lora training type"),
        }
    }

    #[test]
    fn test_valid_lr_scheduler_cosine() {
        let result = TogetherLRScheduler::from_config(Some("cosine"), Some(0.2));

        assert!(result.is_ok());
        let scheduler = result.unwrap();
        assert_eq!(scheduler.lr_scheduler_args.min_lr_ratio, 0.2);
    }
}
