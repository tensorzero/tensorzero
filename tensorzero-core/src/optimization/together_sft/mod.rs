#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::deserialize_from_pyobj;
#[cfg(feature = "pyo3")]
use pyo3::{exceptions::PyValueError, prelude::*};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Display;
use std::io::Write;

use crate::config::{Config, TimeoutsConfig};
use crate::db::clickhouse::ClickHouseConnectionInfo;
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

// Default functions for hyperparameters
fn default_n_epochs() -> u32 {
    1
}

fn default_n_checkpoints() -> u32 {
    1
}

fn default_learning_rate() -> f64 {
    0.00001
}

fn default_warmup_ratio() -> f64 {
    0.0
}

fn default_max_grad_norm() -> f64 {
    1.0
}

fn default_weight_decay() -> f64 {
    0.0
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass)]
#[serde(rename_all = "lowercase")]
pub enum TogetherBatchSizeDescription {
    Max,
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass)]
#[serde(untagged)]
pub enum TogetherBatchSize {
    Number(u32),
    Description(TogetherBatchSizeDescription),
}

impl Default for TogetherBatchSize {
    fn default() -> Self {
        Self::Description(TogetherBatchSizeDescription::Max)
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(test, ts(export))]
pub struct TogetherSFTConfig {
    pub model: String,
    #[serde(skip)]
    pub credentials: TogetherCredentials,
    #[cfg_attr(test, ts(type = "string | null"))]
    pub credential_location: Option<CredentialLocation>,
    pub api_base: Url,
    // Hyperparameters
    pub n_epochs: u32,
    pub n_checkpoints: u32,
    pub n_evals: Option<u32>, // Keep as Option due to conditional logic
    pub batch_size: TogetherBatchSize,
    pub learning_rate: f64,
    pub warmup_ratio: f64,
    pub max_grad_norm: f64,
    pub weight_decay: f64,
    pub suffix: Option<String>,
    // Learning rate scheduler
    pub lr_scheduler: TogetherLRScheduler,
    // Weights & Biases integration
    pub wandb_api_key: Option<String>,
    pub wandb_base_url: Option<String>,
    pub wandb_project_name: Option<String>,
    pub wandb_name: Option<String>,
    // Training method
    pub training_method: TogetherTrainingMethod,
    // Training type
    pub training_type: TogetherTrainingType,
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
    #[serde(default = "default_n_epochs")]
    pub n_epochs: u32,
    #[serde(default = "default_n_checkpoints")]
    pub n_checkpoints: u32,
    pub n_evals: Option<u32>, // Keep as Option due to conditional logic
    #[serde(default)]
    pub batch_size: TogetherBatchSize,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_warmup_ratio")]
    pub warmup_ratio: f64,
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    pub suffix: Option<String>,
    // Learning rate scheduler - nested like Together API
    #[serde(default)]
    pub lr_scheduler: TogetherLRScheduler,
    // Weights & Biases integration
    pub wandb_api_key: Option<String>,
    pub wandb_base_url: Option<String>,
    pub wandb_project_name: Option<String>,
    pub wandb_name: Option<String>,
    // Training method - nested like Together API
    #[serde(default)]
    pub training_method: TogetherTrainingMethod,
    // Training type - nested like Together API
    #[serde(default)]
    pub training_type: TogetherTrainingType,
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
    #[pyo3(signature = (*, model, credentials=None, api_base=None, n_epochs=None, n_checkpoints=None, n_evals=None, batch_size=None, learning_rate=None, warmup_ratio=None, max_grad_norm=None, weight_decay=None, suffix=None, lr_scheduler=None, wandb_api_key=None, wandb_base_url=None, wandb_project_name=None, wandb_name=None, training_method=None, training_type=None, from_checkpoint=None, from_hf_model=None, hf_model_revision=None, hf_api_token=None, hf_output_repo_name=None))]
    pub fn new(
        py: Python,
        model: String,
        credentials: Option<String>,
        api_base: Option<String>,
        n_epochs: Option<u32>,
        n_checkpoints: Option<u32>,
        n_evals: Option<u32>,
        batch_size: Option<&Bound<'_, PyAny>>,
        learning_rate: Option<f64>,
        warmup_ratio: Option<f64>,
        max_grad_norm: Option<f64>,
        weight_decay: Option<f64>,
        suffix: Option<String>,
        lr_scheduler: Option<&Bound<'_, PyAny>>,
        wandb_api_key: Option<String>,
        wandb_base_url: Option<String>,
        wandb_project_name: Option<String>,
        wandb_name: Option<String>,
        training_method: Option<&Bound<'_, PyAny>>,
        training_type: Option<&Bound<'_, PyAny>>,
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
        // Deserialize lr_scheduler from Python dict to Rust TogetherLRScheduler
        let lr_scheduler: TogetherLRScheduler = if let Some(ls) = lr_scheduler {
            if let Ok(lr_scheduler) = ls.extract::<TogetherLRScheduler>() {
                // If it's already a TogetherLRScheduler object, use it directly
                lr_scheduler
            } else {
                // Otherwise, try to deserialize from a Python dict
                deserialize_from_pyobj(py, ls)?
            }
        } else {
            TogetherLRScheduler::default()
        };

        // Deserialize training_method from Python dict to Rust TogetherTrainingMethod
        let training_method: TogetherTrainingMethod = if let Some(tm) = training_method {
            if let Ok(training_method) = tm.extract::<TogetherTrainingMethod>() {
                // If it's already a TogetherTrainingMethod object, use it directly
                training_method
            } else {
                // Otherwise, try to deserialize from a Python dict
                deserialize_from_pyobj(py, tm)?
            }
        } else {
            TogetherTrainingMethod::default()
        };

        // Deserialize training_type from Python dict to Rust TogetherTrainingType
        let training_type: TogetherTrainingType = if let Some(tt) = training_type {
            if let Ok(training_type) = tt.extract::<TogetherTrainingType>() {
                // If it's already a TogetherTrainingType object, use it directly
                training_type
            } else {
                // Otherwise, try to deserialize from a Python dict
                deserialize_from_pyobj(py, tt)?
            }
        } else {
            TogetherTrainingType::default()
        };

        let batch_size: TogetherBatchSize = if let Some(bs) = batch_size {
            if let Ok(batch_size) = bs.extract::<TogetherBatchSize>() {
                batch_size
            } else {
                deserialize_from_pyobj(py, bs)?
            }
        } else {
            TogetherBatchSize::default()
        };

        Ok(Self {
            model,
            credentials,
            api_base,
            n_epochs: n_epochs.unwrap_or_else(default_n_epochs),
            n_checkpoints: n_checkpoints.unwrap_or_else(default_n_checkpoints),
            n_evals,
            batch_size,
            learning_rate: learning_rate.unwrap_or_else(default_learning_rate),
            warmup_ratio: warmup_ratio.unwrap_or_else(default_warmup_ratio),
            max_grad_norm: max_grad_norm.unwrap_or_else(default_max_grad_norm),
            weight_decay: weight_decay.unwrap_or_else(default_weight_decay),
            suffix,
            lr_scheduler,
            wandb_api_key,
            wandb_base_url,
            wandb_project_name,
            wandb_name,
            training_method,
            training_type,
            from_checkpoint,
            from_hf_model,
            hf_model_revision,
            hf_api_token,
            hf_output_repo_name,
        })
    }

    /// Initialize the TogetherSFTConfig. All parameters are optional except for `model`.
    ///
    /// For detailed parameter documentation, see: https://docs.together.ai/reference/post-fine-tunes
    ///
    /// :param model: Name of the base model to run fine-tune job on.
    /// :param credentials: The credentials to use for the fine-tuning job. This should be a string like "env::TOGETHER_API_KEY". See docs for more details.
    /// :param api_base: The base URL to use for the fine-tuning job. This is primarily used for testing.
    /// :param n_epochs: Number of complete passes through the training dataset. Default: 1. Higher values may improve results but increase cost and overfitting risk.
    /// :param n_checkpoints: Number of intermediate model versions saved during training. Default: 1.
    /// :param n_evals: Number of evaluations to be run on a given validation set during training. Default: 0.
    /// :param batch_size: Number of training examples processed together (larger batches use more memory but may train faster). Defaults to "max". Together uses training optimizations like packing, so the effective batch size may be different than the value you set.
    /// :param learning_rate: Controls how quickly the model adapts to new information. Default: 0.00001. Too high may cause instability, too low may slow convergence.
    /// :param warmup_ratio: Percent of steps at the start of training to linearly increase learning rate. Default: 0.
    /// :param max_grad_norm: Max gradient norm for gradient clipping. Default: 1. Set to 0 to disable.
    /// :param weight_decay: Regularization parameter for the optimizer. Default: 0.
    /// :param suffix: Suffix that will be added to your fine-tuned model name.
    /// :param lr_scheduler: Learning rate scheduler configuration as a dictionary. For linear: {'lr_scheduler_type': 'linear', 'lr_scheduler_args': {'min_lr_ratio': 0.0}}. For cosine: {'lr_scheduler_type': 'cosine', 'lr_scheduler_args': {'min_lr_ratio': 0.0, 'num_cycles': 0.5}}.
    /// :param wandb_api_key: Weights & Biases API key for experiment tracking.
    /// :param wandb_base_url: Weights & Biases base URL for dedicated instance.
    /// :param wandb_project_name: Weights & Biases project name. Default: 'together'.
    /// :param wandb_name: Weights & Biases run name.
    /// :param training_method: Training method configuration as a dictionary with 'method' and 'train_on_inputs'.
    /// :param training_type: Training type configuration as a dictionary. For 'full': {'type': 'full'}. For 'lora': {'type': 'lora', 'r': 8, 'alpha': 32, 'dropout': 0.0, 'trainable_modules': 'all-linear'}.
    /// :param from_checkpoint: Continue training from a previous checkpoint job ID.
    /// :param from_hf_model: Start training from a Hugging Face model repository.
    /// :param hf_model_revision: Specific model version/commit from Hugging Face repository.
    /// :param hf_api_token: Hugging Face API token for authentication.
    /// :param hf_output_repo_name: Hugging Face repository name for uploading the fine-tuned model.
    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, credentials=None, api_base=None, n_epochs=None, n_checkpoints=None, n_evals=None, batch_size=None, learning_rate=None, warmup_ratio=None, max_grad_norm=None, weight_decay=None, suffix=None, lr_scheduler=None, wandb_api_key=None, wandb_base_url=None, wandb_project_name=None, wandb_name=None, training_method=None, training_type=None, from_checkpoint=None, from_hf_model=None, hf_model_revision=None, hf_api_token=None, hf_output_repo_name=None))]
    fn __init__(
        this: Py<Self>,
        model: String,
        credentials: Option<String>,
        api_base: Option<String>,
        n_epochs: Option<u32>,
        n_checkpoints: Option<u32>,
        n_evals: Option<u32>,
        batch_size: Option<&Bound<'_, PyAny>>,
        learning_rate: Option<f64>,
        warmup_ratio: Option<f64>,
        max_grad_norm: Option<f64>,
        weight_decay: Option<f64>,
        suffix: Option<String>,
        lr_scheduler: Option<&Bound<'_, PyAny>>,
        wandb_api_key: Option<String>,
        wandb_base_url: Option<String>,
        wandb_project_name: Option<String>,
        wandb_name: Option<String>,
        training_method: Option<&Bound<'_, PyAny>>,
        training_type: Option<&Bound<'_, PyAny>>,
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
            lr_scheduler: self.lr_scheduler,
            // Weights & Biases integration
            wandb_api_key: self.wandb_api_key,
            wandb_base_url: self.wandb_base_url,
            wandb_project_name: self.wandb_project_name,
            wandb_name: self.wandb_name,
            // Training method
            training_method: self.training_method,
            // Training type
            training_type: self.training_type,
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

// Nested configuration structs that match Together's API format
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass)]
#[serde(tag = "lr_scheduler_type", rename_all = "snake_case")]
pub enum TogetherLRScheduler {
    Linear {
        #[serde(default)]
        min_lr_ratio: f64,
    },
    Cosine {
        #[serde(default)]
        min_lr_ratio: f64,
        #[serde(default)]
        num_cycles: f64,
    },
}

impl Default for TogetherLRScheduler {
    fn default() -> Self {
        Self::Linear { min_lr_ratio: 0.0 }
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass)]
#[serde(tag = "type")]
pub enum TogetherTrainingType {
    Full {},
    Lora {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        lora_r: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        lora_alpha: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        lora_dropout: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none", default)]
        lora_trainable_modules: Option<String>,
    },
}

impl Default for TogetherTrainingType {
    fn default() -> Self {
        Self::Lora {
            lora_r: Some(8),
            lora_alpha: Some(16),
            lora_dropout: Some(0.0),
            lora_trainable_modules: Some("all-linear".to_string()),
        }
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum TogetherTrainingMethod {
    #[serde(rename = "sft")]
    Sft {
        #[serde(skip_serializing_if = "Option::is_none", default)]
        train_on_inputs: Option<String>,
    },
}

impl Default for TogetherTrainingMethod {
    fn default() -> Self {
        Self::Sft {
            train_on_inputs: Some("auto".to_string()),
        }
    }
}

impl Optimizer for TogetherSFTConfig {
    type Handle = TogetherSFTJobHandle;

    async fn launch(
        &self,
        client: &reqwest::Client,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
        _clickhouse_connection_info: &ClickHouseConnectionInfo,
        _config: &Config,
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
            .post(self.api_base.join("fine-tunes").convert_parse_error()?)
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
