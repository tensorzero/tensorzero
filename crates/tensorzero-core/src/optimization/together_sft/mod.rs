#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::deserialize_from_pyobj;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tensorzero_derive::TensorZeroDeserialize;
use tensorzero_stored_config::{
    StoredTogetherBatchSize, StoredTogetherLRScheduler, StoredTogetherOptimizerSFTConfig,
    StoredTogetherTrainingMethod, StoredTogetherTrainingType,
};
use url::Url;

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

#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass)]
#[serde(rename_all = "lowercase")]
pub enum TogetherBatchSizeDescription {
    Max,
}

#[derive(ts_rs::TS, Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
#[ts(export)]
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

/// Initialized Together SFT Config (per-job settings only).
/// Provider-level settings (credentials, wandb, hf_api_token) come from
/// `provider_types.together` in the gateway config.
#[derive(ts_rs::TS, Debug, Clone, Serialize)]
#[ts(export, optional_fields)]
pub struct TogetherSFTConfig {
    pub model: String,
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
    // Weights & Biases run name (per-job, not the wandb API key/project which are provider-level)
    pub wandb_name: Option<String>,
    // Training method
    pub training_method: TogetherTrainingMethod,
    // Training type
    pub training_type: TogetherTrainingType,
    // Advanced options
    pub from_checkpoint: Option<String>,
    pub from_hf_model: Option<String>,
    pub hf_model_revision: Option<String>,
    pub hf_output_repo_name: Option<String>,
}

/// Minimal job handle for Together SFT.
/// All configuration needed for polling comes from provider_types at poll time.
#[derive(ts_rs::TS, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct TogetherSFTJobHandle {
    pub job_id: String,
    /// A url to a human-readable page for the job.
    pub job_url: Url,
}

impl std::fmt::Display for TogetherSFTJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Uninitialized Together SFT Config (per-job settings only).
/// Provider-level settings (credentials, wandb, hf_api_token) come from
/// `provider_types.together` in the gateway config.
#[derive(ts_rs::TS, Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[ts(export, optional_fields)]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "TogetherSFTConfig"))]
pub struct UninitializedTogetherSFTConfig {
    pub model: String,
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
    // Weights & Biases run name (per-job, not the wandb API key/project which are provider-level)
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
    pub hf_output_repo_name: Option<String>,
}

impl std::fmt::Display for UninitializedTogetherSFTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl From<StoredTogetherBatchSize> for TogetherBatchSize {
    fn from(stored: StoredTogetherBatchSize) -> Self {
        match stored {
            StoredTogetherBatchSize::Number { value } => Self::Number(value),
            StoredTogetherBatchSize::Max => Self::Description(TogetherBatchSizeDescription::Max),
        }
    }
}

impl From<StoredTogetherLRScheduler> for TogetherLRScheduler {
    fn from(stored: StoredTogetherLRScheduler) -> Self {
        match stored {
            StoredTogetherLRScheduler::Linear { min_lr_ratio } => Self::Linear {
                min_lr_ratio: min_lr_ratio.unwrap_or_default(),
            },
            StoredTogetherLRScheduler::Cosine {
                min_lr_ratio,
                num_cycles,
            } => Self::Cosine {
                min_lr_ratio: min_lr_ratio.unwrap_or_default(),
                num_cycles: num_cycles.unwrap_or_default(),
            },
        }
    }
}

impl From<StoredTogetherTrainingType> for TogetherTrainingType {
    fn from(stored: StoredTogetherTrainingType) -> Self {
        match stored {
            StoredTogetherTrainingType::Full => Self::Full {},
            StoredTogetherTrainingType::Lora {
                lora_r,
                lora_alpha,
                lora_dropout,
                lora_trainable_modules,
            } => Self::Lora {
                lora_r,
                lora_alpha,
                lora_dropout,
                lora_trainable_modules,
            },
        }
    }
}

impl From<StoredTogetherTrainingMethod> for TogetherTrainingMethod {
    fn from(stored: StoredTogetherTrainingMethod) -> Self {
        match stored {
            StoredTogetherTrainingMethod::Sft { train_on_inputs } => Self::Sft { train_on_inputs },
        }
    }
}

// --- Reverse conversions: core -> stored ---

impl From<TogetherBatchSize> for StoredTogetherBatchSize {
    fn from(batch_size: TogetherBatchSize) -> Self {
        match batch_size {
            TogetherBatchSize::Number(value) => Self::Number { value },
            TogetherBatchSize::Description(TogetherBatchSizeDescription::Max) => Self::Max,
        }
    }
}

impl From<TogetherLRScheduler> for StoredTogetherLRScheduler {
    fn from(scheduler: TogetherLRScheduler) -> Self {
        match scheduler {
            TogetherLRScheduler::Linear { min_lr_ratio } => Self::Linear {
                min_lr_ratio: Some(min_lr_ratio),
            },
            TogetherLRScheduler::Cosine {
                min_lr_ratio,
                num_cycles,
            } => Self::Cosine {
                min_lr_ratio: Some(min_lr_ratio),
                num_cycles: Some(num_cycles),
            },
        }
    }
}

impl From<TogetherTrainingType> for StoredTogetherTrainingType {
    fn from(training_type: TogetherTrainingType) -> Self {
        match training_type {
            TogetherTrainingType::Full {} => Self::Full,
            TogetherTrainingType::Lora {
                lora_r,
                lora_alpha,
                lora_dropout,
                lora_trainable_modules,
            } => Self::Lora {
                lora_r,
                lora_alpha,
                lora_dropout,
                lora_trainable_modules,
            },
        }
    }
}

impl From<TogetherTrainingMethod> for StoredTogetherTrainingMethod {
    fn from(method: TogetherTrainingMethod) -> Self {
        match method {
            TogetherTrainingMethod::Sft { train_on_inputs } => Self::Sft { train_on_inputs },
        }
    }
}

impl From<StoredTogetherOptimizerSFTConfig> for UninitializedTogetherSFTConfig {
    fn from(stored: StoredTogetherOptimizerSFTConfig) -> Self {
        UninitializedTogetherSFTConfig {
            model: stored.model,
            n_epochs: stored.n_epochs.unwrap_or_else(default_n_epochs),
            n_checkpoints: stored.n_checkpoints.unwrap_or_else(default_n_checkpoints),
            n_evals: stored.n_evals,
            batch_size: stored.batch_size.map(Into::into).unwrap_or_default(),
            learning_rate: stored.learning_rate.unwrap_or_else(default_learning_rate),
            warmup_ratio: stored.warmup_ratio.unwrap_or_else(default_warmup_ratio),
            max_grad_norm: stored.max_grad_norm.unwrap_or_else(default_max_grad_norm),
            weight_decay: stored.weight_decay.unwrap_or_else(default_weight_decay),
            suffix: stored.suffix,
            lr_scheduler: stored.lr_scheduler.map(Into::into).unwrap_or_default(),
            wandb_name: stored.wandb_name,
            training_method: stored.training_method.map(Into::into).unwrap_or_default(),
            training_type: stored.training_type.map(Into::into).unwrap_or_default(),
            from_checkpoint: stored.from_checkpoint,
            from_hf_model: stored.from_hf_model,
            hf_model_revision: stored.hf_model_revision,
            hf_output_repo_name: stored.hf_output_repo_name,
        }
    }
}

impl From<UninitializedTogetherSFTConfig> for StoredTogetherOptimizerSFTConfig {
    fn from(config: UninitializedTogetherSFTConfig) -> Self {
        StoredTogetherOptimizerSFTConfig {
            model: config.model,
            n_epochs: Some(config.n_epochs),
            n_checkpoints: Some(config.n_checkpoints),
            n_evals: config.n_evals,
            batch_size: Some(config.batch_size.into()),
            learning_rate: Some(config.learning_rate),
            warmup_ratio: Some(config.warmup_ratio),
            max_grad_norm: Some(config.max_grad_norm),
            weight_decay: Some(config.weight_decay),
            suffix: config.suffix,
            lr_scheduler: Some(config.lr_scheduler.into()),
            wandb_name: config.wandb_name,
            training_method: Some(config.training_method.into()),
            training_type: Some(config.training_type.into()),
            from_checkpoint: config.from_checkpoint,
            from_hf_model: config.from_hf_model,
            hf_model_revision: config.hf_model_revision,
            hf_output_repo_name: config.hf_output_repo_name,
        }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UninitializedTogetherSFTConfig {
    /// Initialize the TogetherSFTConfig.
    ///
    /// Provider-level settings (credentials, wandb API key/base URL/project name, hf_api_token) are configured
    /// in the gateway config at `[provider_types.together.sft]`.
    ///
    /// :param model: Name of the base model to run fine-tune job on (required).
    /// :param n_epochs: Number of complete passes through the training dataset. Default: 1. Higher values may improve results but increase cost and overfitting risk.
    /// :param n_checkpoints: Number of intermediate model versions saved during training. Default: 1.
    /// :param n_evals: Number of evaluations to be run on a given validation set during training. Default: 0.
    /// :param batch_size: Number of training examples processed together (larger batches use more memory but may train faster). Defaults to "max". Together uses training optimizations like packing, so the effective batch size may be different than the value you set.
    /// :param learning_rate: Controls how quickly the model adapts to new information. Default: 0.00001. Too high may cause instability, too low may slow convergence.
    /// :param warmup_ratio: Percent of steps at the start of training to linearly increase learning rate. Default: 0.
    /// :param max_grad_norm: Max gradient norm for gradient clipping. Default: 1. Set to 0 to disable.
    /// :param weight_decay: Regularization parameter for the optimizer. Default: 0.
    /// :param suffix: Suffix that will be added to your fine-tuned model name.
    /// :param lr_scheduler: Learning rate scheduler configuration as a dictionary. For linear: {'lr_scheduler_type': 'linear', 'min_lr_ratio': 0.0}. For cosine: {'lr_scheduler_type': 'cosine', 'min_lr_ratio': 0.0, 'num_cycles': 0.5}.
    /// :param wandb_name: Weights & Biases run name (per-job setting; API key/project/base URL are configured at provider level).
    /// :param training_method: Training method configuration as a dictionary with 'method' and 'train_on_inputs'.
    /// :param training_type: Training type configuration as a dictionary. For 'full': {'type': 'full'}. For 'lora': {'type': 'lora', 'lora_r': 8, 'lora_alpha': 32, 'lora_dropout': 0.0, 'lora_trainable_modules': 'all-linear'}.
    /// :param from_checkpoint: Continue training from a previous checkpoint job ID.
    /// :param from_hf_model: Start training from a Hugging Face model repository.
    /// :param hf_model_revision: Specific model version/commit from Hugging Face repository.
    /// :param hf_output_repo_name: Hugging Face repository name for uploading the fine-tuned model (hf_api_token is configured at provider level).
    #[expect(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, model, n_epochs=None, n_checkpoints=None, n_evals=None, batch_size=None, learning_rate=None, warmup_ratio=None, max_grad_norm=None, weight_decay=None, suffix=None, lr_scheduler=None, wandb_name=None, training_method=None, training_type=None, from_checkpoint=None, from_hf_model=None, hf_model_revision=None, hf_output_repo_name=None))]
    pub fn new(
        py: Python,
        model: String,
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
        wandb_name: Option<String>,
        training_method: Option<&Bound<'_, PyAny>>,
        training_type: Option<&Bound<'_, PyAny>>,
        from_checkpoint: Option<String>,
        from_hf_model: Option<String>,
        hf_model_revision: Option<String>,
        hf_output_repo_name: Option<String>,
    ) -> PyResult<Self> {
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
            wandb_name,
            training_method,
            training_type,
            from_checkpoint,
            from_hf_model,
            hf_model_revision,
            hf_output_repo_name,
        })
    }

    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, n_epochs=None, n_checkpoints=None, n_evals=None, batch_size=None, learning_rate=None, warmup_ratio=None, max_grad_norm=None, weight_decay=None, suffix=None, lr_scheduler=None, wandb_name=None, training_method=None, training_type=None, from_checkpoint=None, from_hf_model=None, hf_model_revision=None, hf_output_repo_name=None))]
    fn __init__(
        this: Py<Self>,
        model: String,
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
        wandb_name: Option<String>,
        training_method: Option<&Bound<'_, PyAny>>,
        training_type: Option<&Bound<'_, PyAny>>,
        from_checkpoint: Option<String>,
        from_hf_model: Option<String>,
        hf_model_revision: Option<String>,
        hf_output_repo_name: Option<String>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedTogetherSFTConfig {
    pub fn load(self) -> TogetherSFTConfig {
        TogetherSFTConfig {
            model: self.model,
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
            // Weights & Biases run name
            wandb_name: self.wandb_name,
            // Training method
            training_method: self.training_method,
            // Training type
            training_type: self.training_type,
            // Advanced options
            from_checkpoint: self.from_checkpoint,
            from_hf_model: self.from_hf_model,
            hf_model_revision: self.hf_model_revision,
            hf_output_repo_name: self.hf_output_repo_name,
        }
    }
}

// Nested configuration structs that match Together's API format
#[derive(ts_rs::TS, Clone, Debug, JsonSchema, PartialEq, Serialize, TensorZeroDeserialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass)]
#[serde(tag = "lr_scheduler_type")]
#[serde(rename_all = "snake_case")]
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

#[derive(ts_rs::TS, Clone, Debug, JsonSchema, PartialEq, Serialize, TensorZeroDeserialize)]
#[ts(export)]
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

#[derive(ts_rs::TS, Clone, Debug, JsonSchema, PartialEq, Serialize, TensorZeroDeserialize)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass)]
#[serde(tag = "method")]
#[serde(rename_all = "snake_case")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    fn sample_full_config() -> UninitializedTogetherSFTConfig {
        UninitializedTogetherSFTConfig {
            model: "meta-llama/Llama-3-8b".to_string(),
            n_epochs: 4,
            n_checkpoints: 2,
            n_evals: Some(3),
            batch_size: TogetherBatchSize::Number(64),
            learning_rate: 1e-5,
            warmup_ratio: 0.05,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            suffix: Some("my-suffix".to_string()),
            lr_scheduler: TogetherLRScheduler::Cosine {
                min_lr_ratio: 0.1,
                num_cycles: 0.5,
            },
            wandb_name: Some("my-run".to_string()),
            training_method: TogetherTrainingMethod::Sft {
                train_on_inputs: Some("true".to_string()),
            },
            training_type: TogetherTrainingType::Lora {
                lora_r: Some(8),
                lora_alpha: Some(16),
                lora_dropout: Some(0.0),
                lora_trainable_modules: Some("all-linear".to_string()),
            },
            from_checkpoint: Some("ckpt-id".to_string()),
            from_hf_model: Some("hf-model".to_string()),
            hf_model_revision: Some("rev".to_string()),
            hf_output_repo_name: Some("repo".to_string()),
        }
    }

    #[gtest]
    fn test_together_sft_config_round_trip_full() {
        let original = sample_full_config();
        let stored: StoredTogetherOptimizerSFTConfig = original.clone().into();
        let restored: UninitializedTogetherSFTConfig = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_together_sft_config_round_trip_full_training() {
        let mut original = sample_full_config();
        original.training_type = TogetherTrainingType::Full {};
        let stored: StoredTogetherOptimizerSFTConfig = original.clone().into();
        let restored: UninitializedTogetherSFTConfig = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_together_sft_config_round_trip_linear_scheduler_max_batch() {
        let mut original = sample_full_config();
        original.lr_scheduler = TogetherLRScheduler::Linear { min_lr_ratio: 0.2 };
        original.batch_size = TogetherBatchSize::Description(TogetherBatchSizeDescription::Max);
        let stored: StoredTogetherOptimizerSFTConfig = original.clone().into();
        let restored: UninitializedTogetherSFTConfig = stored.into();
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_together_batch_size_round_trip() {
        for original in [
            TogetherBatchSize::Number(32),
            TogetherBatchSize::Description(TogetherBatchSizeDescription::Max),
        ] {
            let stored: StoredTogetherBatchSize = original.clone().into();
            let restored: TogetherBatchSize = stored.into();
            expect_that!(restored, eq(&original));
        }
    }

    #[gtest]
    fn test_together_lr_scheduler_round_trip() {
        for original in [
            TogetherLRScheduler::Linear { min_lr_ratio: 0.0 },
            TogetherLRScheduler::Cosine {
                min_lr_ratio: 0.1,
                num_cycles: 0.5,
            },
        ] {
            let stored: StoredTogetherLRScheduler = original.clone().into();
            let restored: TogetherLRScheduler = stored.into();
            expect_that!(restored, eq(&original));
        }
    }

    #[gtest]
    fn test_together_training_type_round_trip() {
        for original in [
            TogetherTrainingType::Full {},
            TogetherTrainingType::Lora {
                lora_r: Some(4),
                lora_alpha: Some(8),
                lora_dropout: Some(0.1),
                lora_trainable_modules: Some("all-linear".to_string()),
            },
        ] {
            let stored: StoredTogetherTrainingType = original.clone().into();
            let restored: TogetherTrainingType = stored.into();
            expect_that!(restored, eq(&original));
        }
    }

    #[gtest]
    fn test_together_training_method_round_trip() {
        let original = TogetherTrainingMethod::Sft {
            train_on_inputs: Some("auto".to_string()),
        };
        let stored: StoredTogetherTrainingMethod = original.clone().into();
        let restored: TogetherTrainingMethod = stored.into();
        expect_that!(restored, eq(&original));
    }
}
