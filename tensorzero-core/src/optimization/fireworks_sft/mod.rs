#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

fn default_deploy_after_training() -> bool {
    false
}

/// Initialized Fireworks SFT Config (per-job settings only).
/// Provider-level settings (account_id, credentials) come from
/// `provider_types.fireworks.sft` in the gateway config.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
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
    #[serde(default = "default_deploy_after_training")]
    pub deploy_after_training: bool,
}

/// Uninitialized Fireworks SFT Config (per-job settings only).
/// Provider-level settings (account_id, credentials) come from
/// `provider_types.fireworks.sft` in the gateway config.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
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
    #[serde(default)]
    pub deploy_after_training: Option<bool>,
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
    /// Initialize the FireworksSFTConfig.
    ///
    /// Provider-level settings (account_id, credentials) are configured in the gateway config at
    /// `[provider_types.fireworks.sft]`.
    ///
    /// :param model: The model to use for the fine-tuning job (required).
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
    #[new]
    #[pyo3(signature = (*, model, early_stop=None, epochs=None, learning_rate=None, max_context_length=None, lora_rank=None, batch_size=None, display_name=None, output_model=None, warm_start_from=None, is_turbo=None, eval_auto_carveout=None, nodes=None, mtp_enabled=None, mtp_num_draft_tokens=None, mtp_freeze_base_model=None, deploy_after_training=None))]
    #[expect(clippy::too_many_arguments)]
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
        deploy_after_training: Option<bool>,
    ) -> PyResult<Self> {
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
            deploy_after_training,
        })
    }

    #[expect(unused_variables, clippy::too_many_arguments)]
    #[pyo3(signature = (*, model, early_stop=None, epochs=None, learning_rate=None, max_context_length=None, lora_rank=None, batch_size=None, display_name=None, output_model=None, warm_start_from=None, is_turbo=None, eval_auto_carveout=None, nodes=None, mtp_enabled=None, mtp_num_draft_tokens=None, mtp_freeze_base_model=None, deploy_after_training=None))]
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
        deploy_after_training: Option<bool>,
    ) -> Py<Self> {
        this
    }
}

impl UninitializedFireworksSFTConfig {
    pub fn load(self) -> FireworksSFTConfig {
        FireworksSFTConfig {
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
            deploy_after_training: self
                .deploy_after_training
                .unwrap_or_else(default_deploy_after_training),
        }
    }
}

/// Minimal job handle for Fireworks SFT.
/// All configuration needed for polling comes from provider_types at poll time.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct FireworksSFTJobHandle {
    pub job_url: Url,
    pub job_path: String,
    #[serde(default = "default_deploy_after_training")]
    pub deploy_after_training: bool,
}

impl std::fmt::Display for FireworksSFTJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}
