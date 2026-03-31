use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tensorzero_types::inference_params::{JsonMode, ServiceTier};
use uuid::Uuid;

use crate::{
    StoredExtraBodyConfig, StoredExtraHeadersConfig, StoredPromptRef, StoredRetryConfig,
    StoredTimeoutsConfig,
};

/// Reference to a `variant_configs` row.
/// Replaces inline variant configs in stored function config.
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredVariantRef {
    pub variant_version_id: Uuid,
}

/// Wrapper stored in `variant_configs.config`.
/// Uses an explicit `variant` field instead of `#[serde(flatten)]`.
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredVariantVersionConfig {
    pub variant: StoredVariantConfig,
    pub timeouts: Option<StoredTimeoutsConfig>,
    pub namespace: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "config")]
pub enum StoredVariantConfig {
    #[serde(rename = "chat_completion")]
    ChatCompletion(StoredChatCompletionVariantConfig),
    #[serde(rename = "experimental_best_of_n_sampling")]
    BestOfNSampling(StoredBestOfNVariantConfig),
    #[serde(rename = "experimental_mixture_of_n")]
    MixtureOfN(StoredMixtureOfNVariantConfig),
    #[serde(rename = "experimental_dynamic_in_context_learning")]
    Dicl(StoredDiclVariantConfig),
    #[serde(rename = "experimental_chain_of_thought")]
    ChainOfThought(StoredChatCompletionVariantConfig),
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredChatCompletionVariantConfig {
    pub weight: Option<f64>,
    pub model: Arc<str>,
    pub system_template: Option<StoredPromptRef>,
    pub user_template: Option<StoredPromptRef>,
    pub assistant_template: Option<StoredPromptRef>,
    pub input_wrappers: Option<StoredInputWrappers>,
    pub templates: Option<HashMap<String, StoredPromptRef>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub json_mode: Option<JsonMode>,
    pub stop_sequences: Option<Vec<String>>,
    pub reasoning_effort: Option<String>,
    pub service_tier: Option<ServiceTier>,
    pub thinking_budget_tokens: Option<i32>,
    pub verbosity: Option<String>,
    pub retries: Option<StoredRetryConfig>,
    pub extra_body: Option<StoredExtraBodyConfig>,
    pub extra_headers: Option<StoredExtraHeadersConfig>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredInputWrappers {
    pub user: Option<StoredPromptRef>,
    pub assistant: Option<StoredPromptRef>,
    pub system: Option<StoredPromptRef>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredBestOfNVariantConfig {
    pub weight: Option<f64>,
    pub timeout_s: Option<f64>,
    pub candidates: Option<Vec<String>>,
    pub evaluator: StoredChatCompletionVariantConfig,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredMixtureOfNVariantConfig {
    pub weight: Option<f64>,
    pub timeout_s: Option<f64>,
    pub candidates: Option<Vec<String>>,
    pub fuser: StoredChatCompletionVariantConfig,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredDiclVariantConfig {
    pub weight: Option<f64>,
    pub embedding_model: String,
    pub k: u32,
    pub model: String,
    pub system_instructions: Option<StoredPromptRef>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub json_mode: Option<JsonMode>,
    pub stop_sequences: Option<Vec<String>>,
    pub reasoning_effort: Option<String>,
    pub thinking_budget_tokens: Option<i32>,
    pub verbosity: Option<String>,
    pub max_distance: Option<f32>,
    pub retries: Option<StoredRetryConfig>,
    pub extra_body: Option<StoredExtraBodyConfig>,
    pub extra_headers: Option<StoredExtraHeadersConfig>,
}
