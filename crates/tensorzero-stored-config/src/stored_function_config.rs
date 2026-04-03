use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tensorzero_types::ToolChoice;

use crate::{StoredEvaluatorConfig, StoredPromptRef, StoredVariantRef};

pub const STORED_FUNCTION_CONFIG_SCHEMA_REVISION: i32 = 1;

/// Stored in `function_configs.config`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum StoredFunctionConfig {
    Chat(StoredChatFunctionConfig),
    Json(StoredJsonFunctionConfig),
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredChatFunctionConfig {
    pub variants: Option<HashMap<String, StoredVariantRef>>,
    pub system_schema: Option<StoredPromptRef>,
    pub user_schema: Option<StoredPromptRef>,
    pub assistant_schema: Option<StoredPromptRef>,
    pub schemas: Option<HashMap<String, StoredPromptRef>>,
    pub tools: Option<Vec<String>>,
    pub tool_choice: Option<StoredToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub description: Option<String>,
    pub experimentation: Option<StoredExperimentationConfigWithNamespaces>,
    pub evaluators: Option<HashMap<String, StoredEvaluatorConfig>>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredJsonFunctionConfig {
    pub variants: Option<HashMap<String, StoredVariantRef>>,
    pub system_schema: Option<StoredPromptRef>,
    pub user_schema: Option<StoredPromptRef>,
    pub assistant_schema: Option<StoredPromptRef>,
    pub schemas: Option<HashMap<String, StoredPromptRef>>,
    pub output_schema: Option<StoredPromptRef>,
    pub description: Option<String>,
    pub experimentation: Option<StoredExperimentationConfigWithNamespaces>,
    pub evaluators: Option<HashMap<String, StoredEvaluatorConfig>>,
}

// --- ToolChoice ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum StoredToolChoice {
    None,
    Auto,
    Required,
    Specific { name: String },
}

impl From<&ToolChoice> for StoredToolChoice {
    fn from(value: &ToolChoice) -> Self {
        match value {
            ToolChoice::None => Self::None,
            ToolChoice::Auto => Self::Auto,
            ToolChoice::Required => Self::Required,
            ToolChoice::Specific(tool_name) => Self::Specific {
                name: tool_name.clone(),
            },
        }
    }
}

// --- Experimentation ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredExperimentationConfigWithNamespaces {
    pub base: StoredExperimentationConfig,
    pub namespaces: Option<HashMap<String, StoredExperimentationConfig>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredExperimentationConfig {
    Static(StoredStaticExperimentationConfig),
    Adaptive(StoredAdaptiveExperimentationConfig),
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredStaticExperimentationConfig {
    /// Always stored as a map of variant name → weight.
    pub candidate_variants: Option<HashMap<String, f64>>,
    pub fallback_variants: Option<Vec<String>>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredAdaptiveExperimentationConfig {
    pub algorithm: Option<StoredAdaptiveExperimentationAlgorithm>,
    pub metric: String,
    pub candidate_variants: Option<Vec<String>>,
    pub fallback_variants: Option<Vec<String>>,
    pub min_samples_per_variant: Option<u64>,
    pub delta: Option<f64>,
    pub epsilon: Option<f64>,
    pub update_period_s: Option<u64>,
    pub min_prob: Option<f64>,
    pub max_samples_per_variant: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredAdaptiveExperimentationAlgorithm {
    TrackAndStop,
}
