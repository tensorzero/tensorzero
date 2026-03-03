use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::config::path::ResolvedTomlPathData;
use crate::config::{
    NonStreamingTimeouts, StreamingTimeouts, TimeoutsConfig, UninitializedFunctionConfig,
    UninitializedFunctionConfigChat, UninitializedFunctionConfigJson, UninitializedSchemas,
    UninitializedVariantInfo,
};
use crate::experimentation::UninitializedExperimentationConfigWithNamespaces;
use crate::tool::ToolChoice;

use super::variant_config::{StoredVariantConfig, StoredVariantInfo};

/// Stored version of `UninitializedFunctionConfig`.
///
/// Uses `StoredVariantInfo` so that variants with the deprecated `timeout_s`
/// field can still be deserialized from historical config snapshots.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum StoredFunctionConfig {
    Chat(StoredFunctionConfigChat),
    Json(StoredFunctionConfigJson),
}

/// Stored version of `UninitializedFunctionConfigChat`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredFunctionConfigChat {
    pub variants: HashMap<String, StoredVariantInfo>,
    pub system_schema: Option<ResolvedTomlPathData>,
    pub user_schema: Option<ResolvedTomlPathData>,
    pub assistant_schema: Option<ResolvedTomlPathData>,
    #[serde(default)]
    pub schemas: UninitializedSchemas,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub tool_choice: ToolChoice,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub description: Option<String>,
    pub experimentation: Option<UninitializedExperimentationConfigWithNamespaces>,
}

/// Stored version of `UninitializedFunctionConfigJson`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredFunctionConfigJson {
    pub variants: HashMap<String, StoredVariantInfo>,
    pub system_schema: Option<ResolvedTomlPathData>,
    pub user_schema: Option<ResolvedTomlPathData>,
    pub assistant_schema: Option<ResolvedTomlPathData>,
    #[serde(default)]
    pub schemas: UninitializedSchemas,
    pub output_schema: Option<ResolvedTomlPathData>,
    #[serde(default)]
    pub description: Option<String>,
    pub experimentation: Option<UninitializedExperimentationConfigWithNamespaces>,
}

impl From<UninitializedFunctionConfig> for StoredFunctionConfig {
    fn from(config: UninitializedFunctionConfig) -> Self {
        match config {
            UninitializedFunctionConfig::Chat(chat) => Self::Chat(chat.into()),
            UninitializedFunctionConfig::Json(json) => Self::Json(json.into()),
        }
    }
}

impl From<UninitializedFunctionConfigChat> for StoredFunctionConfigChat {
    fn from(config: UninitializedFunctionConfigChat) -> Self {
        let UninitializedFunctionConfigChat {
            variants,
            system_schema,
            user_schema,
            assistant_schema,
            schemas,
            tools,
            tool_choice,
            parallel_tool_calls,
            description,
            experimentation,
        } = config;

        Self {
            variants: variants.into_iter().map(|(k, v)| (k, v.into())).collect(),
            system_schema,
            user_schema,
            assistant_schema,
            schemas,
            tools,
            tool_choice,
            parallel_tool_calls,
            description,
            experimentation,
        }
    }
}

impl From<UninitializedFunctionConfigJson> for StoredFunctionConfigJson {
    fn from(config: UninitializedFunctionConfigJson) -> Self {
        let UninitializedFunctionConfigJson {
            variants,
            system_schema,
            user_schema,
            assistant_schema,
            schemas,
            output_schema,
            description,
            experimentation,
        } = config;

        Self {
            variants: variants.into_iter().map(|(k, v)| (k, v.into())).collect(),
            system_schema,
            user_schema,
            assistant_schema,
            schemas,
            output_schema,
            description,
            experimentation,
        }
    }
}

impl TryFrom<StoredFunctionConfig> for UninitializedFunctionConfig {
    type Error = &'static str;

    fn try_from(stored: StoredFunctionConfig) -> Result<Self, Self::Error> {
        match stored {
            StoredFunctionConfig::Chat(chat) => Ok(Self::Chat(chat.try_into()?)),
            StoredFunctionConfig::Json(json) => Ok(Self::Json(json.try_into()?)),
        }
    }
}

impl TryFrom<StoredFunctionConfigChat> for UninitializedFunctionConfigChat {
    type Error = &'static str;

    fn try_from(stored: StoredFunctionConfigChat) -> Result<Self, Self::Error> {
        let StoredFunctionConfigChat {
            variants,
            system_schema,
            user_schema,
            assistant_schema,
            schemas,
            tools,
            tool_choice,
            parallel_tool_calls,
            description,
            experimentation,
        } = stored;

        let uninit_variants = migrate_stored_variants(variants)?;

        Ok(Self {
            variants: uninit_variants,
            system_schema,
            user_schema,
            assistant_schema,
            schemas,
            tools,
            tool_choice,
            parallel_tool_calls,
            description,
            experimentation,
        })
    }
}

impl TryFrom<StoredFunctionConfigJson> for UninitializedFunctionConfigJson {
    type Error = &'static str;

    fn try_from(stored: StoredFunctionConfigJson) -> Result<Self, Self::Error> {
        let StoredFunctionConfigJson {
            variants,
            system_schema,
            user_schema,
            assistant_schema,
            schemas,
            output_schema,
            description,
            experimentation,
        } = stored;

        let uninit_variants = migrate_stored_variants(variants)?;

        Ok(Self {
            variants: uninit_variants,
            system_schema,
            user_schema,
            assistant_schema,
            schemas,
            output_schema,
            description,
            experimentation,
        })
    }
}

/// Converts stored variants to uninitialized variants, migrating the deprecated
/// `timeout_s` from BestOfN/MixtureOfN variants to `timeouts` on their
/// candidate variants.
fn migrate_stored_variants(
    stored_variants: HashMap<String, StoredVariantInfo>,
) -> Result<HashMap<String, UninitializedVariantInfo>, &'static str> {
    // First, collect timeout_s propagation info from stored variants
    let mut timeout_propagations: Vec<(f64, Vec<String>)> = Vec::new();

    for variant_info in stored_variants.values() {
        match &variant_info.inner {
            StoredVariantConfig::BestOfNSampling(config) => {
                if let Some(timeout_s) = config.timeout_s {
                    timeout_propagations.push((timeout_s, config.candidates.clone()));
                }
            }
            StoredVariantConfig::MixtureOfN(config) => {
                if let Some(timeout_s) = config.timeout_s {
                    timeout_propagations.push((timeout_s, config.candidates.clone()));
                }
            }
            _ => {}
        }
    }

    // Convert all stored variants to uninitialized
    let mut uninit_variants: HashMap<String, UninitializedVariantInfo> = stored_variants
        .into_iter()
        .map(|(k, v)| (k, v.into()))
        .collect();

    // Apply timeout_s to candidate variants
    for (timeout_s, candidates) in timeout_propagations {
        let timeout_ms = (timeout_s * 1000.0) as u64;
        let timeouts_config = TimeoutsConfig {
            non_streaming: NonStreamingTimeouts {
                total_ms: Some(timeout_ms),
            },
            streaming: StreamingTimeouts {
                ttft_ms: Some(timeout_ms),
                total_ms: None,
            },
        };

        for candidate_name in candidates {
            let candidate_variant = uninit_variants
                .get_mut(&candidate_name)
                .ok_or("stored config references a missing candidate variant")?;

            // Only set timeouts if not already configured
            if candidate_variant.timeouts.is_none() {
                candidate_variant.timeouts = Some(timeouts_config.clone());
            }
        }
    }

    Ok(uninit_variants)
}
