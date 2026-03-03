use serde::{Deserialize, Serialize};

use crate::config::{TimeoutsConfig, UninitializedVariantConfig, UninitializedVariantInfo};
use crate::variant::best_of_n_sampling::{
    UninitializedBestOfNEvaluatorConfig, UninitializedBestOfNSamplingConfig,
};
use crate::variant::chain_of_thought::UninitializedChainOfThoughtConfig;
use crate::variant::chat_completion::UninitializedChatCompletionConfig;
use crate::variant::dicl::UninitializedDiclConfig;
use crate::variant::mixture_of_n::{UninitializedFuserConfig, UninitializedMixtureOfNConfig};

/// Stored version of `UninitializedBestOfNSamplingConfig`.
///
/// Retains the deprecated `timeout_s` field so that historical config snapshots
/// stored in ClickHouse can still be deserialized.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredBestOfNSamplingConfig {
    #[serde(default)]
    pub weight: Option<f64>,
    /// DEPRECATED: Use `[timeouts]` on candidate variants instead.
    #[serde(default)]
    pub timeout_s: Option<f64>,
    pub candidates: Vec<String>,
    pub evaluator: UninitializedBestOfNEvaluatorConfig,
}

impl From<UninitializedBestOfNSamplingConfig> for StoredBestOfNSamplingConfig {
    fn from(config: UninitializedBestOfNSamplingConfig) -> Self {
        let UninitializedBestOfNSamplingConfig {
            weight,
            candidates,
            evaluator,
        } = config;

        Self {
            weight,
            timeout_s: None,
            candidates,
            evaluator,
        }
    }
}

impl From<StoredBestOfNSamplingConfig> for UninitializedBestOfNSamplingConfig {
    fn from(stored: StoredBestOfNSamplingConfig) -> Self {
        let StoredBestOfNSamplingConfig {
            weight,
            timeout_s: _, // dropped; migrated at function-config level
            candidates,
            evaluator,
        } = stored;

        Self {
            weight,
            candidates,
            evaluator,
        }
    }
}

/// Stored version of `UninitializedMixtureOfNConfig`.
///
/// Retains the deprecated `timeout_s` field for backward compatibility.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredMixtureOfNConfig {
    #[serde(default)]
    pub weight: Option<f64>,
    /// DEPRECATED: Use `[timeouts]` on candidate variants instead.
    #[serde(default)]
    pub timeout_s: Option<f64>,
    pub candidates: Vec<String>,
    pub fuser: UninitializedFuserConfig,
}

impl From<UninitializedMixtureOfNConfig> for StoredMixtureOfNConfig {
    fn from(config: UninitializedMixtureOfNConfig) -> Self {
        let UninitializedMixtureOfNConfig {
            weight,
            candidates,
            fuser,
        } = config;

        Self {
            weight,
            timeout_s: None,
            candidates,
            fuser,
        }
    }
}

impl From<StoredMixtureOfNConfig> for UninitializedMixtureOfNConfig {
    fn from(stored: StoredMixtureOfNConfig) -> Self {
        let StoredMixtureOfNConfig {
            weight,
            timeout_s: _, // dropped; migrated at function-config level
            candidates,
            fuser,
        } = stored;

        Self {
            weight,
            candidates,
            fuser,
        }
    }
}

/// Stored version of `UninitializedVariantConfig`.
///
/// Uses `StoredBestOfNSamplingConfig` and `StoredMixtureOfNConfig` for variants
/// that had the deprecated `timeout_s` field. Other variants use the
/// `Uninitialized*` types directly.
///
/// Does NOT use `deny_unknown_fields` for forward-compatibility.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum StoredVariantConfig {
    ChatCompletion(UninitializedChatCompletionConfig),
    #[serde(rename = "experimental_best_of_n_sampling")]
    BestOfNSampling(StoredBestOfNSamplingConfig),
    #[serde(rename = "experimental_dynamic_in_context_learning")]
    Dicl(UninitializedDiclConfig),
    #[serde(rename = "experimental_mixture_of_n")]
    MixtureOfN(StoredMixtureOfNConfig),
    #[serde(rename = "experimental_chain_of_thought")]
    ChainOfThought(UninitializedChainOfThoughtConfig),
}

impl From<UninitializedVariantConfig> for StoredVariantConfig {
    fn from(config: UninitializedVariantConfig) -> Self {
        match config {
            UninitializedVariantConfig::ChatCompletion(c) => Self::ChatCompletion(c),
            UninitializedVariantConfig::BestOfNSampling(c) => Self::BestOfNSampling(c.into()),
            UninitializedVariantConfig::Dicl(c) => Self::Dicl(c),
            UninitializedVariantConfig::MixtureOfN(c) => Self::MixtureOfN(c.into()),
            UninitializedVariantConfig::ChainOfThought(c) => Self::ChainOfThought(c),
        }
    }
}

impl From<StoredVariantConfig> for UninitializedVariantConfig {
    fn from(stored: StoredVariantConfig) -> Self {
        match stored {
            StoredVariantConfig::ChatCompletion(c) => Self::ChatCompletion(c),
            StoredVariantConfig::BestOfNSampling(c) => Self::BestOfNSampling(c.into()),
            StoredVariantConfig::Dicl(c) => Self::Dicl(c),
            StoredVariantConfig::MixtureOfN(c) => Self::MixtureOfN(c.into()),
            StoredVariantConfig::ChainOfThought(c) => Self::ChainOfThought(c),
        }
    }
}

/// Stored version of `UninitializedVariantInfo`.
///
/// Wraps `StoredVariantConfig` plus the `timeouts` field.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredVariantInfo {
    #[serde(flatten)]
    pub inner: StoredVariantConfig,
    #[serde(default)]
    pub timeouts: Option<TimeoutsConfig>,
}

impl From<UninitializedVariantInfo> for StoredVariantInfo {
    fn from(info: UninitializedVariantInfo) -> Self {
        let UninitializedVariantInfo { inner, timeouts } = info;

        Self {
            inner: inner.into(),
            timeouts,
        }
    }
}

impl From<StoredVariantInfo> for UninitializedVariantInfo {
    fn from(stored: StoredVariantInfo) -> Self {
        let StoredVariantInfo { inner, timeouts } = stored;

        Self {
            inner: inner.into(),
            timeouts,
        }
    }
}
