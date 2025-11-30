//! Stored config types for backward-compatible deserialization of historical snapshots.
//!
//! When deprecating a config field:
//! 1. Remove it from the `Uninitialized*` type (fresh configs will reject it)
//! 2. Keep it in the `Stored*` type (snapshots can still load)
//! 3. Implement migration in `From<Stored*> for Uninitialized*`
//!
//! The `From` implementations use explicit destructuring to ensure compile-time errors
//! when fields are added or removed from either the stored or uninitialized types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::gateway::UninitializedGatewayConfig;
use crate::config::provider_types::ProviderTypesConfig;
use crate::config::{
    MetricConfig, PostgresConfig, TimeoutsConfig, UninitializedConfig, UninitializedFunctionConfig,
    UninitializedToolConfig,
};
use crate::embeddings::{UninitializedEmbeddingModelConfig, UninitializedEmbeddingProviderConfig};
use crate::evaluations::UninitializedEvaluationConfig;
use crate::inference::types::extra_body::ExtraBodyConfig;
use crate::inference::types::storage::StorageKind;
use crate::model::UninitializedModelConfig;
use crate::model::UninitializedProviderConfig;
use crate::optimization::UninitializedOptimizerInfo;
use crate::rate_limiting::UninitializedRateLimitingConfig;

/// Stored version of `UninitializedEmbeddingModelConfig`.
///
/// Accepts the deprecated `timeouts` field for backward compatibility with
/// historical config snapshots stored in ClickHouse.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StoredEmbeddingModelConfig {
    pub routing: Vec<Arc<str>>,
    pub providers: HashMap<Arc<str>, StoredEmbeddingProviderConfig>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// DEPRECATED: Use `timeout_ms` instead.
    /// Kept for backward compatibility with stored snapshots.
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
}

impl From<StoredEmbeddingModelConfig> for UninitializedEmbeddingModelConfig {
    fn from(stored: StoredEmbeddingModelConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let StoredEmbeddingModelConfig {
            routing,
            providers,
            timeout_ms,
            timeouts,
        } = stored;

        // Emit deprecation warning if old format detected
        if timeouts.non_streaming.total_ms.is_some() {
            crate::utils::deprecation_warning(
                "`timeouts` is deprecated for embedding models. Use `timeout_ms` instead.",
            );
        }

        Self {
            routing,
            providers: providers.into_iter().map(|(k, v)| (k, v.into())).collect(),
            // Migration: prefer new field, fall back to deprecated
            timeout_ms: timeout_ms.or(timeouts.non_streaming.total_ms),
        }
    }
}

/// Stored version of `UninitializedEmbeddingProviderConfig`.
///
/// Accepts the deprecated `timeouts` field for backward compatibility.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredEmbeddingProviderConfig {
    #[serde(flatten)]
    pub config: UninitializedProviderConfig,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// DEPRECATED: Use `timeout_ms` instead.
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
    #[serde(default)]
    pub extra_body: Option<ExtraBodyConfig>,
}

impl From<StoredEmbeddingProviderConfig> for UninitializedEmbeddingProviderConfig {
    fn from(stored: StoredEmbeddingProviderConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let StoredEmbeddingProviderConfig {
            config,
            timeout_ms,
            timeouts,
            extra_body,
        } = stored;

        // Emit deprecation warning if old format detected
        if timeouts.non_streaming.total_ms.is_some() {
            crate::utils::deprecation_warning(
                "`timeouts` is deprecated for embedding providers. Use `timeout_ms` instead.",
            );
        }

        Self {
            config,
            // Migration: prefer new field, fall back to deprecated
            timeout_ms: timeout_ms.or(timeouts.non_streaming.total_ms),
            extra_body,
        }
    }
}

impl From<UninitializedEmbeddingProviderConfig> for StoredEmbeddingProviderConfig {
    fn from(uninitialized: UninitializedEmbeddingProviderConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let UninitializedEmbeddingProviderConfig {
            config,
            timeout_ms,
            extra_body,
        } = uninitialized;

        Self {
            config,
            timeout_ms,
            timeouts: TimeoutsConfig::default(),
            extra_body,
        }
    }
}

impl From<UninitializedEmbeddingModelConfig> for StoredEmbeddingModelConfig {
    fn from(uninitialized: UninitializedEmbeddingModelConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let UninitializedEmbeddingModelConfig {
            routing,
            providers,
            timeout_ms,
        } = uninitialized;

        Self {
            routing,
            providers: providers.into_iter().map(|(k, v)| (k, v.into())).collect(),
            timeout_ms,
            timeouts: TimeoutsConfig::default(),
        }
    }
}

/// Top-level stored config type.
///
/// Only fields with deprecations in their subtree use `Stored*` types.
/// Other fields re-use `Uninitialized*` types directly.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StoredConfig {
    // Fields WITHOUT deprecations - reuse Uninitialized* types
    #[serde(default)]
    pub gateway: UninitializedGatewayConfig,
    #[serde(default)]
    pub postgres: PostgresConfig,
    #[serde(default)]
    pub rate_limiting: UninitializedRateLimitingConfig,
    pub object_storage: Option<StorageKind>,
    #[serde(default)]
    pub models: HashMap<Arc<str>, UninitializedModelConfig>,
    #[serde(default)]
    pub functions: HashMap<String, UninitializedFunctionConfig>,
    #[serde(default)]
    pub metrics: HashMap<String, MetricConfig>,
    #[serde(default)]
    pub tools: HashMap<String, UninitializedToolConfig>,
    #[serde(default)]
    pub evaluations: HashMap<String, UninitializedEvaluationConfig>,
    #[serde(default)]
    pub provider_types: ProviderTypesConfig,
    #[serde(default)]
    pub optimizers: HashMap<String, UninitializedOptimizerInfo>,

    // Fields WITH deprecations - use Stored* types
    #[serde(default)]
    pub embedding_models: HashMap<Arc<str>, StoredEmbeddingModelConfig>,
}

impl From<UninitializedConfig> for StoredConfig {
    fn from(stored: UninitializedConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let UninitializedConfig {
            gateway,
            postgres,
            rate_limiting,
            object_storage,
            models,
            functions,
            metrics,
            tools,
            evaluations,
            provider_types,
            optimizers,
            embedding_models,
        } = stored;

        // Note: as we migrate the config and deprecate stuff in the future,
        // we'll need to build out this transformation
        Self {
            gateway,
            postgres,
            rate_limiting,
            object_storage,
            models,
            functions,
            metrics,
            tools,
            evaluations,
            provider_types,
            optimizers,
            // Only embedding_models needs conversion
            embedding_models: embedding_models
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        }
    }
}

impl From<StoredConfig> for UninitializedConfig {
    fn from(stored: StoredConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let StoredConfig {
            gateway,
            postgres,
            rate_limiting,
            object_storage,
            models,
            functions,
            metrics,
            tools,
            evaluations,
            provider_types,
            optimizers,
            embedding_models,
        } = stored;

        Self {
            gateway,
            postgres,
            rate_limiting,
            object_storage,
            models,
            functions,
            metrics,
            tools,
            evaluations,
            provider_types,
            optimizers,
            // Only embedding_models needs conversion
            embedding_models: embedding_models
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        }
    }
}
