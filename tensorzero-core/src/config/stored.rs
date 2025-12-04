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
    MetricConfig, PostgresConfig, UninitializedConfig, UninitializedFunctionConfig,
    UninitializedToolConfig,
};
use crate::embeddings::UninitializedEmbeddingModelConfig;
use crate::evaluations::UninitializedEvaluationConfig;
use crate::inference::types::storage::StorageKind;
use crate::model::UninitializedModelConfig;
use crate::optimization::UninitializedOptimizerInfo;
use crate::rate_limiting::UninitializedRateLimitingConfig;

/// Top-level stored config type.
///
/// Only fields with deprecations in their subtree use `Stored*` types.
/// Other fields re-use `Uninitialized*` types directly.
#[derive(Debug, Default, Deserialize, Serialize)]
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
    pub embedding_models: HashMap<Arc<str>, UninitializedEmbeddingModelConfig>,
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
            embedding_models,
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

        // Note: as we migrate the config and deprecate stuff in the future,
        // we'll need to build out this transformation
        Self {
            gateway,
            postgres,
            rate_limiting,
            object_storage,
            models,
            embedding_models,
            functions,
            metrics,
            tools,
            evaluations,
            provider_types,
            optimizers,
        }
    }
}
