//! Stored config types for backward-compatible deserialization of historical snapshots.
//!
//! When deprecating a config field:
//! 1. Remove it from the `Uninitialized*` type (fresh configs will reject it)
//! 2. Keep it in the `Stored*` type (snapshots can still load)
//! 3. Implement migration in `From<Stored*> for Uninitialized*`
//!
//! The `From` implementations use explicit destructuring to ensure compile-time errors
//! when fields are added or removed from either the stored or uninitialized types.

mod cache_config;
mod embedding_model_config;
mod gateway_config;
mod observability_config;

pub use cache_config::StoredCacheConfig;
pub use embedding_model_config::{StoredEmbeddingModelConfig, StoredEmbeddingProviderConfig};
pub use gateway_config::StoredGatewayConfig;
pub use observability_config::StoredObservabilityConfig;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::gateway::UninitializedGatewayConfig;
use crate::config::provider_types::ProviderTypesConfig;
use crate::config::{
    ClickHouseConfig, MetricConfig, PostgresConfig, UninitializedConfig,
    UninitializedFunctionConfig, UninitializedToolConfig,
};
use crate::evaluations::UninitializedEvaluationConfig;
use crate::inference::types::storage::StorageKind;
use crate::model::UninitializedModelConfig;
use crate::optimization::UninitializedOptimizerInfo;
use crate::rate_limiting::UninitializedRateLimitingConfig;

/// Top-level stored config type.
///
/// Omits `deny_unknown_fields` for forward-compatibility: snapshots written by
/// newer gateways (with additional fields) can still be deserialized by older
/// gateways that don't know about those fields.
///
/// Fields whose subtree may change use `Stored*` wrapper types.
/// Other fields re-use `Uninitialized*` types directly.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StoredConfig {
    #[serde(default)]
    pub gateway: StoredGatewayConfig,
    #[serde(default)]
    pub clickhouse: ClickHouseConfig,
    #[serde(default)]
    pub postgres: PostgresConfig,
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

    // Fields WITH deprecations or custom serde - use Stored* types
    #[serde(default)]
    pub rate_limiting: UninitializedRateLimitingConfig,
    #[serde(default)]
    pub embedding_models: HashMap<Arc<str>, StoredEmbeddingModelConfig>,
}

impl From<UninitializedConfig> for StoredConfig {
    fn from(config: UninitializedConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let UninitializedConfig {
            gateway,
            clickhouse,
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
        } = config;

        Self {
            gateway: gateway.into(),
            clickhouse,
            postgres,
            object_storage,
            models,
            functions,
            metrics,
            tools,
            evaluations,
            provider_types,
            optimizers,
            rate_limiting,
            embedding_models: embedding_models
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        }
    }
}

impl TryFrom<StoredConfig> for UninitializedConfig {
    type Error = &'static str;

    fn try_from(stored: StoredConfig) -> Result<Self, Self::Error> {
        // Explicit destructuring ensures compile error if fields are added/removed
        let StoredConfig {
            gateway,
            clickhouse,
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

        // Migrate deprecated `gateway.observability.disable_automatic_migrations`
        // to `clickhouse.disable_automatic_migrations`.
        let clickhouse = ClickHouseConfig {
            disable_automatic_migrations: clickhouse.disable_automatic_migrations
                || gateway.observability.disable_automatic_migrations,
        };

        let gateway_config: UninitializedGatewayConfig = gateway.into();

        Ok(Self {
            gateway: gateway_config,
            clickhouse,
            postgres,
            object_storage,
            models,
            functions,
            metrics,
            tools,
            evaluations,
            provider_types,
            optimizers,
            rate_limiting,
            embedding_models: embedding_models
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::UninitializedEmbeddingModelConfig;

    /// Old snapshot with deprecated `gateway.observability.disable_automatic_migrations`
    /// should silently migrate to `clickhouse.disable_automatic_migrations`.
    #[test]
    fn test_stored_config_with_deprecated_disable_automatic_migrations() {
        let toml_str = r"
            [gateway.observability]
            disable_automatic_migrations = true
        ";

        let stored: StoredConfig =
            toml::from_str(toml_str).expect("should parse deprecated field location");
        assert!(
            stored.gateway.observability.disable_automatic_migrations,
            "deprecated field should be parsed"
        );

        let uninit: UninitializedConfig = stored.try_into().expect("should convert to uninit");
        assert!(
            uninit.clickhouse.disable_automatic_migrations,
            "deprecated field should be migrated to clickhouse config"
        );
    }

    /// New snapshot with `[clickhouse] disable_automatic_migrations` should parse correctly.
    #[test]
    fn test_stored_config_with_new_clickhouse_section() {
        let toml_str = r"
            [clickhouse]
            disable_automatic_migrations = true
        ";

        let stored: StoredConfig =
            toml::from_str(toml_str).expect("should parse new clickhouse section");
        assert!(stored.clickhouse.disable_automatic_migrations);

        let uninit: UninitializedConfig = stored.try_into().expect("should convert to uninit");
        assert!(uninit.clickhouse.disable_automatic_migrations);
    }

    /// Old snapshot with deprecated `timeouts` on embedding model should parse
    /// and migrate the value to `timeout_ms`.
    #[test]
    fn test_stored_embedding_model_deprecated_timeouts_migrates() {
        let toml_str = r#"
            routing = ["provider1"]

            [timeouts.non_streaming]
            total_ms = 5000

            [providers.provider1]
            model_name = "text-embedding-ada-002"
            type = "openai"
        "#;

        let stored: StoredEmbeddingModelConfig =
            toml::from_str(toml_str).expect("should parse deprecated timeouts field");
        assert!(
            stored.timeout_ms.is_none(),
            "new field should not be set when only deprecated field is present"
        );
        assert_eq!(stored.timeouts.non_streaming.total_ms, Some(5000));

        let uninit: UninitializedEmbeddingModelConfig = stored.into();
        assert_eq!(
            uninit.timeout_ms,
            Some(5000),
            "deprecated timeouts.non_streaming.total_ms should migrate to timeout_ms"
        );
    }

    /// Old snapshot with deprecated `timeouts` on embedding provider should parse
    /// and migrate the value to the provider's `timeout_ms`.
    #[test]
    fn test_stored_embedding_provider_deprecated_timeouts_migrates() {
        let toml_str = r#"
            routing = ["provider1"]

            [providers.provider1]
            model_name = "text-embedding-ada-002"
            type = "openai"

            [providers.provider1.timeouts.non_streaming]
            total_ms = 7000
        "#;

        let stored: StoredEmbeddingModelConfig =
            toml::from_str(toml_str).expect("should parse deprecated provider timeouts");
        let provider = stored.providers.get("provider1").unwrap();
        assert_eq!(provider.timeouts.non_streaming.total_ms, Some(7000));

        let uninit: UninitializedEmbeddingModelConfig = stored.into();
        let provider = uninit.providers.get("provider1").unwrap();
        assert_eq!(
            provider.timeout_ms,
            Some(7000),
            "deprecated provider timeouts should migrate to provider timeout_ms"
        );
    }

    /// New snapshot with `timeout_ms` on embedding model should parse correctly.
    #[test]
    fn test_stored_embedding_model_new_timeout_ms() {
        let toml_str = r#"
            routing = ["provider1"]
            timeout_ms = 3000

            [providers.provider1]
            model_name = "text-embedding-ada-002"
            type = "openai"
            timeout_ms = 2000
        "#;

        let stored: StoredEmbeddingModelConfig =
            toml::from_str(toml_str).expect("should parse new timeout_ms field");
        assert_eq!(stored.timeout_ms, Some(3000));

        let uninit: UninitializedEmbeddingModelConfig = stored.into();
        assert_eq!(uninit.timeout_ms, Some(3000));
        let (_, provider) = uninit.providers.into_iter().next().unwrap();
        assert_eq!(provider.timeout_ms, Some(2000));
    }

    /// When both `timeout_ms` and deprecated `timeouts` are set, `timeout_ms` wins.
    #[test]
    fn test_stored_embedding_model_new_field_takes_precedence() {
        let toml_str = r#"
            routing = ["provider1"]
            timeout_ms = 1000

            [timeouts.non_streaming]
            total_ms = 9999

            [providers.provider1]
            model_name = "text-embedding-ada-002"
            type = "openai"
        "#;

        let stored: StoredEmbeddingModelConfig =
            toml::from_str(toml_str).expect("should parse both fields");
        let uninit: UninitializedEmbeddingModelConfig = stored.into();
        assert_eq!(
            uninit.timeout_ms,
            Some(1000),
            "new timeout_ms should take precedence over deprecated timeouts"
        );
    }

    /// Forward-compatibility: `StoredObservabilityConfig` should accept unknown fields
    /// (e.g. a future `backend` field added by a newer gateway version).
    #[test]
    fn test_stored_observability_config_ignores_unknown_fields() {
        let toml_str = r#"
            enabled = true
            async_writes = true
            backend = "postgres"
            some_future_field = 42
        "#;

        let stored: StoredObservabilityConfig =
            toml::from_str(toml_str).expect("should ignore unknown fields");
        assert_eq!(stored.enabled, Some(true));
        assert!(stored.async_writes);
    }

    /// Forward-compatibility: `StoredCacheConfig` should accept unknown fields
    /// (e.g. future `enabled` and `backend` fields).
    #[test]
    fn test_stored_cache_config_ignores_unknown_fields() {
        let toml_str = r#"
            enabled = true
            backend = "valkey"

            [valkey]
            ttl_s = 3600
        "#;

        let stored: StoredCacheConfig =
            toml::from_str(toml_str).expect("should ignore unknown fields");
        assert_eq!(stored.valkey.ttl_s, 3600);
    }

    /// Forward-compatibility: `StoredGatewayConfig` should accept unknown fields.
    #[test]
    fn test_stored_gateway_config_ignores_unknown_fields() {
        let toml_str = r#"
            debug = true
            some_new_feature = "value"
        "#;

        let stored: StoredGatewayConfig =
            toml::from_str(toml_str).expect("should ignore unknown fields");
        assert!(stored.debug);
    }

    /// Forward-compatibility: top-level `StoredConfig` should accept unknown fields.
    #[test]
    fn test_stored_config_ignores_unknown_top_level_fields() {
        let toml_str = r#"
            [gateway]
            debug = true

            [some_future_section]
            key = "value"
        "#;

        let stored: StoredConfig =
            toml::from_str(toml_str).expect("should ignore unknown top-level sections");
        assert!(stored.gateway.debug);
    }

    /// Roundtrip: StoredConfig → TOML → StoredConfig → UninitializedConfig
    /// should preserve all meaningful fields.
    #[test]
    fn test_roundtrip_stored_to_toml_and_back() {
        let toml_str = r"
            [gateway]
            debug = true

            [gateway.observability]
            enabled = true
            async_writes = true

            [clickhouse]
            disable_automatic_migrations = true
        ";

        let stored: StoredConfig = toml::from_str(toml_str).expect("should parse");
        let serialized = toml::to_string(&stored).expect("should serialize to TOML");
        let stored2: StoredConfig =
            toml::from_str(&serialized).expect("should deserialize back from TOML");
        let uninit: UninitializedConfig = stored2.try_into().expect("should convert to uninit");

        assert!(uninit.clickhouse.disable_automatic_migrations);
        assert_eq!(uninit.gateway.observability.enabled, Some(true));
        assert!(uninit.gateway.observability.async_writes);
        assert!(uninit.gateway.debug);
    }
}
