use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::TimeoutsConfig;
use crate::embeddings::{UninitializedEmbeddingModelConfig, UninitializedEmbeddingProviderConfig};
use crate::inference::types::extra_body::ExtraBodyConfig;
use crate::inference::types::extra_headers::ExtraHeadersConfig;
use crate::model::UninitializedProviderConfig;
use tensorzero_types::UninitializedUnifiedCostConfig;

/// Stored version of `UninitializedEmbeddingModelConfig`.
///
/// Accepts the deprecated `timeouts` field for backward compatibility with
/// historical config snapshots stored in ClickHouse.
#[derive(Clone, Debug, Deserialize, Serialize)]
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

        Self {
            routing,
            providers: providers.into_iter().map(|(k, v)| (k, v.into())).collect(),
            // Migration: prefer new field, fall back to deprecated
            timeout_ms: timeout_ms.or_else(|| timeouts.non_streaming.and_then(|ns| ns.total_ms)),
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

/// Stored version of `UninitializedEmbeddingProviderConfig`.
///
/// Accepts the deprecated `timeouts` field for backward compatibility.
#[derive(ts_rs::TS, Clone, Debug, Deserialize, Serialize)]
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
    #[serde(default)]
    #[ts(optional)]
    pub extra_headers: Option<ExtraHeadersConfig>,
    #[serde(default)]
    #[ts(skip)]
    pub cost: Option<UninitializedUnifiedCostConfig>,
}

impl From<StoredEmbeddingProviderConfig> for UninitializedEmbeddingProviderConfig {
    fn from(stored: StoredEmbeddingProviderConfig) -> Self {
        // Explicit destructuring ensures compile error if fields are added/removed
        let StoredEmbeddingProviderConfig {
            config,
            timeout_ms,
            timeouts,
            extra_body,
            extra_headers,
            cost,
        } = stored;

        Self {
            config,
            // Migration: prefer new field, fall back to deprecated
            timeout_ms: timeout_ms.or_else(|| timeouts.non_streaming.and_then(|ns| ns.total_ms)),
            extra_body,
            extra_headers,
            cost,
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
            extra_headers,
            cost,
        } = uninitialized;

        Self {
            config,
            timeout_ms,
            timeouts: TimeoutsConfig::default(),
            extra_body,
            extra_headers,
            cost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Old snapshot with deprecated `timeouts` on embedding model should parse
    /// and migrate the value to `timeout_ms`.
    #[test]
    fn test_deprecated_timeouts_migrates() {
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
        assert_eq!(
            stored
                .timeouts
                .non_streaming
                .as_ref()
                .and_then(|ns| ns.total_ms),
            Some(5000)
        );

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
    fn test_provider_deprecated_timeouts_migrates() {
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
        assert_eq!(
            provider
                .timeouts
                .non_streaming
                .as_ref()
                .and_then(|ns| ns.total_ms),
            Some(7000)
        );

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
    fn test_new_timeout_ms() {
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
    fn test_new_field_takes_precedence() {
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
}
