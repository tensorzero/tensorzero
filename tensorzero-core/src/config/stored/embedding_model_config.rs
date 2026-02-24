use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::TimeoutsConfig;
use crate::embeddings::{UninitializedEmbeddingModelConfig, UninitializedEmbeddingProviderConfig};
use crate::inference::types::extra_body::ExtraBodyConfig;
use crate::inference::types::extra_headers::ExtraHeadersConfig;
use crate::model::UninitializedProviderConfig;
use tensorzero_types::UninitializedCostConfig;

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

        Self {
            routing,
            providers: providers.into_iter().map(|(k, v)| (k, v.into())).collect(),
            // Migration: prefer new field, fall back to deprecated
            timeout_ms: timeout_ms.or(timeouts.non_streaming.total_ms),
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
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
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
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub extra_headers: Option<ExtraHeadersConfig>,
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(skip))]
    pub cost: Option<UninitializedCostConfig>,
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
            timeout_ms: timeout_ms.or(timeouts.non_streaming.total_ms),
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
