use serde::{Deserialize, Serialize};

use crate::config::{BatchWritesConfig, ObservabilityBackend, ObservabilityConfig};

/// Stored version of `ObservabilityConfig`.
///
/// Omits `deny_unknown_fields` so that future fields added in
/// newer versions don't break deserialization in rolled-back gateways.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StoredObservabilityConfig {
    pub enabled: Option<bool>,
    #[serde(default)]
    pub backend: ObservabilityBackend,
    #[serde(default)]
    pub async_writes: bool,
    #[serde(default)]
    pub batch_writes: BatchWritesConfig,

    /// Deprecated since 2026.2
    #[serde(default)]
    pub disable_automatic_migrations: bool,
}

impl From<ObservabilityConfig> for StoredObservabilityConfig {
    fn from(config: ObservabilityConfig) -> Self {
        let ObservabilityConfig {
            enabled,
            backend,
            async_writes,
            batch_writes,
            #[expect(deprecated)]
            disable_automatic_migrations,
        } = config;
        Self {
            enabled,
            backend,
            async_writes,
            batch_writes,
            disable_automatic_migrations,
        }
    }
}

impl From<StoredObservabilityConfig> for ObservabilityConfig {
    fn from(stored: StoredObservabilityConfig) -> Self {
        let StoredObservabilityConfig {
            enabled,
            backend,
            async_writes,
            batch_writes,
            disable_automatic_migrations,
        } = stored;
        Self {
            enabled,
            backend,
            async_writes,
            batch_writes,
            #[expect(deprecated)]
            disable_automatic_migrations,
        }
    }
}
