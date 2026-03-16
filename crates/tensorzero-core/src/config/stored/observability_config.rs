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
    pub batch_writes: StoredBatchWritesConfig,

    /// Deprecated since 2026.2
    #[serde(default)]
    pub disable_automatic_migrations: bool,
}

/// Stored version of `BatchWritesConfig`.
///
/// Omits `deny_unknown_fields` so that snapshots written by newer gateways
/// (which may include additional fields) can still be deserialized by older code.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StoredBatchWritesConfig {
    pub enabled: bool,
    #[serde(default)]
    pub __force_allow_embedded_batch_writes: bool,
    #[serde(default = "crate::config::default_flush_interval_ms")]
    pub flush_interval_ms: u64,
    #[serde(default = "crate::config::default_max_rows")]
    pub max_rows: usize,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_rows_postgres: Option<usize>,
    #[serde(default = "crate::config::default_write_queue_capacity")]
    pub write_queue_capacity: usize,
}

impl Default for StoredBatchWritesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            __force_allow_embedded_batch_writes: false,
            flush_interval_ms: crate::config::default_flush_interval_ms(),
            max_rows: crate::config::default_max_rows(),
            max_rows_postgres: None,
            write_queue_capacity: crate::config::default_write_queue_capacity(),
        }
    }
}

impl From<BatchWritesConfig> for StoredBatchWritesConfig {
    fn from(config: BatchWritesConfig) -> Self {
        Self {
            enabled: config.enabled,
            __force_allow_embedded_batch_writes: config.__force_allow_embedded_batch_writes,
            flush_interval_ms: config.flush_interval_ms,
            max_rows: config.max_rows,
            max_rows_postgres: config.max_rows_postgres,
            write_queue_capacity: config.write_queue_capacity,
        }
    }
}

impl From<StoredBatchWritesConfig> for BatchWritesConfig {
    fn from(stored: StoredBatchWritesConfig) -> Self {
        Self {
            enabled: stored.enabled,
            __force_allow_embedded_batch_writes: stored.__force_allow_embedded_batch_writes,
            flush_interval_ms: stored.flush_interval_ms,
            max_rows: stored.max_rows,
            max_rows_postgres: stored.max_rows_postgres,
            write_queue_capacity: stored.write_queue_capacity,
        }
    }
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
            batch_writes: batch_writes.into(),
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
            batch_writes: batch_writes.into(),
            #[expect(deprecated)]
            disable_automatic_migrations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Historical: `disable_automatic_migrations` was an observability field before
    /// being migrated to a top-level `[clickhouse]` section. Stored snapshots from
    /// that era must still parse.
    #[test]
    fn test_disable_automatic_migrations_parses() {
        let toml_str = r"
            enabled = true
            async_writes = true
            disable_automatic_migrations = true
        ";

        let stored: StoredObservabilityConfig =
            toml::from_str(toml_str).expect("should parse deprecated field");
        assert!(
            stored.disable_automatic_migrations,
            "deprecated field should be preserved"
        );
    }

    /// Historical: before `write_queue_capacity` was added to `BatchWritesConfig`,
    /// stored configs didn't include this field. They should still parse with the default value.
    #[test]
    fn test_historical_no_write_queue_capacity() {
        let toml_str = r"
            enabled = true
            async_writes = true

            [batch_writes]
            enabled = true
            flush_interval_ms = 100
            max_rows = 500
        ";

        let stored: StoredObservabilityConfig =
            toml::from_str(toml_str).expect("should parse without write_queue_capacity");
        let config: ObservabilityConfig = stored.into();
        assert_eq!(
            config.batch_writes.write_queue_capacity,
            crate::config::default_write_queue_capacity(),
            "should use default write_queue_capacity"
        );
    }
}
