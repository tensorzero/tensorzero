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
    pub backend: Option<ObservabilityBackend>,
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
    pub flush_interval_ms: u64,
    pub max_rows: usize,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_rows_postgres: Option<usize>,
    /// `None` means unbounded (legacy behavior).
    /// `Some(n)` means bounded channels with capacity `n`.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write_queue_capacity: Option<usize>,
}

impl Default for StoredBatchWritesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            __force_allow_embedded_batch_writes: false,
            flush_interval_ms: crate::config::default_flush_interval_ms(),
            max_rows: crate::config::default_max_rows(),
            max_rows_postgres: None,
            write_queue_capacity: None,
        }
    }
}

impl From<BatchWritesConfig> for StoredBatchWritesConfig {
    fn from(config: BatchWritesConfig) -> Self {
        let BatchWritesConfig {
            enabled,
            __force_allow_embedded_batch_writes,
            flush_interval_ms,
            max_rows,
            max_rows_postgres,
            write_queue_capacity,
        } = config;
        Self {
            enabled,
            __force_allow_embedded_batch_writes: __force_allow_embedded_batch_writes
                .unwrap_or_default(),
            flush_interval_ms: flush_interval_ms
                .unwrap_or_else(crate::config::default_flush_interval_ms),
            max_rows: max_rows.unwrap_or_else(crate::config::default_max_rows),
            max_rows_postgres,
            write_queue_capacity,
        }
    }
}

impl From<StoredBatchWritesConfig> for BatchWritesConfig {
    fn from(stored: StoredBatchWritesConfig) -> Self {
        let StoredBatchWritesConfig {
            enabled,
            __force_allow_embedded_batch_writes,
            flush_interval_ms,
            max_rows,
            max_rows_postgres,
            write_queue_capacity,
        } = stored;
        Self {
            enabled,
            __force_allow_embedded_batch_writes: Some(__force_allow_embedded_batch_writes),
            flush_interval_ms: Some(flush_interval_ms),
            max_rows: Some(max_rows),
            max_rows_postgres,
            write_queue_capacity,
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
            async_writes: async_writes.unwrap_or_default(),
            batch_writes: batch_writes.unwrap_or_default().into(),
            disable_automatic_migrations: disable_automatic_migrations.unwrap_or_default(),
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
            async_writes: Some(async_writes),
            batch_writes: Some(batch_writes.into()),
            #[expect(deprecated)]
            disable_automatic_migrations: Some(disable_automatic_migrations),
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
    /// stored configs didn't include this field. They should parse with `None` (unbounded).
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
        let batch_writes = config.batch_writes.expect("batch_writes should be Some");
        assert_eq!(
            batch_writes.write_queue_capacity, None,
            "should default to None (unbounded) when not set"
        );
    }

    /// Historical: before async_writes defaulted to enabled, stored configs without
    /// an `async_writes` field should still parse with `async_writes: false`.
    #[test]
    fn test_historical_no_async_writes_defaults_to_disabled() {
        let toml_str = r"
            enabled = true
        ";

        let stored: StoredObservabilityConfig =
            toml::from_str(toml_str).expect("should parse without async_writes field");
        assert!(
            !stored.async_writes,
            "stored snapshot without async_writes should default to false"
        );
        let config: ObservabilityConfig = stored.into();
        assert_eq!(
            config.async_writes,
            Some(false),
            "converted config should preserve disabled async_writes from stored snapshot"
        );
    }

    /// Stored configs that have an explicit `write_queue_capacity` should preserve it.
    #[test]
    fn test_explicit_write_queue_capacity() {
        let toml_str = r"
            enabled = true

            [batch_writes]
            enabled = true
            flush_interval_ms = 100
            max_rows = 500
            write_queue_capacity = 5000
        ";

        let stored: StoredObservabilityConfig =
            toml::from_str(toml_str).expect("should parse with explicit write_queue_capacity");
        let config: ObservabilityConfig = stored.into();
        let batch_writes = config.batch_writes.expect("batch_writes should be Some");
        assert_eq!(
            batch_writes.write_queue_capacity,
            Some(5000),
            "should preserve explicit write_queue_capacity"
        );
    }
}
