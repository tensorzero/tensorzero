use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{ClickHouseConnectionInfo, ExternalDataInfo};
use crate::config::snapshot::{ConfigSnapshot, SnapshotHash};
use crate::db::ConfigQueries;
use crate::error::{DelayedError, Error, ErrorDetails};

#[async_trait]
impl ConfigQueries for ClickHouseConnectionInfo {
    async fn get_config_snapshot(
        &self,
        snapshot_hash: SnapshotHash,
    ) -> Result<ConfigSnapshot, Error> {
        #[derive(Deserialize)]
        struct ConfigSnapshotRow {
            config: String,
            extra_templates: HashMap<String, String>,
            #[serde(default)]
            tags: HashMap<String, String>,
        }

        // ClickHouse stores `hash` as `UInt256`; only the decimal digits are
        // valid input to `toUInt256(...)`. The `Display` form prefixes
        // `can:` for canonical hashes (legacy is bare decimal already),
        // so use `to_decimal_string()` for the SQL literal. The error
        // message uses `Display` so callers can tell schemes apart.
        let hash_decimal = snapshot_hash.to_decimal_string();
        let query = format!(
            "SELECT config, extra_templates, tags \
             FROM ConfigSnapshot FINAL \
             WHERE hash = toUInt256('{hash_decimal}') \
             LIMIT 1 \
             FORMAT JSONEachRow"
        );

        let response = self.run_query_synchronous_no_params(query).await?;

        if response.response.is_empty() {
            return Err(Error::new(ErrorDetails::ConfigSnapshotNotFound {
                snapshot_hash: snapshot_hash.to_string(),
            }));
        }

        let row: ConfigSnapshotRow = serde_json::from_str(&response.response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })?;

        ConfigSnapshot::from_stored(&row.config, row.extra_templates, row.tags, &snapshot_hash)
    }

    async fn write_config_snapshot(&self, snapshot: &ConfigSnapshot) -> Result<(), DelayedError> {
        // The CH `hash` column is `UInt256`. We send the decimal-only
        // form through the JSONEachRow body and inline literals —
        // `SnapshotHash`'s `Display` / `Serialize` impls prefix `can:`
        // for canonical hashes, which `toUInt256` cannot parse.
        //
        // Going forward, CH only stores the **canonical** hash here:
        // every reader (e.g. `/internal/config/{hash}`) routes through
        // `Config.hash` which is now canonical (computed from the
        // structural JSON form). Postgres has both `hash` (legacy) and
        // `canonical_hash` columns; CH currently has one column. A
        // follow-up adds a CH-side migration for parity (see task
        // "ClickHouse parity for snapshot canonical_hash"). For now
        // the column holds canonical bytes, matching the lookup.
        #[derive(Serialize)]
        struct ConfigSnapshotRow<'a> {
            config: &'a str,
            extra_templates: &'a HashMap<String, String>,
            hash: &'a str,
            tensorzero_version: &'static str,
            tags: &'a HashMap<String, String>,
        }

        let canonical_hash = snapshot.config.canonical_hash().map_err(|e| {
            DelayedError::new(ErrorDetails::Serialization {
                message: format!("Failed to compute canonical hash: {e}"),
            })
        })?;
        let hash_decimal = canonical_hash.to_decimal_string().to_string();
        let hash_decimal = hash_decimal.as_str();

        let config_string = toml::to_string(&snapshot.config).map_err(|e| {
            DelayedError::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize config snapshot: {e}"),
            })
        })?;

        let row = ConfigSnapshotRow {
            config: &config_string,
            extra_templates: &snapshot.extra_templates,
            hash: hash_decimal,
            tensorzero_version: crate::endpoints::status::TENSORZERO_VERSION,
            tags: &snapshot.tags,
        };

        let json_data = serde_json::to_string(&row).map_err(|e| {
            DelayedError::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize config snapshot: {e}"),
            })
        })?;

        let external_data = ExternalDataInfo {
            external_data_name: "new_data".to_string(),
            structure: "config String, extra_templates Map(String, String), hash String, tensorzero_version String, tags Map(String, String)".to_string(),
            format: "JSONEachRow".to_string(),
            data: json_data,
        };

        let query = format!(
            r"INSERT INTO ConfigSnapshot
(config, extra_templates, hash, tensorzero_version, tags, created_at, last_used)
SELECT
    new_data.config,
    new_data.extra_templates,
    toUInt256(new_data.hash) as hash,
    new_data.tensorzero_version,
    mapUpdate(
        (SELECT any(tags) FROM ConfigSnapshot FINAL WHERE hash = toUInt256('{hash_decimal}')),
        new_data.tags
    ) as tags,
    ifNull((SELECT any(created_at) FROM ConfigSnapshot FINAL WHERE hash = toUInt256('{hash_decimal}')), now64()) as created_at,
    now64() as last_used
FROM new_data"
        );

        self.run_query_with_external_data(external_data, query)
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
    use crate::db::clickhouse::{ClickHouseResponse, ClickHouseResponseMetadata};
    use crate::db::test_helpers::assert_query_contains;

    #[tokio::test]
    async fn test_get_config_snapshot_found() {
        let mut mock = MockClickHouseClient::new();

        mock.expect_run_query_synchronous()
            .withf(|query, _params| {
                assert_query_contains(query, "SELECT config, extra_templates, tags");
                assert_query_contains(query, "FROM ConfigSnapshot FINAL");
                assert_query_contains(query, "LIMIT 1");
                assert_query_contains(query, "FORMAT JSONEachRow");
                true
            })
            .returning(|_, _| {
                let response =
                    r#"{"config":"[functions]\n","extra_templates":{},"tags":{"env":"test"}}"#;
                Ok(ClickHouseResponse {
                    response: response.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
        let hash = SnapshotHash::new_test();
        let result = conn.get_config_snapshot(hash).await;
        assert!(result.is_ok(), "Should successfully parse config snapshot");
    }

    #[tokio::test]
    async fn test_get_config_snapshot_not_found() {
        let mut mock = MockClickHouseClient::new();

        mock.expect_run_query_synchronous().returning(|_, _| {
            Ok(ClickHouseResponse {
                response: String::new(),
                metadata: ClickHouseResponseMetadata {
                    read_rows: 0,
                    written_rows: 0,
                },
            })
        });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
        let hash = SnapshotHash::new_test();
        let result = conn.get_config_snapshot(hash).await;
        assert!(
            result.is_err(),
            "Should return error when snapshot not found"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(
                err.get_details(),
                ErrorDetails::ConfigSnapshotNotFound { .. }
            ),
            "Error should be ConfigSnapshotNotFound"
        );
    }

    #[tokio::test]
    #[expect(clippy::disallowed_methods)]
    async fn test_write_config_snapshot() {
        let mut mock = MockClickHouseClient::new();

        mock.expect_run_query_with_external_data()
            .withf(|external_data, query| {
                assert_query_contains(query, "INSERT INTO ConfigSnapshot");
                assert_eq!(
                    external_data.external_data_name, "new_data",
                    "External data name should be `new_data`"
                );
                assert_eq!(
                    external_data.format, "JSONEachRow",
                    "Format should be JSONEachRow"
                );
                assert!(
                    external_data.structure.contains("config String"),
                    "Structure should include config column"
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 1,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock));
        let snapshot = ConfigSnapshot::new_empty_for_test();
        let result = conn.write_config_snapshot(&snapshot).await;
        assert!(result.is_ok(), "Should successfully write config snapshot");
    }
}
