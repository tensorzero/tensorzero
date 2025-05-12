use std::time::Duration;

use async_trait::async_trait;

use super::check_table_exists;
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

/// This migration adds the `ModelUsageInfo` table
/// and the `ModelUsageInfoView` materialized view.
/// These will allow us to build a dashboard tracking model usage across
/// all functions in a deployment.
pub struct Migration0029<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0029";

#[async_trait]
impl Migration for Migration0029<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let model_usage_info_table_exists =
            check_table_exists(self.clickhouse, "ModelUsageInfo", MIGRATION_ID).await?;
        let model_usage_info_view_exists =
            check_table_exists(self.clickhouse, "ModelUsageInfoView", MIGRATION_ID).await?;
        Ok(!model_usage_info_table_exists || !model_usage_info_view_exists)
    }

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
        // Only gets used when we are not doing a clean start
        let view_offset = Duration::from_secs(15);
        let current_time = std::time::SystemTime::now();
        let view_timestamp = (current_time
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: MIGRATION_ID.to_string(),
                    message: e.to_string(),
                })
            })?
            + view_offset)
            .as_secs();
        let view_timestamp_where_clause = if !clean_start {
            format!("WHERE updated_at >= toDateTime(toUnixTimestamp({view_timestamp}))")
        } else {
            String::new()
        };
        // Create the `ModelUsageInfo` table if it doesn't exist
        let query = r#"
            CREATE TABLE IF NOT EXISTS ModelUsageInfo
                (
                    model_name LowCardinality(String),
                    model_provider_name LowCardinality(String),
                    input_tokens Nullable(UInt32),
                    output_tokens Nullable(UInt32),
                    response_time_ms Nullable(UInt32),
                    ttft_ms Nullable(UInt32),
                    id_uint UInt128,
                    updated_at DateTime64(6, 'UTC') DEFAULT now(),
                    is_deleted Bool DEFAULT false,
                    INDEX idx_updated_at updated_at TYPE minmax GRANULARITY 1
                ) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
                ORDER BY (model_name, model_provider_name, id_uint);
        "#;
        let _ = self
            .clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        // Create the `ModelUsageInfoView` materialized view if it doesn't exist
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS ModelUsageInfoView
                TO ModelUsageInfo
                AS SELECT
                    model_name,
                    model_provider_name,
                    input_tokens,
                    output_tokens,
                    response_time_ms,
                    ttft_ms,
                    toUInt128(id) AS id_uint,
                    now() AS updated_at,
                    false AS is_deleted
                FROM ModelInference
                {view_timestamp_where_clause};
        "#,
        );
        let _ = self
            .clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        if !clean_start {
            tokio::time::sleep(view_offset).await;
            let query = format!(
                r#"
                INSERT INTO ModelUsageInfo
                SELECT
                    model_name,
                    model_provider_name,
                    input_tokens,
                    output_tokens,
                    response_time_ms,
                    ttft_ms,
                    toUInt128(id) AS id_uint,
                    now() AS updated_at,
                    false AS is_deleted
                FROM ModelInference
                WHERE updated_at < toDateTime(toUnixTimestamp({view_timestamp}));
                "#,
            );
            let _ = self
                .clickhouse
                .run_query_synchronous(query.to_string(), None)
                .await?;
        }
        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "DROP VIEW IF EXISTS ModelUsageInfoView\nDROP TABLE IF EXISTS ModelUsageInfo".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
