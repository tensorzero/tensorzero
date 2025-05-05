use std::time::Duration;

use async_trait::async_trait;

use super::check_table_exists;
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;

/// This migration adds the `ModelUsageInfo` table
/// and the `ModelUsageInfoView` materialized view.
/// These will allow us to build a dashboard tracking model usage across
/// all functions in a deployment.
pub struct Migration0027<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub clean_start: bool,
}

#[async_trait]
impl Migration for Migration0027<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let model_usage_info_table_exists =
            check_table_exists(self.clickhouse, "ModelUsageInfo", "0027").await?;
        let model_usage_info_view_exists =
            check_table_exists(self.clickhouse, "ModelUsageInfoView", "0027").await?;
        Ok(!model_usage_info_table_exists || !model_usage_info_view_exists)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Only gets used when we are not doing a clean start
        let view_offset = Duration::from_secs(15);

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
                    inference_id UUID,
                    updated_at DateTime64(6, 'UTC') DEFAULT now(),
                    is_deleted Bool DEFAULT false
                ) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
                ORDER BY (model_name, model_provider_name, id_uint);
        "#;
        let _ = self
            .clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        let query = r#"
            CREATE TABLE IF NOT EXISTS DynamicEvaluationRunEpisode
            (
                run_id UUID,
                episode_id_uint UInt128, -- UUID encoded as a UInt128
                -- this is duplicated so that we can look it up without joining at inference time
                variant_pins Map(String, String),
                datapoint_name Nullable(String),
                tags Map(String, String),
                is_deleted Bool DEFAULT false,
                updated_at DateTime64(6, 'UTC') DEFAULT now()
            ) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
                ORDER BY episode_id_uint;
        "#;
        let _ = self
            .clickhouse
            .run_query_synchronous(query.to_string(), None)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "DROP TABLE IF EXISTS DynamicEvaluationRun\nDROP TABLE IF EXISTS DynamicEvaluationRunEpisode"
            .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
