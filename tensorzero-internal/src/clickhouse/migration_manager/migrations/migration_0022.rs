use async_trait::async_trait;

use super::check_table_exists;
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;

/// This migration adds the `DynamicEvaluationRun` table.
pub struct Migration0022<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0022";

#[async_trait]
impl Migration for Migration0022<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        Ok(!check_table_exists(self.clickhouse, "DynamicEvaluationRun", MIGRATION_ID).await?)
    }

    async fn apply(&self) -> Result<(), Error> {
        let query = r#"
            CREATE TABLE IF NOT EXISTS DynamicEvaluationRun
                (
                    short_key UInt64,
                    episode_id UUID,
                    variant_pins Map(String, String),
                    experiment_tags Map(String, String),
                    created_at DateTime64(6, 'UTC') DEFAULT now64(),
                )
                ENGINE = MergeTree()
                ORDER BY short_key
                SETTINGS index_granularity = 256
            "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
        -- Drop the `DynamicEvaluationRun` table\n\
        DROP TABLE IF EXISTS DynamicEvaluationRun;
    "
        .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
