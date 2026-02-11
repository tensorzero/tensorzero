use super::check_column_exists;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::Error;
use async_trait::async_trait;

const MIGRATION_ID: &str = "0046";

/// This migration adds a `cost` column to the `ModelInference` table.
/// Cost is stored as `Nullable(Decimal(18, 9))` — 18 total digits, 9 fractional —
/// matching the Postgres `NUMERIC(18, 9)` column.
pub struct Migration0046<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0046<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        Ok(!check_column_exists(self.clickhouse, "ModelInference", "cost", MIGRATION_ID).await?)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "ALTER TABLE ModelInference{on_cluster_name} ADD COLUMN IF NOT EXISTS cost Nullable(Decimal(18, 9))"
            ))
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!("ALTER TABLE ModelInference{on_cluster_name} DROP COLUMN cost;")
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        Ok(check_column_exists(self.clickhouse, "ModelInference", "cost", MIGRATION_ID).await?)
    }
}
