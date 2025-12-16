use super::check_column_exists;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::Error;
use async_trait::async_trait;

const MIGRATION_ID: &str = "0045";

/// This migration adds a `tags` column to the `ConfigSnapshot` table.
pub struct Migration0045<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0045<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        Ok(!check_column_exists(self.clickhouse, "ConfigSnapshot", "tags", MIGRATION_ID).await?)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "ALTER TABLE ConfigSnapshot{on_cluster_name} ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map()"
            ))
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!("ALTER TABLE ConfigSnapshot{on_cluster_name} DROP COLUMN tags;")
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        Ok(check_column_exists(self.clickhouse, "ConfigSnapshot", "tags", MIGRATION_ID).await?)
    }
}
