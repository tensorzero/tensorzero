use super::get_column_type;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::Error;
use async_trait::async_trait;

/// This migration fixes an issue with migration 0042.
/// That migration did not correctly handle replicated deployments.
/// It makes the `input_tokens` and `output_tokens` columns nullable
/// in the `ModelInferenceCache` table to match the Rust type definition (Option<u32>)
/// and handle cases where providers don't return usage information.
pub struct Migration0044<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}
const MIGRATION_ID: &str = "0044";

#[async_trait]
impl Migration for Migration0044<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // Note: since we could have run 0042 against this node, it will be challenging
        // to check if this needs to run cluster-wide. So, we can just run this migration
        // at-least-once to make sure we run a command that propagates the DDL to all nodes.
        let response = self
            .clickhouse
            .run_query_synchronous_no_params(
                "SELECT 1 FROM TensorZeroMigration WHERE migration_id = 44 LIMIT 1".to_string(),
            )
            .await?;
        return Ok(response.response.trim() != "1");
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        self.clickhouse
            .run_query_synchronous_no_params(format!(
                "ALTER TABLE ModelInferenceCache{on_cluster_name}
                MODIFY COLUMN input_tokens Nullable(UInt32)"
            ))
            .await?;

        self.clickhouse
            .run_query_synchronous_no_params(format!(
                r"ALTER TABLE ModelInferenceCache{on_cluster_name}
                    MODIFY COLUMN output_tokens Nullable(UInt32)"
            ))
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            "ALTER TABLE ModelInferenceCache{on_cluster_name} MODIFY COLUMN input_tokens UInt32 DEFAULT 0;
          ALTER TABLE ModelInferenceCache{on_cluster_name} MODIFY COLUMN output_tokens UInt32 DEFAULT 0;"
        )
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        // NOTE: this is best-effort since we can't iterate over every node in the cluster.
        if get_column_type(
            self.clickhouse,
            "ModelInferenceCache",
            "input_tokens",
            MIGRATION_ID,
        )
        .await?
            != "Nullable(UInt32)"
        {
            return Ok(false);
        }

        if get_column_type(
            self.clickhouse,
            "ModelInferenceCache",
            "output_tokens",
            MIGRATION_ID,
        )
        .await?
            != "Nullable(UInt32)"
        {
            return Ok(false);
        }
        Ok(true)
    }
}
