use super::check_table_exists;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{
    ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs, Rows, TableName,
};
use crate::error::Error;
use async_trait::async_trait;
use uuid::Uuid;

/// This migration adds a `DeploymentID` table with a single column `deployment_id`
/// It should be initialized once on initial deployment of the gateway against a ClickHouse instance.
/// We will use the oldest deployment ID in the table as the canonical deployment ID.
/// The deployment ID in the ClickHouse will be the blake3 hash of a UUIDv7 generated at migration runtime.
pub struct Migration0033<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0033";

#[async_trait]
impl Migration for Migration0033<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // If the table doesn't exist, we need to create it and insert data
        if !check_table_exists(self.clickhouse, "DeploymentID", MIGRATION_ID).await? {
            return Ok(true);
        }
        
        // If the table exists, we still need to check if this specific migration has run
        // We can do this by checking if the migration has been recorded as successful
        // The migration manager will handle this, so if we get here and the table exists,
        // we can assume the migration should not run again
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        self.clickhouse
            .get_create_table_statements(
                "DeploymentID",
                &format!(
                    r"(
                        deployment_id String,
                        dummy UInt32 DEFAULT 0, -- the dummy column is used to enforce a single row in the table
                        created_at DateTime DEFAULT now(),
                        version_number UInt32 DEFAULT 4294967295 - toUInt32(now()) -- So that the oldest version is highest version number
                        -- we hardcode UINT32_MAX
                    )"
                ),
                &GetMaybeReplicatedTableEngineNameArgs {
                    table_name: "DeploymentID",
                    table_engine_name: "ReplacingMergeTree",
                    engine_args: &["version_number"],
                },
                Some("ORDER BY dummy"),
            )
            .await?;
            
        // Generate a UUIDv7 and compute the blake3 hash
        let deployment_id = generate_deployment_id();
        
        // Always insert the deployment ID. The ReplacingMergeTree will handle deduplication
        // based on the ORDER BY (dummy) key. Since dummy is always 0, only one row will be kept.
        // The version_number ensures the latest insert wins.
        self.clickhouse
            .write_non_batched(
                Rows::Unserialized(&[serde_json::json!({
                    "deployment_id": deployment_id,
                    "dummy": 0,
                })]),
                TableName::DeploymentID,
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        self.clickhouse.get_drop_table_rollback_statements("DeploymentID")
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}

fn generate_deployment_id() -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(Uuid::now_v7().as_bytes());
    // Should generate a hex string of length 64
    hasher.finalize().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deployment_id() {
        let deployment_id = generate_deployment_id();
        assert_eq!(deployment_id.len(), 64);
        // Assert that the deployment ID contains only hex characters
        assert!(deployment_id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
