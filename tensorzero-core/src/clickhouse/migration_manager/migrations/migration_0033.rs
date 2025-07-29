use super::{check_table_exists, table_is_nonempty, create_replacing_table_engine, create_cluster_clause};
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::error::Error;
use async_trait::async_trait;
use uuid::Uuid;

/// This migration adds a `DeploymentID` table with a single column `deployment_id`
/// It should be initialized once on initial deployment of the gateway against a ClickHouse instance.
/// We will use the oldest deployment ID in the table as the canonical deployment ID.
/// The deployment ID in the ClickHouse will be the blake3 hash of a UUIDv7 generated at migration runtime.
/// 
/// As of the replication-aware migration system, this migration creates tables with
/// the appropriate engine (replicated vs non-replicated) based on configuration.
pub struct Migration0033<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub config: &'a Config,
}

const MIGRATION_ID: &str = "0033";

#[async_trait]
impl Migration for Migration0033<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // If the table doesn't exist, we need to create it
        if !check_table_exists(self.clickhouse, "DeploymentID", MIGRATION_ID).await? {
            return Ok(true);
        }
        // If the table is empty, we need to insert the deployment ID
        if !table_is_nonempty(self.clickhouse, "DeploymentID", MIGRATION_ID).await? {
            return Ok(true);
        }
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let cluster_clause = create_cluster_clause(
            self.config.clickhouse.replication_enabled, 
            &self.config.clickhouse.cluster_name
        );
        
        let engine = create_replacing_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "DeploymentID",
            Some("version_number")
        );
        
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS DeploymentID {cluster_clause} (
                        deployment_id String,
                        dummy UInt32 DEFAULT 0, -- the dummy column is used to enforce a single row in the table
                        created_at DateTime DEFAULT now(),
                        version_number UInt32 DEFAULT 4294967295 - toUInt32(now()) -- So that the oldest version is highest version number
                        -- we hardcode UINT32_MAX
                    )
                    ENGINE = {engine}
                    ORDER BY dummy;"
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;
        // Generate a UUIDv7 and compute the blake3 hash
        let deployment_id = generate_deployment_id();
        // Insert the deployment ID into the table
        self.clickhouse
            .write(
                &[serde_json::json!({
                    "deployment_id": deployment_id,
                })],
                "DeploymentID",
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        r"DROP TABLE DeploymentID;".to_string()
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
