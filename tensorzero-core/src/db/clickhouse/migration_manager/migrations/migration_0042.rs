use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

pub struct Migration0042<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0042<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let query =
            "SELECT 1 FROM system.functions WHERE name = 'tensorzero_uint_to_uuid'".to_string();
        let result = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;
        if result.response.contains('1') {
            return Ok(false);
        }
        if result.response.trim().is_empty() || result.response.contains("0") {
            return Ok(true);
        }
        Err(Error::new(ErrorDetails::ClickHouseMigration {
            id: "0042".to_string(),
            message: format!(
                "Unexpected response when checking for tensorzero_uint_to_uuid function: {}",
                result.response.trim()
            ),
        }))
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let query = format!(
            r"CREATE FUNCTION IF NOT EXISTS tensorzero_uint_to_uuid{on_cluster_name} AS (x) -> reinterpretAsUUID(
            concat(
                substring(reinterpretAsString(x), 9, 8),
                substring(reinterpretAsString(x), 1, 8)
            )
        );",
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        // Note - we deliberately don't include any rollback instructions.
        // User-defined functions are global, so dropping them would interfere with concurrent migration runs.
        "".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
