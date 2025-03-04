use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager::migration_trait::Migration;
use crate::error::Error;
use async_trait::async_trait;

use super::{check_table_exists, get_column_type};

/// This migration alters the updated_at column in the `ChatInferenceDataset` and `JsonInferenceDataset` tables.``
/// so that times are explicitly stored in UTC
pub struct Migration0016<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0016<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        check_table_exists(self.clickhouse, "ChatInferenceDataset", "0016").await?;
        check_table_exists(self.clickhouse, "JsonInferenceDataset", "0016").await?;

        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let column_type = get_column_type(
            self.clickhouse,
            "ChatInferenceDataset",
            "updated_at",
            "0016",
        )
        .await?;
        if column_type != "DateTime(\\'UTC\\')" {
            return Ok(true);
        }
        let column_type = get_column_type(
            self.clickhouse,
            "JsonInferenceDataset",
            "updated_at",
            "0016",
        )
        .await?;
        if column_type != "DateTime(\\'UTC\\')" {
            return Ok(true);
        }

        Ok(false)
    }

    async fn apply(&self) -> Result<(), Error> {
        let query =
            "ALTER TABLE ChatInferenceDataset MODIFY COLUMN updated_at DateTime('UTC')".to_string();
        self.clickhouse.run_query(query, None).await?;

        let query =
            "ALTER TABLE JsonInferenceDataset MODIFY COLUMN updated_at DateTime('UTC')".to_string();
        self.clickhouse.run_query(query, None).await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "ALTER TABLE ChatInferenceDataset MODIFY COLUMN updated_at DateTime;
        ALTER TABLE JsonInferenceDataset MODIFY COLUMN updated_at DateTime;"
            .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
