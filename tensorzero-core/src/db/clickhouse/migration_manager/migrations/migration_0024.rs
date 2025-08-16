use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

use super::{check_column_exists, check_table_exists};

/// This migration adds a column to the `JsonInference` table called
/// `auxiliary_content`.
/// This column should contain a JSON array of `ContentBlock` objects.
/// It is used to store any auxiliary content that is part of the process of
/// producing a JSON inference. This is used for things like `Thought` blocks
/// in chain of thought inferences.
pub struct Migration0024<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0024<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        let json_inference_table_exists =
            check_table_exists(self.clickhouse, "JsonInference", "0024").await?;
        if !json_inference_table_exists {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: "0024".to_string(),
                message: "JsonInference table does not exist".to_string(),
            }));
        }

        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let auxiliary_content_column_exists = check_column_exists(
            self.clickhouse,
            "JsonInference",
            "auxiliary_content",
            "0024",
        )
        .await?;

        Ok(!auxiliary_content_column_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        self.clickhouse
            .run_query_synchronous_no_params(
                "ALTER TABLE JsonInference ADD COLUMN IF NOT EXISTS auxiliary_content String"
                    .to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "ALTER TABLE JsonInference DROP COLUMN auxiliary_content".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
