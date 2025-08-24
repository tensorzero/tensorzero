use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::migration_manager::migrations::get_column_type;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;

/// This migration adds a 'stop_sequence variant to the 'finish_reason enum on ModelInference/ModelInferenceCache
pub struct Migration0030<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0030";

const NEW_COLUMN_TYPE: &str = "Nullable(Enum8(\\'stop\\' = 1, \\'length\\' = 2, \\'tool_call\\' = 3, \\'content_filter\\' = 4, \\'unknown\\' = 5, \\'stop_sequence\\' = 6))";

#[async_trait]
impl Migration for Migration0030<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let first_column_type = get_column_type(
            self.clickhouse,
            "ModelInference",
            "finish_reason",
            MIGRATION_ID,
        )
        .await?;
        if first_column_type != NEW_COLUMN_TYPE {
            return Ok(true);
        }

        let second_column_type = get_column_type(
            self.clickhouse,
            "ModelInferenceCache",
            "finish_reason",
            MIGRATION_ID,
        )
        .await?;
        if second_column_type != NEW_COLUMN_TYPE {
            return Ok(true);
        }

        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ModelInference
                MODIFY COLUMN finish_reason Nullable(Enum8('stop' = 1, 'length' = 2, 'tool_call' = 3, 'content_filter' = 4, 'unknown' = 5, 'stop_sequence' = 6));".to_string(),
            )
            .await?;

        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ModelInferenceCache
                MODIFY COLUMN finish_reason Nullable(Enum8('stop' = 1, 'length' = 2, 'tool_call' = 3, 'content_filter' = 4, 'unknown' = 5, 'stop_sequence' = 6));".to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        r"ALTER TABLE ModelInference MODIFY COLUMN finish_reason Nullable(Enum8('stop' = 1, 'length' = 2, 'tool_call' = 3, 'content_filter' = 4, 'unknown' = 5));
           ALTER TABLE ModelInferenceCache MODIFY COLUMN finish_reason Nullable(Enum8('stop' = 1, 'length' = 2, 'tool_call' = 3, 'content_filter' = 4, 'unknown' = 5));".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
