use async_trait::async_trait;

use super::{check_index_exists, check_table_exists};
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

/// This migration adds an index by `inference_id` to the `TagInference` table.
/// This allows us to efficiently query for all tag values for a given key for a given inference.
pub struct Migration0027<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0027<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        if !check_table_exists(self.clickhouse, "TagInference", "0027").await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0027".to_string(),
                message: "TagInference table does not exist".to_string(),
            }
            .into());
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let index_exists =
            check_index_exists(self.clickhouse, "TagInference", "inference_id_index").await?;
        Ok(!index_exists)
    }

    async fn apply(&self) -> Result<(), Error> {
        let create_index_query = r#"
            ALTER TABLE TagInference ADD INDEX inference_id_index inference_id TYPE bloom_filter GRANULARITY 1;
        "#;
        let _ = self
            .clickhouse
            .run_query_synchronous(create_index_query.to_string(), None)
            .await?;

        let materialize_index_query = r#"
            ALTER TABLE TagInference MATERIALIZE INDEX inference_id_index;
        "#;
        let _ = self
            .clickhouse
            .run_query_synchronous(materialize_index_query.to_string(), None)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "ALTER TABLE TagInference DROP INDEX IF EXISTS inference_id_index".to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
