use async_trait::async_trait;

use super::check_index_exists;
use super::check_table_exists;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::error::{Error, ErrorDetails};

/// This migration adds bloom filter indices on `id` columns to the feedback tables:
/// `BooleanMetricFeedback`, `FloatMetricFeedback`, `CommentFeedback`, and `DemonstrationFeedback`.
/// This allows efficient lookup of feedback rows by their primary ID (used by the resolve_uuid endpoint).
pub struct Migration0046<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const TABLES: [&str; 4] = [
    "BooleanMetricFeedback",
    "FloatMetricFeedback",
    "CommentFeedback",
    "DemonstrationFeedback",
];

#[async_trait]
impl Migration for Migration0046<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        for table in &TABLES {
            if !check_table_exists(self.clickhouse, table, "0046").await? {
                return Err(ErrorDetails::ClickHouseMigration {
                    id: "0046".to_string(),
                    message: format!("{table} table does not exist"),
                }
                .into());
            }
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        for table in &TABLES {
            if !check_index_exists(self.clickhouse, table, "id_index").await? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        for table in &TABLES {
            let create_index_query = format!(
                "ALTER TABLE {table} ADD INDEX IF NOT EXISTS id_index id TYPE bloom_filter GRANULARITY 1;"
            );
            let _ = self
                .clickhouse
                .run_query_synchronous_no_params(create_index_query)
                .await?;

            let materialize_index_query =
                format!("ALTER TABLE {table} MATERIALIZE INDEX id_index;");
            let _ = self
                .clickhouse
                .run_query_synchronous_no_params(materialize_index_query)
                .await?;
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        TABLES
            .iter()
            .map(|table| format!("ALTER TABLE {table} DROP INDEX IF EXISTS id_index;"))
            .collect::<Vec<_>>()
            .join("\n        ")
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
