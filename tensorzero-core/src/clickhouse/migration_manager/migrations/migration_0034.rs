use std::time::Duration;

use super::check_table_exists;
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

/// This migration adds a `TokenTotal` table and `TokenTotalView` materialized view
/// This will allow the sum of tokens in the ModelInference table to be amortized looked up as needed.
/// TODO (Viraj): will concurrent migrations break this?
pub struct Migration0034<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0034";

#[async_trait]
impl Migration for Migration0034<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        if !check_table_exists(self.clickhouse, "ModelInference", MIGRATION_ID).await? {
            return Err(Error::new(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "ModelInference table does not exist".to_string(),
            }));
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // If either the TokenTotal table or TokenTotalView view doesn't exist, we need to create it
        if !check_table_exists(self.clickhouse, "TokenTotal", MIGRATION_ID).await? {
            return Ok(true);
        }
        if !check_table_exists(self.clickhouse, "TokenTotalView", MIGRATION_ID).await? {
            return Ok(true);
        }
        Ok(false)
    }

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
        let view_offset = Duration::from_secs(15);
        let view_timestamp = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0009".to_string(),
                    message: e.to_string(),
                })
            })?
            + view_offset)
            .as_secs();
        self.clickhouse
            .run_query_synchronous_no_params(
                r#"CREATE TABLE IF NOT EXISTS TokenTotal (
                        type LowCardinality(String),
                        count UInt64,
                    )
                    ENGINE = SummingMergeTree
                    ORDER BY type;"#
                    .to_string(),
            )
            .await?;

        // Create the materialized view for the TokenTotal table from ModelInference
        // If we are not doing a clean start, we need to add a where clause ot the view to only include rows that have been created
        // after the view_timestamp
        let view_where_clause = if !clean_start {
            format!("AND UUIDv7ToDateTime(id) >= toDateTime(toUnixTimestamp({view_timestamp}))")
        } else {
            String::new()
        };
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS TokenTotalView
            TO TokenTotal
            AS
                    SELECT
                        'input' as type,
                        input_tokens as count
                    FROM ModelInference
                    WHERE input_tokens IS NOT NULL
                    {view_where_clause}
                UNION ALL
                    SELECT
                        'output' as type,
                        output_tokens as count
                    FROM ModelInference
                    WHERE output_tokens IS NOT NULL
                    {view_where_clause}

            "#
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // If we are not clean starting, we must backfill this table
        if !clean_start {
            tokio::time::sleep(view_offset).await;
            let query = format!(
                r#"
                INSERT INTO TokenTotal
                SELECT
                    'input' as type,
                    input_tokens as count
                FROM ModelInference
                WHERE input_tokens IS NOT NULL
                AND UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}))
                UNION ALL
                SELECT
                    'output' as type,
                    output_tokens as count
                FROM ModelInference
                WHERE output_tokens IS NOT NULL
                AND UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "#
            );
            self.clickhouse
                .run_query_synchronous_no_params(query)
                .await?;
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        r#"
        DROP TABLE TokenTotalView;
        DROP TABLE TokenTotal;"#
            .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
