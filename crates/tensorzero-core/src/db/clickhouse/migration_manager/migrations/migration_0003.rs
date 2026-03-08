use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

use super::{check_column_exists, check_table_exists};

/// This migration is used to set up the ClickHouse database for tagged feedback.
/// The primary queries we contemplate are: Select all feedback for a given tag, or select all tags for a given feedback item.
/// We will store the tags in a new table `FeedbackTag` and create a materialized view for each original feedback table that writes them
/// We will also denormalize and store the tags on the original tables for efficiency.
/// There are 3 main changes:
///
///  - First, we create a new table `FeedbackTag` to store the tags
///  - Second, we add a column `tags` to each original feedback table
///  - Third, we create a materialized view for each original feedback table that writes the tags to the `FeedbackTag`
///    table as they are written to the original tables
pub struct Migration0003<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0003";

#[async_trait]
impl Migration for Migration0003<'_> {
    /// Check if the four feedback tables exist as the sources for the materialized views
    /// If all of this is OK, then we can apply the migration
    async fn can_apply(&self) -> Result<(), Error> {
        let tables = vec![
            "BooleanMetricFeedback",
            "CommentFeedback",
            "DemonstrationFeedback",
            "FloatMetricFeedback",
        ];

        for table in tables {
            if !check_table_exists(self.clickhouse, table, "0003").await? {
                return Err(ErrorDetails::ClickHouseMigration {
                    id: "0003".to_string(),
                    message: format!("Table {table} does not exist"),
                }
                .into());
            }
        }

        Ok(())
    }

    /// Check if the migration has already been applied
    /// This should be equivalent to checking if `FeedbackTag` exists
    async fn should_apply(&self) -> Result<bool, Error> {
        // Check if FeedbackTag table exists
        if !check_table_exists(self.clickhouse, "FeedbackTag", "0003").await? {
            return Ok(true);
        }

        // Check each of the original feedback tables for a `tags` column
        let tables = vec![
            "BooleanMetricFeedback",
            "CommentFeedback",
            "DemonstrationFeedback",
            "FloatMetricFeedback",
        ];

        for table in tables {
            check_column_exists(self.clickhouse, table, "tags", MIGRATION_ID).await?;
        }

        // Check that each of the materialized views exists
        let views = vec![
            "BooleanMetricFeedbackTagView",
            "CommentFeedbackTagView",
            "DemonstrationFeedbackTagView",
            "FloatMetricFeedbackTagView",
        ];

        for view in views {
            if !check_table_exists(self.clickhouse, view, "0003").await? {
                return Ok(true);
            }
        }
        // Everything is in place, so we should not apply the migration
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        // Create the `FeedbackTag` table
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "MergeTree",
                table_name: "FeedbackTag",
                engine_args: &[],
            },
        );
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS FeedbackTag{on_cluster_name}
            (
                metric_name LowCardinality(String),
                key String,
                value String,
                feedback_id UUID, -- must be a UUIDv7
            ) ENGINE = {table_engine_name}
            ORDER BY (metric_name, key, value);
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Add a column `tags` to the `BooleanMetricFeedback` table
        let query = r"
            ALTER TABLE BooleanMetricFeedback
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Add a column `tags` to the `CommentFeedback` table
        let query = r"
            ALTER TABLE CommentFeedback
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Add a column `tags` to the `DemonstrationFeedback` table
        let query = r"
            ALTER TABLE DemonstrationFeedback
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Add a column `tags` to the `FloatMetricFeedback` table
        let query = r"
            ALTER TABLE FloatMetricFeedback
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // In the following few queries we create the materialized views that map the tags from the original tables to the new `FeedbackTag` table
        // We do not need to handle the case where there are already tags in the table since we created those columns just now.
        // So, we don't worry about timestamps for cutting over to the materialized views.
        // Create the materialized view for the `FeedbackTag` table from BooleanMetricFeedback
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS BooleanMetricFeedbackTagView{on_cluster_name}
            TO FeedbackTag
            AS
                SELECT
                    metric_name,
                    key,
                    tags[key] as value,
                    id as feedback_id
                FROM BooleanMetricFeedback
                ARRAY JOIN mapKeys(tags) as key
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the materialized view for the `FeedbackTag` table from CommentFeedback
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS CommentFeedbackTagView{on_cluster_name}
            TO FeedbackTag
            AS
                SELECT
                    'comment' as metric_name,
                    key,
                    tags[key] as value,
                    id as feedback_id
                FROM CommentFeedback
                ARRAY JOIN mapKeys(tags) as key
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the materialized view for the `FeedbackTag` table from DemonstrationFeedback
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS DemonstrationFeedbackTagView{on_cluster_name}
            TO FeedbackTag
            AS
                SELECT
                    'demonstration' as metric_name,
                    key,
                    tags[key] as value,
                    id as feedback_id
                FROM DemonstrationFeedback
                ARRAY JOIN mapKeys(tags) as key
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the materialized view for the `FeedbackTag` table from FloatMetricFeedback
        let query = format!(
            r"
            CREATE MATERIALIZED VIEW IF NOT EXISTS FloatMetricFeedbackTagView{on_cluster_name}
            TO FeedbackTag
            AS
                SELECT
                    metric_name,
                    key,
                    tags[key] as value,
                    id as feedback_id
                FROM FloatMetricFeedback
                ARRAY JOIN mapKeys(tags) as key
            "
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(
            "/* Drop the materialized views */\
            DROP VIEW IF EXISTS BooleanMetricFeedbackTagView{on_cluster_name};
            DROP VIEW IF EXISTS CommentFeedbackTagView{on_cluster_name};
            DROP VIEW IF EXISTS DemonstrationFeedbackTagView{on_cluster_name};
            DROP VIEW IF EXISTS FloatMetricFeedbackTagView{on_cluster_name};
            /* Drop the table */\
            DROP TABLE IF EXISTS FeedbackTag{on_cluster_name} SYNC;
            /* Drop the columns */\
            ALTER TABLE BooleanMetricFeedback DROP COLUMN tags;
            ALTER TABLE CommentFeedback DROP COLUMN tags;
            ALTER TABLE DemonstrationFeedback DROP COLUMN tags;
            ALTER TABLE FloatMetricFeedback DROP COLUMN tags;"
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
