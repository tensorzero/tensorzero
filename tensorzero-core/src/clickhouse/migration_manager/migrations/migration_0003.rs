use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::migration_manager::migrations::{
    check_table_exists, create_table_engine, create_cluster_clause
};
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

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
/// 
/// As of the replication-aware migration system, this migration creates tables with
/// the appropriate engine (replicated vs non-replicated) based on configuration.
pub struct Migration0003<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub config: &'a Config,
}

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

        let database = self.clickhouse.database();

        for table in tables {
            let query = format!(
                r"SELECT EXISTS(
                    SELECT 1
                    FROM system.columns
                    WHERE database = '{database}'
                      AND table = '{table}'
                      AND name = 'tags'
                )"
            );
            match self.clickhouse.run_query_synchronous_no_params(query).await {
                Err(e) => {
                    return Err(ErrorDetails::ClickHouseMigration {
                        id: "0003".to_string(),
                        message: e.to_string(),
                    }
                    .into());
                }
                Ok(response) => {
                    if response.response.trim() != "1" {
                        return Ok(true);
                    }
                }
            }
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
        let cluster_clause = create_cluster_clause(
            self.config.clickhouse.replication_enabled, 
            &self.config.clickhouse.cluster_name
        );
        
        // Create the `FeedbackTag` table
        let engine = create_table_engine(
            self.config.clickhouse.replication_enabled,
            &self.config.clickhouse.cluster_name,
            "FeedbackTag"
        );
        let query = format!(
            r"CREATE TABLE IF NOT EXISTS FeedbackTag {cluster_clause}
            (
                metric_name LowCardinality(String),
                key String,
                value String,
                feedback_id UUID, -- must be a UUIDv7
            ) ENGINE = {engine}
            ORDER BY (metric_name, key, value);"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Add a column `tags` to the `BooleanMetricFeedback` table
        let query = format!(
            "ALTER TABLE BooleanMetricFeedback {cluster_clause}
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Add a column `tags` to the `CommentFeedback` table
        let query = format!(
            "ALTER TABLE CommentFeedback {cluster_clause}
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Add a column `tags` to the `DemonstrationFeedback` table
        let query = format!(
            "ALTER TABLE DemonstrationFeedback {cluster_clause}
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Add a column `tags` to the `FloatMetricFeedback` table
        let query = format!(
            "ALTER TABLE FloatMetricFeedback {cluster_clause}
            ADD COLUMN IF NOT EXISTS tags Map(String, String) DEFAULT map();"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // In the following few queries we create the materialized views that map the tags from the original tables to the new `FeedbackTag` table
        // We do not need to handle the case where there are already tags in the table since we created those columns just now.
        // So, we don't worry about timestamps for cutting over to the materialized views.
        // Create the materialized view for the `FeedbackTag` table from BooleanMetricFeedback
        let query = format!(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS BooleanMetricFeedbackTagView {cluster_clause}
            TO FeedbackTag
            AS
                SELECT
                    metric_name,
                    key,
                    tags[key] as value,
                    id as feedback_id
                FROM BooleanMetricFeedback
                ARRAY JOIN mapKeys(tags) as key"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Create the materialized view for the `FeedbackTag` table from CommentFeedback
        let query = format!(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS CommentFeedbackTagView {cluster_clause}
            TO FeedbackTag
            AS
                SELECT
                    'comment' as metric_name,
                    key,
                    tags[key] as value,
                    id as feedback_id
                FROM CommentFeedback
                ARRAY JOIN mapKeys(tags) as key"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Create the materialized view for the `FeedbackTag` table from DemonstrationFeedback
        let query = format!(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS DemonstrationFeedbackTagView {cluster_clause}
            TO FeedbackTag
            AS
                SELECT
                    'demonstration' as metric_name,
                    key,
                    tags[key] as value,
                    id as feedback_id
                FROM DemonstrationFeedback
                ARRAY JOIN mapKeys(tags) as key"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Create the materialized view for the `FeedbackTag` table from FloatMetricFeedback
        let query = format!(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS FloatMetricFeedbackTagView {cluster_clause}
            TO FeedbackTag
            AS
                SELECT
                    metric_name,
                    key,
                    tags[key] as value,
                    id as feedback_id
                FROM FloatMetricFeedback
                ARRAY JOIN mapKeys(tags) as key"
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let cluster_clause = create_cluster_clause(
            self.config.clickhouse.replication_enabled, 
            &self.config.clickhouse.cluster_name
        );
        format!(
            "/* Drop the materialized views */\
            DROP VIEW IF EXISTS BooleanMetricFeedbackTagView {cluster_clause};
            DROP VIEW IF EXISTS CommentFeedbackTagView {cluster_clause};
            DROP VIEW IF EXISTS DemonstrationFeedbackTagView {cluster_clause};
            DROP VIEW IF EXISTS FloatMetricFeedbackTagView {cluster_clause};
            /* Drop the table */\
            DROP TABLE IF EXISTS FeedbackTag {cluster_clause};
            /* Drop the columns */\
            ALTER TABLE BooleanMetricFeedback {cluster_clause} DROP COLUMN tags;
            ALTER TABLE CommentFeedback {cluster_clause} DROP COLUMN tags;
            ALTER TABLE DemonstrationFeedback {cluster_clause} DROP COLUMN tags;
            ALTER TABLE FloatMetricFeedback {cluster_clause} DROP COLUMN tags;"
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
