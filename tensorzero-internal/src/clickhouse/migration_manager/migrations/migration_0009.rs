use std::time::Duration;

use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

use super::check_table_exists;
use async_trait::async_trait;

/// This migration allows us to efficiently query feedback tables by their target IDs.
pub struct Migration0009<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
    pub clean_start: bool,
}

#[async_trait]
impl Migration for Migration0009<'_> {
    /// Check if the four feedback tables exist
    /// If all of this is OK, then we can apply the migration
    async fn can_apply(&self) -> Result<(), Error> {
        let tables = vec![
            "BooleanMetricFeedback",
            "CommentFeedback",
            "DemonstrationFeedback",
            "FloatMetricFeedback",
        ];

        for table in tables {
            if !check_table_exists(self.clickhouse, table, "0009").await? {
                return Err(ErrorDetails::ClickHouseMigration {
                    id: "0009".to_string(),
                    message: format!("Table {} does not exist", table),
                }
                .into());
            }
        }

        Ok(())
    }

    /// Check if the migration has already been applied
    /// This should be equivalent to checking if `BooleanMetricFeedbackByTargetId`,
    /// `CommentFeedbackByTargetId`, `DemonstrationFeedbackByTargetId` and
    /// `FloatMetricFeedbackByTargetId` exist
    async fn should_apply(&self) -> Result<bool, Error> {
        let tables = vec![
            "BooleanMetricFeedbackByTargetId",
            "CommentFeedbackByTargetId",
            "DemonstrationFeedbackByInferenceId",
            "FloatMetricFeedbackByTargetId",
        ];
        for table in tables {
            if !check_table_exists(self.clickhouse, table, "0009").await? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Only gets used when we are not doing a clean start
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

        // Create the `BooleanMetricFeedbackByTargetId` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS BooleanMetricFeedbackByTargetId
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                metric_name LowCardinality(String),
                value Bool,
                tags Map(String, String)
            ) ENGINE = MergeTree()
            ORDER BY target_id;
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;
        // Create the materialized view for the `BooleanMetricFeedbackByTargetId` table from BooleanMetricFeedback
        // If we are not doing a clean start, we need to add a where clause to the view to only include rows that have been created after the view_timestamp
        let view_where_clause = if !self.clean_start {
            format!("WHERE UUIDv7ToDateTime(id) >= toDateTime(toUnixTimestamp({view_timestamp}))")
        } else {
            String::new()
        };
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS BooleanMetricFeedbackByTargetIdView
            TO BooleanMetricFeedbackByTargetId
            AS
                SELECT
                    id,
                    target_id,
                    metric_name,
                    value,
                    tags
                FROM BooleanMetricFeedback
                {view_where_clause};
            "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query, None).await?;

        // Create the `CommentFeedbackByTargetId` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS CommentFeedbackByTargetId
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                target_type Enum('inference' = 1, 'episode' = 2),
                value String,
                tags Map(String, String)
            ) ENGINE = MergeTree()
            ORDER BY target_id;
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the materialized view for the `CommentFeedbackByTargetId` table from CommentFeedback
        // If we are not doing a clean start, we need to add a where clause to the view to only include rows that have been created after the view_timestamp
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS CommentFeedbackByTargetIdView
            TO CommentFeedbackByTargetId
            AS
                SELECT
                    id,
                    target_id,
                    target_type,
                    value,
                    tags
                FROM CommentFeedback
                {view_where_clause};
            "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query, None).await?;

        // Create the `DemonstrationFeedbackByInferenceId` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS DemonstrationFeedbackByInferenceId
            (
                id UUID, -- must be a UUIDv7
                inference_id UUID, -- must be a UUIDv7
                value String,
                tags Map(String, String)
            ) ENGINE = MergeTree()
            ORDER BY inference_id;
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the materialized view for the `DemonstrationFeedbackByInferenceId` table from DemonstrationFeedback
        // If we are not doing a clean start, we need to add a where clause to the view to only include rows that have been created after the view_timestamp
        let query = format!(
            r#"
            CREATE MATERIALIZED VIEW IF NOT EXISTS DemonstrationFeedbackByInferenceIdView
            TO DemonstrationFeedbackByInferenceId
            AS
                SELECT
                    id,
                    inference_id,
                    value,
                    tags
                FROM DemonstrationFeedback
                {view_where_clause};
            "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query, None).await?;

        // Create the `FloatMetricFeedbackByTargetId` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS FloatMetricFeedbackByTargetId
           (
               id UUID, -- must be a UUIDv7
               target_id UUID, -- must be a UUIDv7
               metric_name LowCardinality(String),
               value Float32,
               tags Map(String, String)
           ) ENGINE = MergeTree()
           ORDER BY target_id;
       "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the materialized view for the `FloatMetricFeedbackByTargetId` table from FloatMetricFeedback
        // If we are not doing a clean start, we need to add a where clause to the view to only include rows that have been created after the view_timestamp
        let view_where_clause = if !self.clean_start {
            format!("WHERE UUIDv7ToDateTime(id) >= toDateTime(toUnixTimestamp({view_timestamp}))")
        } else {
            String::new()
        };
        let query = format!(
            r#"
           CREATE MATERIALIZED VIEW IF NOT EXISTS FloatMetricFeedbackByTargetIdView
           TO FloatMetricFeedbackByTargetId
           AS
               SELECT
                   id,
                   target_id,
                   metric_name,
                   value,
                   tags
               FROM FloatMetricFeedback
               {view_where_clause};
           "#,
            view_where_clause = view_where_clause
        );
        let _ = self.clickhouse.run_query(query, None).await?;

        // Insert the data from the original tables into the new table (we do this concurrently since it could theoretically take a long time)
        if !self.clean_start {
            // Sleep for the duration specified by view_offset to allow the materialized views to catch up
            tokio::time::sleep(view_offset).await;
            let insert_boolean_metric_feedback = async {
                let query = format!(
                    r#"
                    INSERT INTO BooleanMetricFeedbackByTargetId
                    SELECT
                        id,
                        target_id,
                        metric_name,
                        value,
                        tags
                    FROM BooleanMetricFeedback
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "#,
                    view_timestamp = view_timestamp
                );
                self.clickhouse.run_query(query, None).await
            };

            let insert_comment_feedback = async {
                let query = format!(
                    r#"
                    INSERT INTO CommentFeedbackByTargetId
                    SELECT
                        id,
                        target_id,
                        target_type,
                        value,
                        tags
                    FROM CommentFeedback
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "#,
                    view_timestamp = view_timestamp
                );
                self.clickhouse.run_query(query, None).await
            };

            let insert_demonstration_feedback = async {
                let query = format!(
                    r#"
                    INSERT INTO DemonstrationFeedbackByInferenceId
                    SELECT
                        id,
                        inference_id,
                        value,
                        tags
                    FROM DemonstrationFeedback
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "#,
                    view_timestamp = view_timestamp
                );
                self.clickhouse.run_query(query, None).await
            };

            let insert_float_metric_feedback = async {
                let query = format!(
                    r#"
                    INSERT INTO FloatMetricFeedbackByTargetId
                    SELECT
                        id,
                        target_id,
                        metric_name,
                        value,
                        tags
                    FROM FloatMetricFeedback
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "#,
                    view_timestamp = view_timestamp
                );
                self.clickhouse.run_query(query, None).await
            };

            tokio::try_join!(
                insert_boolean_metric_feedback,
                insert_comment_feedback,
                insert_demonstration_feedback,
                insert_float_metric_feedback
            )?;
        }

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
            -- Drop the materialized views\n\
            DROP VIEW IF EXISTS BooleanMetricFeedbackByTargetIdView;\n\
            DROP VIEW IF EXISTS CommentFeedbackByTargetIdView;\n\
            DROP VIEW IF EXISTS DemonstrationFeedbackByInferenceIdView;\n\
            DROP VIEW IF EXISTS FloatMetricFeedbackByTargetIdView;\n\
            \n\
            -- Drop the tables\n\
            DROP TABLE IF EXISTS BooleanMetricFeedbackByTargetId;\n\
            DROP TABLE IF EXISTS CommentFeedbackByTargetId;\n\
            DROP TABLE IF EXISTS DemonstrationFeedbackByInferenceId;\n\
            DROP TABLE IF EXISTS FloatMetricFeedbackByTargetId;\n\
            "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
