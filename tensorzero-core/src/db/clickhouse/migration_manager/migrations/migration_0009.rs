use std::time::Duration;

use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};

use super::check_table_exists;
use async_trait::async_trait;

/// This migration allows us to efficiently query feedback tables by their target IDs.
pub struct Migration0009<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
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
                    message: format!("Table {table} does not exist"),
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

    async fn apply(&self, clean_start: bool) -> Result<(), Error> {
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
        self.clickhouse.get_create_table_statements(
            "BooleanMetricFeedbackByTargetId",
            r"
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                metric_name LowCardinality(String),
                value Bool,
                tags Map(String, String)
            )",
            &GetMaybeReplicatedTableEngineNameArgs {
                table_name: "BooleanMetricFeedbackByTargetId",
                table_engine_name: "MergeTree",
                engine_args: &[],
            },
            Some("ORDER BY target_id"),
        ).await?;
        // Create the materialized view for the `BooleanMetricFeedbackByTargetId` table from BooleanMetricFeedback
        // If we are not doing a clean start, we need to add a where clause to the view to only include rows that have been created after the view_timestamp
        let view_where_clause = if clean_start {
            None
        } else {
            Some(format!("WHERE UUIDv7ToDateTime(id) >= toDateTime(toUnixTimestamp({view_timestamp}))"))
        };
        
        self.clickhouse
            .get_create_materialized_view_statements(
                "BooleanMetricFeedbackByTargetIdView",
                "BooleanMetricFeedbackByTargetId",
                "BooleanMetricFeedback",
                "id,
                    target_id,
                    metric_name,
                    value,
                    tags",
                view_where_clause.as_deref(),
            )
            .await?;

        // Create the `CommentFeedbackByTargetId` table
        self.clickhouse.get_create_table_statements(
            "CommentFeedbackByTargetId",
            r"
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                target_type Enum('inference' = 1, 'episode' = 2),
                value String,
                tags Map(String, String)
            )",
            &GetMaybeReplicatedTableEngineNameArgs {
                table_name: "CommentFeedbackByTargetId",
                table_engine_name: "MergeTree",
                engine_args: &[],
            },
            Some("ORDER BY target_id"),
        ).await?;

        // Create the materialized view for the `CommentFeedbackByTargetId` table from CommentFeedback
        self.clickhouse
            .get_create_materialized_view_statements(
                "CommentFeedbackByTargetIdView",
                "CommentFeedbackByTargetId",
                "CommentFeedback",
                "id,
                    target_id,
                    target_type,
                    value,
                    tags",
                view_where_clause.as_deref(),
            )
            .await?;

        // Create the `DemonstrationFeedbackByInferenceId` table
        self.clickhouse.get_create_table_statements(
            "DemonstrationFeedbackByInferenceId",
            r"
            (
                id UUID, -- must be a UUIDv7
                target_id UUID, -- must be a UUIDv7
                metric_name LowCardinality(String),
                value Float32,
                tags Map(String, String)
            )",
            &GetMaybeReplicatedTableEngineNameArgs {
                table_name: "DemonstrationFeedbackByInferenceId",
                table_engine_name: "MergeTree",
                engine_args: &[],
            },
            Some("ORDER BY target_id"),
        ).await?;

        // Create the materialized view for the `DemonstrationFeedbackByInferenceId` table from DemonstrationFeedback
        self.clickhouse
            .get_create_materialized_view_statements(
                "DemonstrationFeedbackByInferenceIdView",
                "DemonstrationFeedbackByInferenceId", 
                "DemonstrationFeedback",
                "id,
                    inference_id,
                    value,
                    tags",
                view_where_clause.as_deref(),
            )
            .await?;

        // Create the `FloatMetricFeedbackByTargetId` table
        self.clickhouse.get_create_table_statements(
            "FloatMetricFeedbackByTargetId",
            r"
           (
               id UUID, -- must be a UUIDv7
               target_id UUID, -- must be a UUIDv7
               metric_name LowCardinality(String),
               value Float32,
               tags Map(String, String)
           )",
            &GetMaybeReplicatedTableEngineNameArgs {
                table_name: "FloatMetricFeedbackByTargetId",
                table_engine_name: "MergeTree",
                engine_args: &[],
            },
            Some("ORDER BY target_id"),
        ).await?;

        // Create the materialized view for the `FloatMetricFeedbackByTargetId` table from FloatMetricFeedback
        let view_where_clause_float = if clean_start {
            None
        } else {
            Some(format!("WHERE UUIDv7ToDateTime(id) >= toDateTime(toUnixTimestamp({view_timestamp}))"))
        };
        
        self.clickhouse
            .get_create_materialized_view_statements(
                "FloatMetricFeedbackByTargetIdView",
                "FloatMetricFeedbackByTargetId",
                "FloatMetricFeedback",
                "id,
                    target_id,
                    metric_name,
                    value,
                    tags",
                view_where_clause_float.as_deref(),
            )
            .await?;

        // Insert the data from the original tables into the new table (we do this concurrently since it could theoretically take a long time)
        if !clean_start {
            // Sleep for the duration specified by view_offset to allow the materialized views to catch up
            tokio::time::sleep(view_offset).await;
            let insert_boolean_metric_feedback = async {
                // For INSERT operations, use distributed table names for both source and target
                let source_table = "BooleanMetricFeedback"; // Use distributed table for SELECT 
                let target_table = "BooleanMetricFeedbackByTargetId"; // Use distributed table for INSERT
                
                let query = format!(
                    r"
                    INSERT INTO {target_table}
                    SELECT
                        id,
                        target_id,
                        metric_name,
                        value,
                        tags
                    FROM {source_table}
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "
                );
                self.clickhouse.run_query_synchronous_no_params(query).await
            };

            let insert_comment_feedback = async {
                // For INSERT operations, use distributed table names for both source and target
                let source_table = "CommentFeedback"; // Use distributed table for SELECT
                let target_table = "CommentFeedbackByTargetId"; // Use distributed table for INSERT
                
                let query = format!(
                    r"
                    INSERT INTO {target_table}
                    SELECT
                        id,
                        target_id,
                        target_type,
                        value,
                        tags
                    FROM {source_table}
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "
                );
                self.clickhouse.run_query_synchronous_no_params(query).await
            };

            let insert_demonstration_feedback = async {
                // For INSERT operations, use distributed table names for both source and target
                let source_table = "DemonstrationFeedback"; // Use distributed table for SELECT
                let target_table = "DemonstrationFeedbackByInferenceId"; // Use distributed table for INSERT
                
                let query = format!(
                    r"
                    INSERT INTO {target_table}
                    SELECT
                        id,
                        inference_id,
                        value,
                        tags
                    FROM {source_table}
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "
                );
                self.clickhouse.run_query_synchronous_no_params(query).await
            };

            let insert_float_metric_feedback = async {
                // For INSERT operations, use distributed table names for both source and target
                let source_table = "FloatMetricFeedback"; // Use distributed table for SELECT
                let target_table = "FloatMetricFeedbackByTargetId"; // Use distributed table for INSERT
                
                let query = format!(
                    r"
                    INSERT INTO {target_table}
                    SELECT
                        id,
                        target_id,
                        metric_name,
                        value,
                        tags
                    FROM {source_table}
                    WHERE UUIDv7ToDateTime(id) < toDateTime(toUnixTimestamp({view_timestamp}));
                "
                );
                self.clickhouse.run_query_synchronous_no_params(query).await
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
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        
        format!(
            "/* Drop the materialized views */\
            DROP VIEW IF EXISTS BooleanMetricFeedbackByTargetIdView{on_cluster_name};\
            DROP VIEW IF EXISTS CommentFeedbackByTargetIdView{on_cluster_name};\
            DROP VIEW IF EXISTS DemonstrationFeedbackByInferenceIdView{on_cluster_name};\
            DROP VIEW IF EXISTS FloatMetricFeedbackByTargetIdView{on_cluster_name};\
            /* Drop the tables */\
            {}\
            {}\
            {}\
            {}",
            self.clickhouse.get_drop_table_rollback_statements("BooleanMetricFeedbackByTargetId"),
            self.clickhouse.get_drop_table_rollback_statements("CommentFeedbackByTargetId"),
            self.clickhouse.get_drop_table_rollback_statements("DemonstrationFeedbackByInferenceId"),
            self.clickhouse.get_drop_table_rollback_statements("FloatMetricFeedbackByTargetId")
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
