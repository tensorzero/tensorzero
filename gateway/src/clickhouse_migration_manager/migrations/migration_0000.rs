use crate::clickhouse::ClickHouseConnectionInfo;
use crate::clickhouse_migration_manager::migration_trait::Migration;
use crate::error::Error;

/// This migration is used to create the initial tables in the ClickHouse database.
///
/// It is used to create the following tables:
/// - BooleanMetricFeedback
/// - CommentFeedback
/// - DemonstrationFeedback
/// - FloatMetricFeedback
/// - Inference
/// - ModelInference

pub struct Migration0000<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

impl<'a> Migration for Migration0000<'a> {
    /// Check if you can connect to the database
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse
            .health()
            .await
            .map_err(|e| Error::ClickHouseMigration {
                id: "0000".to_string(),
                message: e.to_string(),
            })
    }

    /// Check if the tables exist
    async fn should_apply(&self) -> Result<bool, Error> {
        let tables = vec![
            "BooleanMetricFeedback",
            "CommentFeedback",
            "DemonstrationFeedback",
            "FloatMetricFeedback",
            "Inference",
            "ModelInference",
        ];

        for table in tables {
            let query = format!(
                r#"SELECT EXISTS(
                    SELECT 1
                    FROM system.tables
                    WHERE database = 'tensorzero' AND name = '{table}'
                )"#
            );

            match self.clickhouse.run_query(query).await {
                // If `can_apply` succeeds but this fails, it likely means the database does not exist
                Err(_) => return Ok(true),
                Ok(response) => {
                    if response.trim() != "1" {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    async fn apply(&self) -> Result<(), Error> {
        // TODO (#69): parametrize the database name
        // Create the database if it doesn't exist
        self.clickhouse.create_database("tensorzero").await?;

        // Create the `BooleanMetricFeedback` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS BooleanMetricFeedback
            (
                id UUID,
                target_id UUID,
                metric_name LowCardinality(String),
                value Bool
            ) ENGINE = MergeTree()
            ORDER BY (metric_name, target_id);
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Create the `BooleanMetricFeedback` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS BooleanMetricFeedback
            (
                id UUID,
                target_id UUID,
                metric_name LowCardinality(String),
                value Bool
            ) ENGINE = MergeTree()
            ORDER BY (metric_name, target_id);
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Create the `CommentFeedback` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS CommentFeedback
            (
                id UUID,
                target_id UUID,
                target_type Enum('inference' = 1, 'episode' = 2),
                value String
            ) ENGINE = MergeTree()
            ORDER BY target_id;
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Create the `DemonstrationFeedback` table
        let query = r#"
           CREATE TABLE IF NOT EXISTS DemonstrationFeedback
            (
                id UUID,
                inference_id UUID,
                value String
            ) ENGINE = MergeTree()
            ORDER BY inference_id;
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Create the `FloatMetricFeedback` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS FloatMetricFeedback
            (
                id UUID,
                target_id UUID,
                metric_name LowCardinality(String),
                value Float32
            ) ENGINE = MergeTree()
            ORDER BY (metric_name, target_id);
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Create the `Inference` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS Inference
            (
                id UUID,
                function_name LowCardinality(String),
                variant_name LowCardinality(String),
                episode_id UUID,
                input String,
                output Nullable(String),
                processing_time_ms UInt32,
                -- This is whatever string we got from the Inference, without output sanitization
                raw_output String,
            ) ENGINE = MergeTree()
            ORDER BY (function_name, variant_name, episode_id);
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        // Create the `ModelInference` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS ModelInference
            (
                id UUID,
                inference_id UUID,
                input String,
                output String,
                raw_response String,
                input_tokens UInt32,
                output_tokens UInt32,
                response_time_ms UInt32,
                ttft_ms Nullable(UInt32),
            ) ENGINE = MergeTree()
            ORDER BY inference_id;
        "#;
        let _ = self.clickhouse.run_query(query.to_string()).await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
        **CAREFUL: THIS WILL DELETE ALL DATA**\n\
        \n\
        -- Drop the database\n\
        DROP DATABASE IF EXISTS tensorzero;\n\
        \n\
        **CAREFUL: THIS WILL DELETE ALL DATA**\n\
        "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        Ok(!self.should_apply().await?)
    }
}
