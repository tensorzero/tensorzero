use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

use super::check_table_exists;
use async_trait::async_trait;

/// This migration is used to set up the ClickHouse database for the datasets feature.
/// It creates two tables: `ChatInferenceDataset` and `JsonInferenceDataset`
/// These tables store the information required to do things like run evaluations,
/// implement dynamic in-context learning, and run curated SFT jobs.
/// We anticipate unpredictable future uses for datasets as well.
pub struct Migration0012<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0012<'_> {
    /// Check if you can connect to the database
    async fn can_apply(&self) -> Result<(), Error> {
        self.clickhouse.health().await.map_err(|e| {
            Error::new(ErrorDetails::ClickHouseMigration {
                id: "0012".to_string(),
                message: e.to_string(),
            })
        })?;

        Ok(())
    }

    /// Check if the migration needs to be applied
    /// This should be equivalent to checking if `ChatInferenceDataset` is missing
    /// or if `JsonInferenceDataset` is missing
    async fn should_apply(&self) -> Result<bool, Error> {
        let chat_inference_dataset_table_exists =
            check_table_exists(self.clickhouse, "ChatInferenceDataset", "0012").await?;
        let json_inference_dataset_table_exists =
            check_table_exists(self.clickhouse, "JsonInferenceDataset", "0012").await?;
        Ok(!chat_inference_dataset_table_exists || !json_inference_dataset_table_exists)
    }

    async fn apply(&self) -> Result<(), Error> {
        // Create the `ChatInferenceDataset` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS ChatInferenceDataset
            (
                dataset_name LowCardinality(String),
                function_name LowCardinality(String),
                id UUID, -- more important to join than to
                        -- sort and sorting expected dataset sizes should be cheap
                        -- If this example is generated from an inference then this
                        -- should be the inference ID
                episode_id UUID,
                input String,
                output String,
                tool_params String,
                -- Don't think we need inference params, processing time,
                -- timestamp, variant_name
                tags Map(String, String),
                auxiliary String, -- a JSON (unstructured, for now)
                is_deleted Bool DEFAULT false,
                created_at DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(created_at, is_deleted)
            ORDER BY (dataset_name, function_name, id)
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        // Create the `JsonInferenceDataset` table
        let query = r#"
            CREATE TABLE IF NOT EXISTS JsonInferenceDataset
            (
                dataset_name LowCardinality(String),
                function_name LowCardinality(String),
                id UUID, -- same comment as above
                episode_id UUID,
                input String,
                output String,
                output_schema String,
                tags Map(String, String),
                auxiliary String, -- a JSON (unstructured, for now)
                is_deleted Bool DEFAULT false,
                created_at DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(created_at, is_deleted)
            ORDER BY (dataset_name, function_name, id)
        "#;
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        "\
            -- Drop the tables\n\
            DROP TABLE IF EXISTS ChatInferenceDataset;\n\
            DROP TABLE IF EXISTS JsonInferenceDataset;\n\
            "
        .to_string()
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
