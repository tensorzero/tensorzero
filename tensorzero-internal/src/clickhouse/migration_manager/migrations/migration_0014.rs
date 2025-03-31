use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

use super::{check_table_exists, get_column_type, table_is_nonempty};

/// This migration is used to set up the ClickHouse database for the datasets feature.
/// It creates two tables: `ChatInferenceDataset` and `JsonInferenceDataset`
/// These tables store the information required to do things like run evaluations,
/// implement dynamic in-context learning, and run curated SFT jobs.
/// We anticipate unpredictable future uses for datasets as well.
///
/// This migration should subsume migration 0012.
/// They should have been removed from the binary upon merging of this migration.
///
/// There were two changes made form 0014: changing created_at to updated_at and
/// changing the output column to be Nullable(String) instead of String.
pub struct Migration0014<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0014<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    /// Check if the migration needs to be applied
    /// This should be equivalent to checking if `ChatInferenceDataset` is missing
    /// or if `JsonInferenceDataset` is missing
    /// OR if they exist and either `output` column is not Nullable(String)
    async fn should_apply(&self) -> Result<bool, Error> {
        let chat_inference_dataset_table_exists =
            check_table_exists(self.clickhouse, "ChatInferenceDataset", "0014").await?;
        let json_inference_dataset_table_exists =
            check_table_exists(self.clickhouse, "JsonInferenceDataset", "0014").await?;
        if !chat_inference_dataset_table_exists || !json_inference_dataset_table_exists {
            return Ok(true);
        }

        let chat_inference_dataset_output_type =
            get_column_type(self.clickhouse, "ChatInferenceDataset", "output", "0014").await?;
        let json_inference_dataset_output_type =
            get_column_type(self.clickhouse, "JsonInferenceDataset", "output", "0014").await?;
        if chat_inference_dataset_output_type != "Nullable(String)"
            || json_inference_dataset_output_type != "Nullable(String)"
        {
            return Ok(true);
        }

        Ok(false)
    }

    async fn apply(&self) -> Result<(), Error> {
        if check_table_exists(self.clickhouse, "ChatInferenceDataset", "0014").await? {
            let chat_inference_dataset_has_data =
                table_is_nonempty(self.clickhouse, "ChatInferenceDataset", "0014").await?;
            if chat_inference_dataset_has_data {
                return Err(Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0014".to_string(),
                    message: "ChatInferenceDataset has data. Your database state is invalid."
                        .to_string(),
                }));
            }
        }

        if check_table_exists(self.clickhouse, "JsonInferenceDataset", "0014").await? {
            let json_inference_dataset_has_data =
                table_is_nonempty(self.clickhouse, "JsonInferenceDataset", "0014").await?;
            if json_inference_dataset_has_data {
                return Err(Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0014".to_string(),
                    message: "JsonInferenceDataset has data. Your database state is invalid."
                        .to_string(),
                }));
            }
        }

        // First, drop the tables if they were created in 0012
        let query = "DROP TABLE IF EXISTS ChatInferenceDataset";
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

        let query = "DROP TABLE IF EXISTS JsonInferenceDataset";
        let _ = self.clickhouse.run_query(query.to_string(), None).await?;

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
                output Nullable(String),
                tool_params String,
                -- Don't think we need inference params, processing time,
                -- timestamp, variant_name
                tags Map(String, String),
                auxiliary String, -- a JSON (unstructured, for now)
                is_deleted Bool DEFAULT false,
                updated_at DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
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
                output Nullable(String),
                output_schema String,
                tags Map(String, String),
                auxiliary String, -- a JSON (unstructured, for now)
                is_deleted Bool DEFAULT false,
                updated_at DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(updated_at, is_deleted)
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
