use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::{ClickHouseConnectionInfo, GetMaybeReplicatedTableEngineNameArgs};
use crate::error::{Error, ErrorDetails};
use async_trait::async_trait;

use super::{check_table_exists, table_is_nonempty};

/// This migration is used to set up the ClickHouse database for the datasets feature.
/// It creates two tables: `ChatInferenceDatapoint` and `JsonInferenceDatapoint`
/// These tables store the information required to do things like run evaluations,
/// implement dynamic in-context learning, and run curated SFT jobs.
/// We anticipate unpredictable future uses for datasets as well.
///
/// We'll also drop `ChatInferenceDataset` and `JsonInferenceDataset` if they exist and are empty (they should be).
///
/// This migration should subsume migration 0014.
/// They should have been removed from the binary upon merging of this migration.
///
/// This migration differs from 0014 in that it uses DateTime64(6, 'UTC') instead of DateTime('UTC')
pub struct Migration0016<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0016<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    /// Check if the migration needs to be applied
    /// This should be equivalent to checking if `ChatInferenceDatapoint` is missing
    /// or if `JsonInferenceDatapoint` is missing
    /// OR if they exist and either `output` column is not Nullable(String)
    async fn should_apply(&self) -> Result<bool, Error> {
        let chat_inference_dataset_table_exists =
            check_table_exists(self.clickhouse, "ChatInferenceDatapoint", "0016").await?;
        let json_inference_dataset_table_exists =
            check_table_exists(self.clickhouse, "JsonInferenceDatapoint", "0016").await?;
        if !chat_inference_dataset_table_exists || !json_inference_dataset_table_exists {
            return Ok(true);
        }

        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        if check_table_exists(self.clickhouse, "ChatInferenceDataset", "0016").await? {
            let chat_inference_dataset_has_data =
                table_is_nonempty(self.clickhouse, "ChatInferenceDataset", "0016").await?;
            if chat_inference_dataset_has_data {
                return Err(Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0016".to_string(),
                    message: "ChatInferenceDataset has data. Your database state is invalid."
                        .to_string(),
                }));
            }
        }

        if check_table_exists(self.clickhouse, "JsonInferenceDataset", "0016").await? {
            let json_inference_dataset_has_data =
                table_is_nonempty(self.clickhouse, "JsonInferenceDataset", "0016").await?;
            if json_inference_dataset_has_data {
                return Err(Error::new(ErrorDetails::ClickHouseMigration {
                    id: "0016".to_string(),
                    message: "JsonInferenceDataset has data. Your database state is invalid."
                        .to_string(),
                }));
            }
        }

        // First, drop the ChatInferenceDataset and JsonInferenceDataset tables if they were created in 0014
        let query = "DROP TABLE IF EXISTS ChatInferenceDataset";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        let query = "DROP TABLE IF EXISTS JsonInferenceDataset";
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the `ChatInferenceDatapoint` table
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "ReplacingMergeTree",
                table_name: "ChatInferenceDatapoint",
                engine_args: &["updated_at", "is_deleted"],
            },
        );
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS ChatInferenceDatapoint{on_cluster_name}
            (
                dataset_name LowCardinality(String),
                function_name LowCardinality(String),
                id UUID, -- more important to join than to
                        -- sort and sorting expected dataset sizes should be cheap
                        -- If this example is generated from an inference then this
                        -- should be the inference ID
                episode_id Nullable(UUID), -- If this is a synthetic datapoint
                                           -- (not based on an existing inference),
                                           -- then this will be null
                input String,
                output Nullable(String),
                tool_params String,
                -- Don't think we need inference params, processing time,
                -- timestamp, variant_name
                tags Map(String, String),
                auxiliary String, -- a JSON (unstructured, for now)
                is_deleted Bool DEFAULT false,
                updated_at DateTime64(6, 'UTC') DEFAULT now()
            ) ENGINE = {table_engine_name}
            ORDER BY (dataset_name, function_name, id)
        ",
        );
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(query.to_string())
            .await?;

        // Create the `JsonInferenceDatapoint` table
        let table_engine_name = self.clickhouse.get_maybe_replicated_table_engine_name(
            GetMaybeReplicatedTableEngineNameArgs {
                table_engine_name: "ReplacingMergeTree",
                table_name: "JsonInferenceDatapoint",
                engine_args: &["updated_at", "is_deleted"],
            },
        );
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let query = format!(
            r"
            CREATE TABLE IF NOT EXISTS JsonInferenceDatapoint{on_cluster_name}
            (
                dataset_name LowCardinality(String),
                function_name LowCardinality(String),
                id UUID, -- same comment as above
                episode_id Nullable(UUID), -- same comment as above
                input String,
                output Nullable(String),
                output_schema String,
                tags Map(String, String),
                auxiliary String, -- a JSON (unstructured, for now)
                is_deleted Bool DEFAULT false,
                updated_at DateTime64(6, 'UTC') DEFAULT now()
            ) ENGINE = {table_engine_name}
            ORDER BY (dataset_name, function_name, id)
        ",
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
            "/* Drop the tables */\
            DROP TABLE IF EXISTS ChatInferenceDatapoint{on_cluster_name} SYNC;
            DROP TABLE IF EXISTS JsonInferenceDatapoint{on_cluster_name} SYNC;
            "
        )
    }

    /// Check if the migration has succeeded (i.e. it should not be applied again)
    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
