use async_trait::async_trait;

use super::{check_index_exists, check_table_exists};
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

const MIGRATION_ID: &str = "0044";

/// This migration adds a bloom filter index on `episode_id` for `ChatInference` and `JsonInference` tables.
/// It also re-applies the indexes from migration_0027 with ON CLUSTER support to fix cluster deployments.
///
/// The indexes added:
/// - `episode_id_index` on `ChatInference.episode_id` (new)
/// - `episode_id_index` on `JsonInference.episode_id` (new)
/// - Re-applies `inference_id_index` on `TagInference`, `ChatInference`, `JsonInference` with ON CLUSTER
/// - Re-applies `id_index` on `ChatInferenceDatapoint`, `JsonInferenceDatapoint` with ON CLUSTER
pub struct Migration0044<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0044<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        if !check_table_exists(self.clickhouse, "ChatInference", MIGRATION_ID).await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "ChatInference table does not exist".to_string(),
            }
            .into());
        }
        if !check_table_exists(self.clickhouse, "JsonInference", MIGRATION_ID).await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "JsonInference table does not exist".to_string(),
            }
            .into());
        }
        if !check_table_exists(self.clickhouse, "TagInference", MIGRATION_ID).await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "TagInference table does not exist".to_string(),
            }
            .into());
        }
        if !check_table_exists(self.clickhouse, "ChatInferenceDatapoint", MIGRATION_ID).await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "ChatInferenceDatapoint table does not exist".to_string(),
            }
            .into());
        }
        if !check_table_exists(self.clickhouse, "JsonInferenceDatapoint", MIGRATION_ID).await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: MIGRATION_ID.to_string(),
                message: "JsonInferenceDatapoint table does not exist".to_string(),
            }
            .into());
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // Check if the new episode_id indexes exist
        let chat_episode_index_exists =
            check_index_exists(self.clickhouse, "ChatInference", "episode_id_index").await?;
        let json_episode_index_exists =
            check_index_exists(self.clickhouse, "JsonInference", "episode_id_index").await?;

        // If either episode_id index is missing, we should apply
        Ok(!chat_episode_index_exists || !json_episode_index_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();

        // Part 1: Re-apply existing indexes from migration_0027 with ON CLUSTER
        // These are idempotent due to IF NOT EXISTS
        let query = format!(
            r"ALTER TABLE TagInference{on_cluster_name} ADD INDEX IF NOT EXISTS inference_id_index inference_id TYPE bloom_filter GRANULARITY 1"
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        let query = format!(
            r"ALTER TABLE ChatInference{on_cluster_name} ADD INDEX IF NOT EXISTS inference_id_index id TYPE bloom_filter GRANULARITY 1"
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        let query = format!(
            r"ALTER TABLE JsonInference{on_cluster_name} ADD INDEX IF NOT EXISTS inference_id_index id TYPE bloom_filter GRANULARITY 1"
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        let query = format!(
            r"ALTER TABLE ChatInferenceDatapoint{on_cluster_name} ADD INDEX IF NOT EXISTS id_index id TYPE bloom_filter GRANULARITY 1"
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        let query = format!(
            r"ALTER TABLE JsonInferenceDatapoint{on_cluster_name} ADD INDEX IF NOT EXISTS id_index id TYPE bloom_filter GRANULARITY 1"
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Part 2: Add new episode_id indexes
        let query = format!(
            r"ALTER TABLE ChatInference{on_cluster_name} ADD INDEX IF NOT EXISTS episode_id_index episode_id TYPE bloom_filter GRANULARITY 1"
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Materialize the index for existing data
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE ChatInference MATERIALIZE INDEX episode_id_index".to_string(),
            )
            .await?;

        let query = format!(
            r"ALTER TABLE JsonInference{on_cluster_name} ADD INDEX IF NOT EXISTS episode_id_index episode_id TYPE bloom_filter GRANULARITY 1"
        );
        self.clickhouse
            .run_query_synchronous_no_params(query)
            .await?;

        // Materialize the index for existing data
        self.clickhouse
            .run_query_synchronous_no_params(
                r"ALTER TABLE JsonInference MATERIALIZE INDEX episode_id_index".to_string(),
            )
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        r"ALTER TABLE ChatInference DROP INDEX IF EXISTS episode_id_index;
ALTER TABLE JsonInference DROP INDEX IF EXISTS episode_id_index;"
            .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        Ok(!self.should_apply().await?)
    }
}
