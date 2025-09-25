use async_trait::async_trait;

use super::{check_index_exists, check_table_exists};
use crate::db::clickhouse::migration_manager::migration_trait::Migration;
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};

/// This migration adds an index by `inference_id` to the `TagInference`,
/// `ChatInference`, and `JsonInference` tables.
/// This allows us to efficiently query for all inferences by id and/or tag values for a given key for a given inference.
/// It also adds an index by `id` to the `ChatInferenceDatapoint` and `JsonInferenceDatapoint` tables.
pub struct Migration0027<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

#[async_trait]
impl Migration for Migration0027<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        if !check_table_exists(self.clickhouse, "TagInference", "0027").await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0027".to_string(),
                message: "TagInference table does not exist".to_string(),
            }
            .into());
        }
        if !check_table_exists(self.clickhouse, "ChatInference", "0027").await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0027".to_string(),
                message: "ChatInference table does not exist".to_string(),
            }
            .into());
        }
        if !check_table_exists(self.clickhouse, "JsonInference", "0027").await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0027".to_string(),
                message: "JsonInference table does not exist".to_string(),
            }
            .into());
        }
        if !check_table_exists(self.clickhouse, "ChatInferenceDatapoint", "0027").await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0027".to_string(),
                message: "ChatInferenceDatapoint table does not exist".to_string(),
            }
            .into());
        }
        if !check_table_exists(self.clickhouse, "JsonInferenceDatapoint", "0027").await? {
            return Err(ErrorDetails::ClickHouseMigration {
                id: "0027".to_string(),
                message: "JsonInferenceDatapoint table does not exist".to_string(),
            }
            .into());
        }
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        let index_exists =
            check_index_exists(self.clickhouse, "TagInference", "inference_id_index").await?;
        let chat_index_exists =
            check_index_exists(self.clickhouse, "ChatInference", "inference_id_index").await?;
        let json_index_exists =
            check_index_exists(self.clickhouse, "JsonInference", "inference_id_index").await?;
        let chat_datapoint_index_exists =
            check_index_exists(self.clickhouse, "ChatInferenceDatapoint", "id_index").await?;
        let json_datapoint_index_exists =
            check_index_exists(self.clickhouse, "JsonInferenceDatapoint", "id_index").await?;
        Ok(!index_exists
            || !chat_index_exists
            || !json_index_exists
            || !chat_datapoint_index_exists
            || !json_datapoint_index_exists)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        let create_index_query = format!(r"
            ALTER TABLE TagInference{on_cluster_name} ADD INDEX IF NOT EXISTS inference_id_index inference_id TYPE bloom_filter GRANULARITY 1;
        ");
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(create_index_query)
            .await?;

        let materialize_index_query = format!(r"
            ALTER TABLE TagInference{on_cluster_name} MATERIALIZE INDEX inference_id_index;
        ");
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(materialize_index_query)
            .await?;

        let create_index_query = format!(r"
            ALTER TABLE ChatInference{on_cluster_name} ADD INDEX IF NOT EXISTS inference_id_index id TYPE bloom_filter GRANULARITY 1;
        ");
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(create_index_query)
            .await?;

        let materialize_index_query = format!(r"
            ALTER TABLE ChatInference{on_cluster_name} MATERIALIZE INDEX inference_id_index;
        ");
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(materialize_index_query)
            .await?;

        let create_index_query = format!(r"
            ALTER TABLE JsonInference{on_cluster_name} ADD INDEX IF NOT EXISTS inference_id_index id TYPE bloom_filter GRANULARITY 1;
        ");
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(create_index_query)
            .await?;

        let materialize_index_query = format!(r"
            ALTER TABLE JsonInference{on_cluster_name} MATERIALIZE INDEX inference_id_index;
        ");
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(materialize_index_query)
            .await?;

        let create_index_query = format!(r"
            ALTER TABLE ChatInferenceDatapoint{on_cluster_name} ADD INDEX IF NOT EXISTS id_index id TYPE bloom_filter GRANULARITY 1;
        ");
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(create_index_query)
            .await?;

        let materialize_index_query = format!(r"
            ALTER TABLE ChatInferenceDatapoint{on_cluster_name} MATERIALIZE INDEX id_index;
        ");
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(materialize_index_query)
            .await?;

        let create_index_query = format!(r"
            ALTER TABLE JsonInferenceDatapoint{on_cluster_name} ADD INDEX IF NOT EXISTS id_index id TYPE bloom_filter GRANULARITY 1;
        ");
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(create_index_query)
            .await?;

        let materialize_index_query = format!(r"
            ALTER TABLE JsonInferenceDatapoint{on_cluster_name} MATERIALIZE INDEX id_index;
        ");
        let _ = self
            .clickhouse
            .run_query_synchronous_no_params(materialize_index_query)
            .await?;

        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        let on_cluster_name = self.clickhouse.get_on_cluster_name();
        format!(r"
        ALTER TABLE TagInference{on_cluster_name} DROP INDEX IF EXISTS inference_id_index;
        ALTER TABLE ChatInference{on_cluster_name} DROP INDEX IF EXISTS inference_id_index;
        ALTER TABLE JsonInference{on_cluster_name} DROP INDEX IF EXISTS inference_id_index;
        ALTER TABLE ChatInferenceDatapoint{on_cluster_name} DROP INDEX IF EXISTS id_index;
        ALTER TABLE JsonInferenceDatapoint{on_cluster_name} DROP INDEX IF EXISTS id_index;
        ")
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
