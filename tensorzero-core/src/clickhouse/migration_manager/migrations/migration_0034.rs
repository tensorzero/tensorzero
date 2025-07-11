use super::check_table_exists;
use crate::clickhouse::migration_manager::migration_trait::Migration;
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::Error;
use async_trait::async_trait;


/// This migration adds a `TensorZeroError` table and associated materialized views
pub struct Migration0034<'a> {
    pub clickhouse: &'a ClickHouseConnectionInfo,
}

const MIGRATION_ID: &str = "0034";

#[async_trait]
impl Migration for Migration0034<'_> {
    async fn can_apply(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn should_apply(&self) -> Result<bool, Error> {
        // If the table doesn't exist, we need to create it
        if !check_table_exists(self.clickhouse, "TensorZeroError", MIGRATION_ID).await? {
            return Ok(true);
        }
        Ok(false)
    }

    async fn apply(&self, _clean_start: bool) -> Result<(), Error> {
        self.clickhouse
            .run_query_synchronous_no_params(
                r#"CREATE TABLE IF NOT EXISTS TensorZeroError (
                    id_uint UInt128,
                    error_kind LowCardinality(String),
                    message String,
                    http_status Nullable(UInt16),
                    raw_request Nullable(String),
                    raw_response Nullable(String),
                    function_name LowCardinality(Nullable(String)),
                    variant_name LowCardinality(Nullable(String)),
                    model_name LowCardinality(Nullable(String)),
                    model_provider_name LowCardinality(Nullable(String)),
                    episode_id Nullable(UUID),
                    inference_id Nullable(UUID),
                )
                ENGINE = ReplacingMergeTree(id_uint)
                ORDER BY id_uint
                PARTITION BY toStartOfMonth(UUIDv7ToDateTime(uint_to_uuid(id_uint)))
                "#
                .to_string(),
            )
            .await?;

        self.clickhouse
            .run_query_synchronous_no_params(
                r#"CREATE TABLE IF NOT EXISTS TensorZeroErrorByFunction (
                    function_name LowCardinality(String),
                    variant_name LowCardinality(Nullable(String)),
                    id_uint UInt128,
                    error_kind LowCardinality(String),
                    http_status Nullable(UInt16),
                )
                ENGINE = ReplacingMergeTree(id_uint)
                ORDER BY (function_name, variant_name)
                PARTITION BY toStartOfMonth(UUIDv7ToDateTime(uint_to_uuid(id_uint)))
                SETTINGS allow_nullable_key = 1
                "#
                .to_string(),
            )
            .await?;

        self.clickhouse
            .run_query_synchronous_no_params(
                r#"CREATE MATERIALIZED VIEW IF NOT EXISTS TensorZeroErrorByFunctionView
                TO TensorZeroErrorByFunction
                AS
                SELECT function_name, variant_name, id_uint, error_kind, http_status FROM TensorZeroError
                WHERE function_name IS NOT NULL
                ORDER BY (function_name, variant_name)
                "#
                .to_string(),
            )
            .await?;

        self.clickhouse
            .run_query_synchronous_no_params(
                r#"CREATE TABLE IF NOT EXISTS TensorZeroErrorByModel (
                    model_name LowCardinality(String),
                    model_provider_name LowCardinality(Nullable(String)),
                    id_uint UInt128,
                    error_kind LowCardinality(String),
                    http_status Nullable(UInt16),
                )
                ENGINE = ReplacingMergeTree(id_uint)
                ORDER BY (model_name, model_provider_name)
                PARTITION BY toStartOfMonth(UUIDv7ToDateTime(uint_to_uuid(id_uint)))
                SETTINGS allow_nullable_key = 1
                "#
                .to_string(),
            )
            .await?;

        self.clickhouse
            .run_query_synchronous_no_params(
                r#"CREATE MATERIALIZED VIEW IF NOT EXISTS TensorZeroErrorByModelView
                TO TensorZeroErrorByModel
                AS
                SELECT model_name, model_provider_name, id_uint, error_kind, http_status FROM TensorZeroError
                WHERE model_name IS NOT NULL
                ORDER BY (model_name, model_provider_name)
                "#
                .to_string(),
            )
            .await?;
        Ok(())
    }

    fn rollback_instructions(&self) -> String {
        r#"DROP TABLE TensorZeroErrorByFunction;
           DROP TABLE TensorZeroErrorByModel;
           DROP TABLE TensorZeroError;
        "#
        .to_string()
    }

    async fn has_succeeded(&self) -> Result<bool, Error> {
        let should_apply = self.should_apply().await?;
        Ok(!should_apply)
    }
}
