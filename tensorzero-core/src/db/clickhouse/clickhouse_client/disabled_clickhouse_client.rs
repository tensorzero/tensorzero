use async_trait::async_trait;
use lazy_static::lazy_static;
use secrecy::SecretString;
use std::collections::HashMap;

use crate::db::clickhouse::batching::BatchWriterHandle;
use crate::db::clickhouse::clickhouse_client::ClickHouseClientType;
use crate::db::clickhouse::{
    ClickHouseClient, ClickHouseResponse, ClickHouseResponseMetadata, ExternalDataInfo,
    GetMaybeReplicatedTableEngineNameArgs, HealthCheckable, TableName,
};
use crate::error::{DelayedError, Error};

lazy_static! {
    static ref DISABLED_DATABASE_URL: SecretString = SecretString::from("disabled");
    static ref DISABLED_CLUSTER_NAME: Option<String> = None;
}

/// Disabled implementation of ClickHouseClient (no-op)
///
/// This is used in a few cases in production when we don't want to write to ClickHouse.
#[derive(Debug, Clone, Copy)]
pub struct DisabledClickHouseClient;

#[async_trait]
impl ClickHouseClient for DisabledClickHouseClient {
    fn database_url(&self) -> &SecretString {
        &DISABLED_DATABASE_URL
    }

    fn cluster_name(&self) -> &Option<String> {
        &DISABLED_CLUSTER_NAME
    }

    fn database(&self) -> &str {
        "disabled"
    }

    fn is_batching_enabled(&self) -> bool {
        false
    }

    fn batcher_join_handle(&self) -> Option<BatchWriterHandle> {
        None
    }

    async fn write_batched_internal(
        &self,
        _rows: Vec<String>,
        _table: TableName,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn write_non_batched_internal(
        &self,
        _rows: Vec<String>,
        _table: TableName,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn run_query_synchronous(
        &self,
        _query: String,
        _parameters: &HashMap<&str, &str>,
    ) -> Result<ClickHouseResponse, Error> {
        Ok(ClickHouseResponse {
            response: String::new(),
            metadata: ClickHouseResponseMetadata {
                read_rows: 0,
                written_rows: 0,
            },
        })
    }

    async fn run_query_synchronous_delayed_err(
        &self,
        _query: String,
        _parameters: &HashMap<&str, &str>,
    ) -> Result<ClickHouseResponse, DelayedError> {
        Ok(ClickHouseResponse {
            response: String::new(),
            metadata: ClickHouseResponseMetadata {
                read_rows: 0,
                written_rows: 0,
            },
        })
    }

    async fn run_query_with_external_data(
        &self,
        _external_data: ExternalDataInfo,
        _query: String,
    ) -> Result<ClickHouseResponse, Error> {
        Ok(ClickHouseResponse {
            response: String::new(),
            metadata: ClickHouseResponseMetadata {
                read_rows: 0,
                written_rows: 0,
            },
        })
    }

    async fn check_database_and_migrations_table_exists(&self) -> Result<bool, Error> {
        Ok(true)
    }

    async fn create_database_and_migrations_table(&self) -> Result<(), Error> {
        Ok(())
    }

    fn is_cluster_configured(&self) -> bool {
        false
    }

    fn get_on_cluster_name(&self) -> String {
        String::new()
    }

    fn get_maybe_replicated_table_engine_name(
        &self,
        args: GetMaybeReplicatedTableEngineNameArgs<'_>,
    ) -> String {
        args.table_engine_name.to_string()
    }

    fn client_type(&self) -> ClickHouseClientType {
        ClickHouseClientType::Disabled
    }
}

#[async_trait]
impl HealthCheckable for DisabledClickHouseClient {
    async fn health(&self) -> Result<(), Error> {
        Ok(())
    }
}
