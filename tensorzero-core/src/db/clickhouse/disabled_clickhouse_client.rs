use async_trait::async_trait;
use secrecy::SecretString;
use std::collections::HashMap;

use crate::config::Config;
use crate::error::Error;
use crate::stored_inference::StoredInference;

use super::batching::BatchWriterHandle;
use super::query_builder::ListInferencesParams;
use super::{
    ClickHouseClient, ClickHouseResponse, ClickHouseResponseMetadata, ExternalDataInfo,
    GetMaybeReplicatedTableEngineNameArgs, HealthCheckable, TableName,
};


/// Disabled implementation of ClickHouseClient (no-op)
/// 
/// This is used in a few cases in production when we don't want to write to ClickHouse.
#[derive(Debug, Clone, Copy)]
struct DisabledClickHouseClient;

#[async_trait]
impl ClickHouseClient for DisabledClickHouseClient {
    fn database_url(&self) -> &SecretString {
        &SecretString::new("disabled".to_string())
    }

    fn cluster_name(&self) -> &Option<String> {
        &None
    }

    fn database(&self) -> &str {
        "disabled"
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

    async fn run_query_synchronous_with_err_logging(
        &self,
        _query: String,
        _parameters: &HashMap<&str, &str>,
        _err_logging: bool,
    ) -> Result<ClickHouseResponse, Error> {
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

    async fn list_inferences(
        &self,
        _config: &Config,
        _opts: &ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInference>, Error> {
        Ok(Vec::new())
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

    fn variant_name(&self) -> &'static str {
        "Disabled"
    }
}

#[async_trait]
impl HealthCheckable for DisabledClickHouseClient {
    async fn health(&self) -> Result<(), Error> {
        Ok(())
    }
}