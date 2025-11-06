use async_trait::async_trait;
use secrecy::SecretString;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::db::clickhouse::BatchWriterHandle;
use crate::db::clickhouse::ClickHouseResponse;
use crate::db::clickhouse::ExternalDataInfo;
use crate::db::clickhouse::GetMaybeReplicatedTableEngineNameArgs;
use crate::db::clickhouse::TableName;
use crate::db::HealthCheckable;
use crate::error::{DelayedError, Error};

#[cfg(test)]
use mockall::mock;

pub(crate) use disabled_clickhouse_client::DisabledClickHouseClient;
#[cfg(any(test, feature = "pyo3"))]
pub(crate) use fake_clickhouse_client::FakeClickHouseClient;
pub use production_clickhouse_client::ProductionClickHouseClient;

mod disabled_clickhouse_client;
mod fake_clickhouse_client;
mod production_clickhouse_client;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClickHouseClientType {
    Production,
    Fake,
    Disabled,
}

/// Trait defining the interface for ClickHouse database operations.
///
/// For testing, use `FakeClickHouseClient` from the `mock_clickhouse_client` module.
/// For advanced mocking scenarios, you can use `mockall` to create custom mocks.
#[async_trait]
pub trait ClickHouseClient: Send + Sync + Debug + HealthCheckable {
    /// Returns the database URL
    fn database_url(&self) -> &SecretString;

    /// Returns the cluster name
    fn cluster_name(&self) -> &Option<String>;

    /// Returns the database name
    fn database(&self) -> &str;

    /// Returns whether batching is enabled
    fn is_batching_enabled(&self) -> bool;

    /// Returns the batch writer join handle if batching is enabled
    fn batcher_join_handle(&self) -> Option<BatchWriterHandle>;

    /// Writes rows to ClickHouse using batched writes (if enabled)
    async fn write_batched_internal(
        &self,
        rows: Vec<String>,
        table: TableName,
    ) -> Result<(), Error>;

    /// Writes rows to ClickHouse without batching
    async fn write_non_batched_internal(
        &self,
        rows: Vec<String>,
        table: TableName,
    ) -> Result<(), Error>;

    /// Runs a query with parameters, waiting for mutations to complete
    async fn run_query_synchronous(
        &self,
        query: String,
        parameters: &HashMap<&str, &str>,
    ) -> Result<ClickHouseResponse, Error>;

    /// Runs a query with parameters, returning a `DelayedError` rather than an `Error`
    async fn run_query_synchronous_delayed_err(
        &self,
        query: String,
        parameters: &HashMap<&str, &str>,
    ) -> Result<ClickHouseResponse, DelayedError>;

    /// Runs a query with external data
    async fn run_query_with_external_data(
        &self,
        external_data: ExternalDataInfo,
        query: String,
    ) -> Result<ClickHouseResponse, Error>;

    /// Checks if the database and migrations table exist
    async fn check_database_and_migrations_table_exists(&self) -> Result<bool, Error>;

    /// Creates the database and migrations table
    async fn create_database_and_migrations_table(&self) -> Result<(), Error>;

    /// Returns whether a cluster is configured
    fn is_cluster_configured(&self) -> bool;

    /// Returns the "ON CLUSTER {name}" string if a cluster is configured
    fn get_on_cluster_name(&self) -> String;

    /// Returns the table engine name, potentially with replication configured
    fn get_maybe_replicated_table_engine_name(
        &self,
        args: GetMaybeReplicatedTableEngineNameArgs<'_>,
    ) -> String;

    /// Returns the client type (for logging/debugging)
    fn client_type(&self) -> ClickHouseClientType;
}

// Because this is a supertrait of HealthCheckable, we need to use a custom mock macro instead of automock.
#[cfg(test)]
mock! {
    #[derive(Debug)]
    pub ClickHouseClient {}

    #[async_trait]
    impl ClickHouseClient for ClickHouseClient {
        fn database_url(&self) -> &SecretString;
        fn cluster_name(&self) -> &Option<String>;
        fn database(&self) -> &str;
        fn is_batching_enabled(&self) -> bool;
        fn batcher_join_handle(&self) -> Option<BatchWriterHandle>;
        async fn write_batched_internal(
            &self,
            rows: Vec<String>,
            table: TableName,
        ) -> Result<(), Error>;
        async fn write_non_batched_internal(
            &self,
            rows: Vec<String>,
            table: TableName,
        ) -> Result<(), Error>;
        async fn run_query_synchronous<'a, 'b, 'c, 'd>(
            &'a self,
            query: String,
            parameters: &'b HashMap<&'c str, &'d str>,
        ) -> Result<ClickHouseResponse, Error>;
        async fn run_query_synchronous_delayed_err<'a, 'b, 'c, 'd>(
            &'a self,
            query: String,
            parameters: &'b HashMap<&'c str, &'d str>,
        ) -> Result<ClickHouseResponse, DelayedError>;
        async fn run_query_with_external_data(
            &self,
            external_data: ExternalDataInfo,
            query: String,
        ) -> Result<ClickHouseResponse, Error>;
        async fn check_database_and_migrations_table_exists(&self) -> Result<bool, Error>;
        async fn create_database_and_migrations_table(&self) -> Result<(), Error>;
        fn is_cluster_configured(&self) -> bool;
        fn get_on_cluster_name(&self) -> String;
        fn get_maybe_replicated_table_engine_name<'a, 'b>(
            &'a self,
            args: GetMaybeReplicatedTableEngineNameArgs<'b>,
        ) -> String;
        fn client_type(&self) -> ClickHouseClientType;
    }

    #[async_trait]
    impl HealthCheckable for ClickHouseClient {
        async fn health(&self) -> Result<(), Error>;
    }
}
