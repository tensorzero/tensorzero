#![cfg(any(test, feature = "pyo3"))]

use async_trait::async_trait;
use lazy_static::lazy_static;
use secrecy::SecretString;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, RwLockWriteGuard};

use crate::db::clickhouse::batching::BatchWriterHandle;
use crate::db::clickhouse::clickhouse_client::ClickHouseClientType;
use crate::db::clickhouse::{
    ClickHouseClient, ClickHouseResponse, ClickHouseResponseMetadata, ExternalDataInfo,
    GetMaybeReplicatedTableEngineNameArgs, HealthCheckable, Rows, TableName,
};
use crate::error::{DelayedError, Error, ErrorDetails};

lazy_static! {
    static ref FAKE_DATABASE_URL: SecretString = SecretString::from("fake");
    static ref FAKE_CLUSTER_NAME: Option<String> = None;
}

/// Simple fake implementation of ClickHouseClient for testing
///
/// This implementation stores data in memory and provides basic fake functionality.
/// It's suitable for most testing scenarios where you need a simple fake that
/// records writes and returns empty responses for queries.
///
/// # Example
/// ```ignore
/// use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
///
/// let fake = ClickHouseConnectionInfo::new_fake();
/// // Use fake in your tests...
/// ```
#[derive(Debug, Clone)]
pub struct FakeClickHouseClient {
    pub data: Arc<RwLock<HashMap<String, Vec<serde_json::Value>>>>,
    pub healthy: bool,
}

impl FakeClickHouseClient {
    pub fn new(healthy: bool) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            healthy,
        }
    }
}

async fn write_fake<T: serde::Serialize + Send + Sync>(
    rows: Rows<'_, T>,
    table: TableName,
    tables: &mut RwLockWriteGuard<'_, HashMap<String, Vec<serde_json::Value>>>,
) -> Result<(), Error> {
    for row in rows.as_json()?.iter() {
        tables
            .entry(table.as_str().to_string())
            .or_default()
            .push(serde_json::Value::String(row.clone()));
    }
    Ok(())
}

#[async_trait]
impl ClickHouseClient for FakeClickHouseClient {
    fn database_url(&self) -> &SecretString {
        &FAKE_DATABASE_URL
    }

    fn cluster_name(&self) -> &Option<String> {
        &FAKE_CLUSTER_NAME
    }

    fn database(&self) -> &str {
        "fake"
    }

    fn is_batching_enabled(&self) -> bool {
        false
    }

    fn batcher_join_handle(&self) -> Option<BatchWriterHandle> {
        None
    }

    async fn write_batched_internal(
        &self,
        rows: Vec<String>,
        table: TableName,
    ) -> Result<(), Error> {
        write_fake(
            Rows::<String>::Serialized(&rows),
            table,
            &mut self.data.write().await,
        )
        .await
    }

    async fn write_non_batched_internal(
        &self,
        rows: Vec<String>,
        table: TableName,
    ) -> Result<(), Error> {
        write_fake(
            Rows::<String>::Serialized(&rows),
            table,
            &mut self.data.write().await,
        )
        .await
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
        ClickHouseClientType::Fake
    }
}

#[async_trait]
impl HealthCheckable for FakeClickHouseClient {
    async fn health(&self) -> Result<(), Error> {
        if self.healthy {
            Ok(())
        } else {
            Err(ErrorDetails::ClickHouseConnection {
                message: "Fake ClickHouse is not healthy".to_string(),
            }
            .into())
        }
    }
}
