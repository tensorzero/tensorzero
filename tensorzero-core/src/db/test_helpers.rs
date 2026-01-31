#![cfg(feature = "e2e_tests")]

use std::time::Duration;

use tonic::async_trait;

use crate::db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo};

/// Trait for database operations needed in tests.
/// Provides a database-agnostic way to ensure writes are visible before reads.
#[async_trait]
pub trait TestDatabaseHelpers: Send + Sync {
    /// Ensures all pending writes are visible for subsequent reads.
    async fn flush_pending_writes(&self);

    /// (In ClickHouse only) sleeps for a given duration to ensure writes are visible.
    async fn sleep_for_writes_to_be_visible(&self);
}

#[async_trait]
impl TestDatabaseHelpers for ClickHouseConnectionInfo {
    /// For ClickHouse, this flushes the async insert queue.
    async fn flush_pending_writes(&self) {
        if let Err(e) = self
            .run_query_synchronous_no_params("SYSTEM FLUSH ASYNC INSERT QUEUE".to_string())
            .await
        {
            tracing::warn!("Failed to run `SYSTEM FLUSH ASYNC INSERT QUEUE`: {}", e);
        }
    }

    /// For ClickHouse, this sleeps for a given duration to ensure writes are visible.
    async fn sleep_for_writes_to_be_visible(&self) {
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

#[async_trait]
impl TestDatabaseHelpers for PostgresConnectionInfo {
    /// For Postgres, this is a no-op since writes are immediately visible.
    async fn flush_pending_writes(&self) {}

    /// For Postgres, this is a no-op since writes are immediately visible.
    async fn sleep_for_writes_to_be_visible(&self) {}
}
