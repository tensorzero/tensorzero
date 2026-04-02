#![cfg(any(test, feature = "e2e_tests"))]

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

    /// Prepares the model provider statistics rollup for querying after an insert.
    ///
    /// ClickHouse: flushes the async insert queue so the materialized view has
    /// processed the new rows. Callers should then poll `get_model_usage_timeseries`
    /// until the expected data appears.
    ///
    /// Postgres: triggers `refresh_model_provider_statistics_incremental` so the
    /// rollup table is up to date before querying.
    async fn prepare_model_provider_statistics(&self);
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

    /// For ClickHouse, flush the async insert queue so the MV has processed the rows.
    async fn prepare_model_provider_statistics(&self) {
        self.flush_pending_writes().await;
    }
}

#[async_trait]
impl TestDatabaseHelpers for PostgresConnectionInfo {
    /// For Postgres, this is a no-op since writes are immediately visible.
    async fn flush_pending_writes(&self) {}

    /// For Postgres, this is a no-op since writes are immediately visible.
    async fn sleep_for_writes_to_be_visible(&self) {}

    /// For Postgres, trigger the incremental refresh so the rollup table is up to date.
    #[expect(clippy::panic)]
    async fn prepare_model_provider_statistics(&self) {
        if let Some(pool) = self.get_pool() {
            sqlx::query(
                "SELECT tensorzero.refresh_model_provider_statistics_incremental(full_refresh => TRUE)",
            )
            .execute(pool)
            .await
            .unwrap_or_else(|e| {
                panic!("refresh_model_provider_statistics_incremental failed: {e}")
            });
        }
    }
}

/// Normalize whitespace and newlines in a query for comparison
pub fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Assert that the query is exactly equal to the expected query (ignoring whitespace and newline differences)
///
/// # Panics
///
/// This function will panic if the query is not exactly equal to the expected query. For test usage only.
pub fn assert_query_equals(query: &str, expected_query: &str) {
    let normalized_query = normalize_whitespace(query);
    let normalized_expected_query = normalize_whitespace(expected_query);
    assert_eq!(normalized_query, normalized_expected_query);
}

/// Assert that the query contains a section (ignoring whitespace and newline differences)
///
/// # Panics
///
/// This function will panic if the query does not contain the expected section. For test usage only.
pub fn assert_query_contains(query: &str, expected_section: &str) {
    let normalized_query = normalize_whitespace(query);
    let normalized_section = normalize_whitespace(expected_section);
    assert!(
        normalized_query.contains(&normalized_section),
        "Query does not contain expected section.\nExpected section: {normalized_section}\nFull query: {normalized_query}"
    );
}

/// Assert that the query does not contain a section (ignoring whitespace and newline differences)
///
/// # Panics
///
/// This function will panic if the query contains the unexpected section. For test usage only.
pub fn assert_query_does_not_contain(query: &str, unexpected_section: &str) {
    let normalized_query = normalize_whitespace(query);
    let normalized_section = normalize_whitespace(unexpected_section);
    assert!(
        !normalized_query.contains(&normalized_section),
        "Query contains unexpected section: {normalized_section}\nFull query: {normalized_query}"
    );
}
