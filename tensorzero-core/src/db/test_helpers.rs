#![cfg(any(test, feature = "e2e_tests"))]

use std::ops::AsyncFn;
use std::time::Duration;

use tonic::async_trait;

use crate::db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo};

/// Polls an async closure until it returns `Some(T)`, with exponential backoff.
/// Used in e2e tests to wait for ClickHouse eventual consistency after gateway writes.
///
/// # Examples
///
/// ```ignore
/// // Polling a select helper (most common pattern):
/// let result = poll_result_until_some(async || {
///     select_feedback_clickhouse(&clickhouse, "CommentFeedback", feedback_id).await
/// }).await;
///
/// // Polling a direct query with custom logic:
/// let result: Value = poll_result_until_some(async || {
///     clickhouse.flush_pending_writes().await;
///     let query = format!("SELECT * FROM MyTable WHERE id='{}' FORMAT JSONEachRow", my_id);
///     let response = clickhouse.run_query_synchronous_no_params(query).await.ok()?;
///     serde_json::from_str(&response.response).ok()
/// }).await;
/// ```
///
/// # Panics
///
/// Panics if the closure does not return `Some` within 10 seconds.
pub async fn poll_result_until_some<T>(f: impl AsyncFn() -> Option<T>) -> T {
    let timeout = Duration::from_secs(10);
    let start = std::time::Instant::now();
    let mut delay = Duration::from_millis(100);
    let max_delay = Duration::from_secs(2);
    loop {
        if let Some(result) = f().await {
            return result;
        }
        assert!(
            start.elapsed() <= timeout,
            "Timed out polling for expected data after {timeout:?}",
        );
        tokio::time::sleep(delay).await;
        delay = std::cmp::min(delay * 2, max_delay);
    }
}

/// Legacy macro form of [`poll_result_until_some`]. Prefer the function form for new code.
///
/// Polls until the expression evaluates to `Some(T)`, with exponential backoff.
/// The expression is re-evaluated on each attempt.
///
/// # Panics
///
/// Panics if the expression does not return `Some` within 10 seconds.
#[macro_export]
macro_rules! poll_clickhouse_for_result {
    ($query_expr:expr) => {{
        let __timeout = ::std::time::Duration::from_secs(10);
        let __start = ::std::time::Instant::now();
        let mut __delay = ::std::time::Duration::from_millis(100);
        let __max_delay = ::std::time::Duration::from_secs(2);
        loop {
            if let ::std::option::Option::Some(__result) = $query_expr {
                break __result;
            }
            if __start.elapsed() > __timeout {
                panic!("Timed out polling for expected data after {:?}", __timeout);
            }
            ::tokio::time::sleep(__delay).await;
            __delay = ::std::cmp::min(__delay * 2, __max_delay);
        }
    }};
}

/// Trait for database operations needed in tests.
/// Provides a database-agnostic way to ensure writes are visible before reads.
#[async_trait]
pub trait TestDatabaseHelpers: Send + Sync {
    /// Ensures all pending writes are visible for subsequent reads.
    async fn flush_pending_writes(&self);
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
}

#[async_trait]
impl TestDatabaseHelpers for PostgresConnectionInfo {
    /// For Postgres, this is a no-op since writes are immediately visible.
    async fn flush_pending_writes(&self) {}
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
