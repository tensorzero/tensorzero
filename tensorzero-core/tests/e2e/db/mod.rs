//! Database E2E tests and helpers.
//!
//! This module provides connection helpers for e2e tests against ClickHouse and Postgres.

use sqlx::postgres::PgPoolOptions;
use tensorzero_core::db::postgres::PostgresConnectionInfo;

/// Generates test functions for both ClickHouse and Postgres backends.
macro_rules! make_db_test {
    ($test_name:ident) => {
        paste::paste! {
            #[tokio::test]
            async fn [<$test_name _clickhouse>]() {
                let conn = tensorzero_core::db::clickhouse::test_helpers::get_clickhouse().await;
                $test_name(conn).await;
            }

            #[tokio::test]
            async fn [<$test_name _postgres>]() {
                let conn = crate::db::get_test_postgres().await;
                $test_name(conn).await;
            }
        }
    };
}

/// Generates test functions for ClickHouse only.
macro_rules! make_clickhouse_only_test {
    ($test_name:ident) => {
        paste::paste! {
            #[tokio::test]
            async fn [<$test_name _clickhouse>]() {
                let conn = tensorzero_core::db::clickhouse::test_helpers::get_clickhouse().await;
                $test_name(conn).await;
            }
        }
    };
}

mod bandit_queries;
mod batch_inference_endpoint_internals;
mod batch_inference_queries;
mod dataset_queries;
mod evaluation_queries;
mod feedback_queries;
mod inference_count_queries;
mod inference_queries;
mod model_inference_queries;
mod model_provider_statistics_queries;
mod postgres;
mod rate_limit_queries;
mod select_queries;
mod workflow_evaluation_queries;

// ===== CONNECTION HELPERS =====

pub async fn get_test_postgres() -> PostgresConnectionInfo {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("Environment variable TENSORZERO_POSTGRES_URL must be set");

    let start = std::time::Instant::now();
    println!("Connecting to Postgres");
    let pool = PgPoolOptions::new()
        .connect(&postgres_url)
        .await
        .expect("Failed to connect to Postgres");
    println!("Connected to Postgres in {:?}", start.elapsed());
    PostgresConnectionInfo::new_with_pool(pool)
}
