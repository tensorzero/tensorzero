//! Database E2E tests and helpers.
//!
//! This module provides connection helpers for e2e tests against ClickHouse and Postgres.

use sqlx::postgres::PgPoolOptions;
use tensorzero_core::db::postgres::PostgresConnectionInfo;

mod bandit_queries;
mod batch_inference_queries;
mod dataset_queries;
mod evaluation_queries;
mod feedback_queries;
mod inference_count_queries;
mod inference_queries;
mod postgres;
mod rate_limit_queries;
mod select_queries;
mod workflow_evaluation_queries;

// ===== CONNECTION HELPERS =====

pub async fn get_test_postgres() -> PostgresConnectionInfo {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("Environment variable TENSORZERO_POSTGRES_URL must be set");

    let start = std::time::Instant::now();
    println!("Connecting to PostgreSQL");
    let pool = PgPoolOptions::new()
        .connect(&postgres_url)
        .await
        .expect("Failed to connect to PostgreSQL");
    println!("Connected to PostgreSQL in {:?}", start.elapsed());
    PostgresConnectionInfo::new_with_pool(pool)
}
