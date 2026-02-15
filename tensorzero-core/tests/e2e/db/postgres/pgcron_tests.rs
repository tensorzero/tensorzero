//! Tests for pg_cron setup and validation.

use crate::db::get_test_postgres;
use sqlx::PgPool;
use tensorzero_core::db::postgres::pgcron;

/// Tests that pg_cron is available in our e2e test Postgres setup.
#[tokio::test]
async fn test_pgcron_is_available_in_e2e_setup() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    pgcron::check_pgcron_configured_correctly(pool)
        .await
        .expect("pg_cron should be available in our Postgres setup");
}

/// Tests that `setup_pgcron` is idempotent - running it multiple times should
/// result in exactly one schedule per job, not duplicates.
#[tokio::test]
async fn test_setup_pgcron_is_idempotent() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    // Run pgcron setup twice
    pgcron::setup_pgcron(pool)
        .await
        .expect("First setup_pgcron call should succeed");
    pgcron::setup_pgcron(pool)
        .await
        .expect("Second setup_pgcron call should succeed");

    // Verify there's exactly one job for partition creation
    let create_partitions_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM cron.job WHERE jobname = 'tensorzero_create_inference_partitions'",
    )
    .fetch_one(pool)
    .await
    .expect("Should be able to query cron.job table");

    assert_eq!(
        create_partitions_count, 1,
        "Should have exactly one 'tensorzero_create_inference_partitions' job after running setup twice"
    );

    // Verify there's exactly one job for dropping old partitions
    let drop_partitions_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM cron.job WHERE jobname = 'tensorzero_drop_old_inference_partitions'",
    )
    .fetch_one(pool)
    .await
    .expect("Should be able to query cron.job table");

    assert_eq!(
        drop_partitions_count, 1,
        "Should have exactly one 'tensorzero_drop_old_inference_partitions' job after running setup twice"
    );

    // Verify there's exactly one job for incremental minute latency histogram refresh
    let refresh_latency_histogram_minute_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM cron.job WHERE jobname = 'tensorzero_refresh_model_latency_histogram_minute_incremental'",
    )
    .fetch_one(pool)
    .await
    .expect("Should be able to query cron.job table");

    assert_eq!(
        refresh_latency_histogram_minute_count, 1,
        "Should have exactly one 'tensorzero_refresh_model_latency_histogram_minute_incremental' job after running setup twice"
    );

    // Verify there's exactly one job for incremental hour latency histogram refresh
    let refresh_latency_histogram_hour_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM cron.job WHERE jobname = 'tensorzero_refresh_model_latency_histogram_hour_incremental'",
    )
    .fetch_one(pool)
    .await
    .expect("Should be able to query cron.job table");

    assert_eq!(
        refresh_latency_histogram_hour_count, 1,
        "Should have exactly one 'tensorzero_refresh_model_latency_histogram_hour_incremental' job after running setup twice"
    );

    // Verify there's exactly one job for incremental model provider statistics refresh
    let refresh_model_provider_stats_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM cron.job WHERE jobname = 'tensorzero_refresh_model_provider_statistics_incremental'",
    )
    .fetch_one(pool)
    .await
    .expect("Should be able to query cron.job table");

    assert_eq!(
        refresh_model_provider_stats_count, 1,
        "Should have exactly one 'tensorzero_refresh_model_provider_statistics_incremental' job after running setup twice"
    );
}

/// Tests that `check_pgcron_configured_correctly` returns an error when pg_cron is not installed.
/// Uses `#[sqlx::test]` to get a fresh database without pg_cron.
#[sqlx::test]
async fn test_check_pgcron_configured_correctly_returns_error_without_pgcron(pool: PgPool) {
    let result = pgcron::check_pgcron_configured_correctly(&pool).await;

    assert!(
        result.is_err(),
        "check_pgcron_configured_correctly should return error when pg_cron is not installed"
    );

    let err = result.unwrap_err();
    let err_msg = err.suppress_logging_of_error_message();
    assert!(
        err_msg.contains("pg_cron"),
        "Error message should mention pg_cron, got: {err_msg}"
    );
}
