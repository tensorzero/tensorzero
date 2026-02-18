//! Tests for Postgres setup: pg_cron and trigram indexes.

use crate::db::get_test_postgres;
use sqlx::PgPool;
use tensorzero_core::db::postgres::postgres_setup;

// ===== pg_cron tests =====

/// Tests that pg_cron is available in our e2e test Postgres setup.
#[tokio::test]
async fn test_pgcron_is_available_in_e2e_setup() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    postgres_setup::check_pgcron_configured_correctly(pool)
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
    postgres_setup::setup_pgcron(pool)
        .await
        .expect("First setup_pgcron call should succeed");
    postgres_setup::setup_pgcron(pool)
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

    // Verify there's exactly one job for dropping old data partitions
    let drop_data_partitions_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM cron.job WHERE jobname = 'tensorzero_drop_old_inference_data_partitions'",
    )
    .fetch_one(pool)
    .await
    .expect("Should be able to query cron.job table");

    assert_eq!(
        drop_data_partitions_count, 1,
        "Should have exactly one 'tensorzero_drop_old_inference_data_partitions' job after running setup twice"
    );

    // Verify there's exactly one job for dropping old metadata partitions
    let drop_metadata_partitions_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM cron.job WHERE jobname = 'tensorzero_drop_old_inference_metadata_partitions'",
    )
    .fetch_one(pool)
    .await
    .expect("Should be able to query cron.job table");

    assert_eq!(
        drop_metadata_partitions_count, 1,
        "Should have exactly one 'tensorzero_drop_old_inference_metadata_partitions' job after running setup twice"
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
    let result = postgres_setup::check_pgcron_configured_correctly(&pool).await;

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

// ===== Trigram index tests =====

/// Tests that trigram indexes are available in our e2e test Postgres setup
/// (migrations have already run).
#[tokio::test]
async fn test_trigram_indexes_available_in_e2e_setup() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    postgres_setup::check_trigram_indexes_configured_correctly(pool)
        .await
        .expect("Trigram indexes should be configured in our Postgres setup");
}

/// Tests that `setup_trigram_indexes` is idempotent - running it multiple times
/// should not fail or create duplicate indexes.
#[tokio::test]
async fn test_setup_trigram_indexes_is_idempotent() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    postgres_setup::setup_trigram_indexes(pool)
        .await
        .expect("First setup_trigram_indexes call should succeed");
    postgres_setup::setup_trigram_indexes(pool)
        .await
        .expect("Second setup_trigram_indexes call should succeed");

    postgres_setup::check_trigram_indexes_configured_correctly(pool)
        .await
        .expect("Trigram indexes should still be valid after running setup twice");
}

/// Tests that trigram indexes exist on all expected tables and columns,
/// including both partitioned and non-partitioned tables.
#[tokio::test]
async fn test_trigram_indexes_exist_on_all_tables() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    let expected_indexes = [
        "idx_chat_inference_data_input_trgm",
        "idx_chat_inference_data_output_trgm",
        "idx_json_inference_data_input_trgm",
        "idx_json_inference_data_output_trgm",
        "idx_chat_datapoints_input_trgm",
        "idx_chat_datapoints_output_trgm",
        "idx_json_datapoints_input_trgm",
        "idx_json_datapoints_output_trgm",
    ];

    for index_name in expected_indexes {
        let exists: bool = sqlx::query_scalar(
            r"
            SELECT EXISTS(
                SELECT 1 FROM pg_class c
                JOIN pg_namespace ns ON c.relnamespace = ns.oid
                WHERE c.relname = $1
                  AND ns.nspname = 'tensorzero'
                  AND c.relkind IN ('i', 'I')
            )
            ",
        )
        .bind(index_name)
        .fetch_one(pool)
        .await
        .unwrap_or_else(|e| panic!("Failed to query index `{index_name}`: {e}"));

        assert!(exists, "Index `{index_name}` should exist");
    }
}

/// Tests that partition indexes are attached to their parent index for partitioned tables.
#[tokio::test]
async fn test_partition_indexes_are_attached() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    // For each partitioned table, check that every partition has an attached trigram index
    let partitioned_tables = ["chat_inference_data", "json_inference_data"];
    let columns = ["input", "output"];

    for table in partitioned_tables {
        for column in columns {
            let parent_index = format!("idx_{table}_{column}_trgm");

            // Get the number of partitions on the table
            let partition_count: i64 = sqlx::query_scalar(
                r"
                SELECT COUNT(*)
                FROM pg_inherits
                JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
                JOIN pg_namespace ns ON parent.relnamespace = ns.oid
                WHERE parent.relname = $1
                  AND ns.nspname = 'tensorzero'
                ",
            )
            .bind(table)
            .fetch_one(pool)
            .await
            .unwrap_or_else(|e| panic!("Failed to count partitions for `{table}`: {e}"));

            assert!(
                partition_count > 0,
                "`{table}` should have at least one partition"
            );

            // Count how many child indexes are attached to the parent index
            let attached_count: i64 = sqlx::query_scalar(
                r"
                SELECT COUNT(*)
                FROM pg_inherits
                JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
                JOIN pg_namespace ns ON parent.relnamespace = ns.oid
                WHERE parent.relname = $1
                  AND ns.nspname = 'tensorzero'
                ",
            )
            .bind(&parent_index)
            .fetch_one(pool)
            .await
            .unwrap_or_else(|e| {
                panic!("Failed to count attached indexes for `{parent_index}`: {e}")
            });

            assert_eq!(
                attached_count, partition_count,
                "All {partition_count} partitions of `{table}` should have `{column}` trigram indexes attached to `{parent_index}`, but only {attached_count} are attached"
            );
        }
    }
}

/// Tests that `check_trigram_indexes_configured_correctly` returns an error when
/// pg_trgm is not installed. Uses `#[sqlx::test]` to get a fresh database.
#[sqlx::test]
async fn test_check_trigram_indexes_returns_error_without_pg_trgm(pool: PgPool) {
    let result = postgres_setup::check_trigram_indexes_configured_correctly(&pool).await;

    assert!(
        result.is_err(),
        "check_trigram_indexes_configured_correctly should return error when pg_trgm is not installed"
    );

    let err = result.unwrap_err();
    let err_msg = err.suppress_logging_of_error_message();
    assert!(
        err_msg.contains("pg_trgm"),
        "Error message should mention pg_trgm, got: {err_msg}"
    );
}
