//! Tests for Postgres setup: pg_cron and trigram indexes.

use crate::db::get_test_postgres;
use sqlx::{AssertSqlSafe, PgPool};
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

/// Tests that `setup_trigram_indexes` is idempotent when new partitions are created
/// between runs. When the parent index exists (via ON ONLY), Postgres auto-creates
/// and auto-attaches indexes on new partitions. A subsequent `setup_trigram_indexes`
/// must detect these auto-created indexes and skip them rather than failing on
/// `ALTER INDEX ... ATTACH PARTITION`.
#[tokio::test]
async fn test_setup_trigram_indexes_idempotent_with_new_partitions() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    // Ensure trigram indexes are set up (parent indexes exist)
    postgres_setup::setup_trigram_indexes(pool)
        .await
        .expect("Initial setup_trigram_indexes should succeed");

    // Create new partitions for day 8 (beyond the 0..7 range that create_partitions covers).
    // Because parent indexes exist, Postgres will auto-create and auto-attach indexes.
    let partitioned_tables_with_trigrams = ["chat_inference_data", "json_inference_data"];
    let mut created_partitions = Vec::new();
    for table in partitioned_tables_with_trigrams {
        let partition_name: String = sqlx::query_scalar(AssertSqlSafe(format!(
            "SELECT '{table}_' || to_char(CURRENT_DATE + 8, 'YYYY_MM_DD')"
        )))
        .fetch_one(pool)
        .await
        .unwrap_or_else(|e| panic!("Failed to compute partition name for `{table}`: {e}"));

        let sql = format!(
            "CREATE TABLE IF NOT EXISTS tensorzero.{partition_name} \
             PARTITION OF tensorzero.{table} \
             FOR VALUES FROM (CURRENT_DATE + 8) TO (CURRENT_DATE + 9)"
        );
        sqlx::raw_sql(AssertSqlSafe(sql))
            .execute(pool)
            .await
            .unwrap_or_else(|e| {
                panic!("Failed to create partition `{partition_name}` for `{table}`: {e}")
            });

        // Verify Postgres auto-created an index on the new partition
        let has_auto_index: bool = sqlx::query_scalar(
            r"
            SELECT EXISTS(
                SELECT 1 FROM pg_indexes
                WHERE schemaname = 'tensorzero'
                  AND tablename = $1
            )
            ",
        )
        .bind(&partition_name)
        .fetch_one(pool)
        .await
        .unwrap_or_else(|e| {
            panic!("Failed to check auto-created index on `{partition_name}`: {e}")
        });

        assert!(
            has_auto_index,
            "Postgres should auto-create indexes on new partition `{partition_name}` because parent index exists"
        );

        created_partitions.push(partition_name);
    }

    // Re-run trigram setup — this must not fail despite the auto-created indexes
    postgres_setup::setup_trigram_indexes(pool).await.expect(
        "setup_trigram_indexes should succeed after new partitions with auto-created indexes",
    );

    postgres_setup::check_trigram_indexes_configured_correctly(pool)
        .await
        .expect("Trigram indexes should still be valid after re-running setup with new partitions");

    // Clean up: drop the partitions we created
    for partition_name in &created_partitions {
        let sql = format!("DROP TABLE IF EXISTS tensorzero.{partition_name}");
        sqlx::raw_sql(AssertSqlSafe(sql))
            .execute(pool)
            .await
            .unwrap_or_else(|e| panic!("Failed to drop partition `{partition_name}`: {e}"));
    }
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
