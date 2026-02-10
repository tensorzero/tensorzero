//! Tests for Postgres functions in the tensorzero schema.

use crate::db::get_test_postgres;
use chrono::{Days, TimeZone, Utc};
use sqlx::QueryBuilder;
use uuid::{NoContext, Timestamp, Uuid};

/// Tests that the Postgres `uint128_to_uuid` function correctly converts a UInt128 decimal
/// representation (as exported by ClickHouse) back to a UUID.
///
/// ClickHouse stores UUIDs as UInt128 internally, and when exporting to JSON/CSV, it
/// represents them as decimal strings. This function reverses that conversion.
#[tokio::test]
async fn test_uint128_to_uuid_known_value() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    // Known UUID from fixture data: 01968d04-142c-7e53-8ea7-3a3255b518dc
    let expected_uuid = Uuid::parse_str("01968d04-142c-7e53-8ea7-3a3255b518dc").unwrap();

    // Convert UUID to UInt128 decimal representation (big-endian interpretation)
    let uint128_value = expected_uuid.as_u128();
    let uint128_decimal = uint128_value.to_string();

    // Call the Postgres function
    let (result_uuid,): (Uuid,) = sqlx::query_as("SELECT tensorzero.uint128_to_uuid($1::NUMERIC)")
        .bind(&uint128_decimal)
        .fetch_one(pool)
        .await
        .expect("uint128_to_uuid query should succeed");

    assert_eq!(
        result_uuid, expected_uuid,
        "uint128_to_uuid should correctly convert UInt128 decimal back to UUID"
    );
}

/// Tests that uint128_to_uuid works with zero.
#[tokio::test]
async fn test_uint128_to_uuid_zero() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    let (result_uuid,): (Uuid,) = sqlx::query_as("SELECT tensorzero.uint128_to_uuid(0::NUMERIC)")
        .fetch_one(pool)
        .await
        .expect("uint128_to_uuid query should succeed");

    assert_eq!(
        result_uuid,
        Uuid::nil(),
        "uint128_to_uuid(0) should return the nil UUID"
    );
}

/// Tests that uint128_to_uuid works with max value (all bits set).
#[tokio::test]
async fn test_uint128_to_uuid_max() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    let max_uint128 = u128::MAX.to_string();

    let (result_uuid,): (Uuid,) = sqlx::query_as("SELECT tensorzero.uint128_to_uuid($1::NUMERIC)")
        .bind(&max_uint128)
        .fetch_one(pool)
        .await
        .expect("uint128_to_uuid query should succeed");

    assert_eq!(
        result_uuid,
        Uuid::max(),
        "uint128_to_uuid(MAX) should return the max UUID (all 1s)"
    );
}

/// Tests that the Postgres `uuid_v7_to_timestamp` function extracts the same timestamp
/// that Rust's `uuid` crate encodes into a UUIDv7.
#[tokio::test]
async fn test_uuid_v7_to_timestamp() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    // Generate a UUIDv7 in Rust
    let rust_uuid = Uuid::now_v7();

    // Extract timestamp using Postgres function
    let (postgres_timestamp,): (chrono::DateTime<Utc>,) =
        sqlx::query_as("SELECT tensorzero.uuid_v7_to_timestamp($1::uuid)")
            .bind(rust_uuid)
            .fetch_one(pool)
            .await
            .expect("Query should succeed");

    // Generate a new UUIDv7 using the timestamp extracted by Postgres
    let postgres_millis = postgres_timestamp.timestamp_millis() as u64;
    let reconstructed_uuid = Uuid::new_v7(Timestamp::from_unix(
        NoContext,
        postgres_millis / 1000,
        (postgres_millis % 1000) as u32 * 1_000_000,
    ));

    // The first 48 bits (timestamp portion) should match
    let rust_timestamp_bits = &rust_uuid.as_bytes()[..6];
    let reconstructed_timestamp_bits = &reconstructed_uuid.as_bytes()[..6];

    assert_eq!(
        rust_timestamp_bits, reconstructed_timestamp_bits,
        "Postgres-extracted timestamp should match the original UUIDv7 timestamp"
    );
}

/// Tests the function with a known timestamp value.
#[tokio::test]
async fn test_uuid_v7_to_timestamp_known_value() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    // Create a UUIDv7 with a known timestamp: Jan 1, 2024 00:00:00.123 UTC
    let known_time =
        Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::milliseconds(123);
    let secs = known_time.timestamp() as u64;
    let nanos = known_time.timestamp_subsec_nanos();
    let rust_uuid = Uuid::new_v7(Timestamp::from_unix(NoContext, secs, nanos));

    // Extract timestamp using Postgres function
    let (postgres_timestamp,): (chrono::DateTime<Utc>,) =
        sqlx::query_as("SELECT tensorzero.uuid_v7_to_timestamp($1::uuid)")
            .bind(rust_uuid)
            .fetch_one(pool)
            .await
            .expect("Query should succeed");

    // Postgres should return the same millisecond-precision timestamp
    assert_eq!(
        postgres_timestamp.timestamp_millis(),
        known_time.timestamp_millis(),
        "Postgres should extract the correct timestamp from UUIDv7"
    );
}

const TEST_PARTITIONED_TABLE: &str = "test_partitioned_table";
const TEST_RETENTION_TABLE: &str = "test_retention_table";
const TEST_RETENTION_KEY: &str = "test_retention_days";

/// Tests that `create_partitions` creates partitions for the next 8 days.
#[tokio::test]
async fn test_create_partitions() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    // Clean up any existing test table and its partitions
    let mut qb: QueryBuilder<sqlx::Postgres> =
        QueryBuilder::new("DROP TABLE IF EXISTS tensorzero.");
    qb.push(TEST_PARTITIONED_TABLE);
    qb.push(" CASCADE");
    qb.build()
        .execute(pool)
        .await
        .expect("Cleanup should succeed");

    // Create a partitioned table
    let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("CREATE TABLE tensorzero.");
    qb.push(TEST_PARTITIONED_TABLE);
    qb.push(" (id UUID NOT NULL, created_at DATE NOT NULL, PRIMARY KEY (id, created_at)) PARTITION BY RANGE (created_at)");
    qb.build()
        .execute(pool)
        .await
        .expect("Table creation should succeed");

    // Call create_partitions
    sqlx::query("SELECT tensorzero.create_partitions($1)")
        .bind(TEST_PARTITIONED_TABLE)
        .execute(pool)
        .await
        .expect("create_partitions should succeed");

    // Verify partitions were created for days 0..7 (8 partitions)
    let (partition_count,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'tensorzero' AND tablename LIKE $1",
    )
    .bind(format!("{TEST_PARTITIONED_TABLE}_%"))
    .fetch_one(pool)
    .await
    .expect("Count query should succeed");

    assert_eq!(
        partition_count, 8,
        "create_partitions should create 8 partitions (today + 7 future days)"
    );

    // Verify the partition names match expected dates
    let today = Utc::now().date_naive();
    for i in 0..8 {
        let partition_date = today + Days::new(i);
        let expected_partition = format!(
            "{}_{}",
            TEST_PARTITIONED_TABLE,
            partition_date.format("%Y_%m_%d")
        );

        let (exists,): (bool,) = sqlx::query_as(
            "SELECT EXISTS(SELECT 1 FROM pg_tables WHERE schemaname = 'tensorzero' AND tablename = $1)",
        )
        .bind(&expected_partition)
        .fetch_one(pool)
        .await
        .expect("Existence check should succeed");

        assert!(
            exists,
            "Partition {expected_partition} should exist for day offset {i}"
        );
    }

    // Clean up
    let mut qb: QueryBuilder<sqlx::Postgres> =
        QueryBuilder::new("DROP TABLE IF EXISTS tensorzero.");
    qb.push(TEST_PARTITIONED_TABLE);
    qb.push(" CASCADE");
    qb.build()
        .execute(pool)
        .await
        .expect("Cleanup should succeed");
}

/// Tests that `drop_old_partitions` drops partitions older than the retention period.
#[tokio::test]
async fn test_drop_old_partitions() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    // Clean up any existing test table and retention config
    let mut qb: QueryBuilder<sqlx::Postgres> =
        QueryBuilder::new("DROP TABLE IF EXISTS tensorzero.");
    qb.push(TEST_RETENTION_TABLE);
    qb.push(" CASCADE");
    qb.build()
        .execute(pool)
        .await
        .expect("Cleanup should succeed");

    sqlx::query("DELETE FROM tensorzero.retention_config WHERE key = $1")
        .bind(TEST_RETENTION_KEY)
        .execute(pool)
        .await
        .expect("Retention config cleanup should succeed");

    // Create a partitioned table
    let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("CREATE TABLE tensorzero.");
    qb.push(TEST_RETENTION_TABLE);
    qb.push(" (id UUID NOT NULL, created_at DATE NOT NULL, PRIMARY KEY (id, created_at)) PARTITION BY RANGE (created_at)");
    qb.build()
        .execute(pool)
        .await
        .expect("Table creation should succeed");

    // Manually create partitions: some old (30+ days ago), some recent
    let today = Utc::now().date_naive();
    let old_dates = [
        today - Days::new(60),
        today - Days::new(45),
        today - Days::new(31),
    ];
    let recent_dates = [today - Days::new(5), today - Days::new(1), today];

    for date in old_dates.iter().chain(recent_dates.iter()) {
        let partition_name = format!("{}_{}", TEST_RETENTION_TABLE, date.format("%Y_%m_%d"));
        let next_date = *date + Days::new(1);

        let mut qb: QueryBuilder<sqlx::Postgres> =
            QueryBuilder::new("CREATE TABLE IF NOT EXISTS tensorzero.");
        qb.push(&partition_name);
        qb.push(" PARTITION OF tensorzero.");
        qb.push(TEST_RETENTION_TABLE);
        qb.push(" FOR VALUES FROM ('");
        qb.push(date.to_string());
        qb.push("') TO ('");
        qb.push(next_date.to_string());
        qb.push("')");
        qb.build()
            .execute(pool)
            .await
            .expect("Partition creation should succeed");
    }

    // Verify all 6 partitions exist
    let (initial_count,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'tensorzero' AND tablename LIKE $1",
    )
    .bind(format!("{TEST_RETENTION_TABLE}_%"))
    .fetch_one(pool)
    .await
    .expect("Count query should succeed");

    assert_eq!(initial_count, 6, "Should have 6 partitions initially");

    // Set retention to 30 days
    sqlx::query(
        "INSERT INTO tensorzero.retention_config (key, value) VALUES ($1, '30') ON CONFLICT (key) DO UPDATE SET value = '30'",
    )
    .bind(TEST_RETENTION_KEY)
    .execute(pool)
    .await
    .expect("Retention config insert should succeed");

    // Call drop_old_partitions
    sqlx::query("SELECT tensorzero.drop_old_partitions($1, $2)")
        .bind(TEST_RETENTION_TABLE)
        .bind(TEST_RETENTION_KEY)
        .execute(pool)
        .await
        .expect("drop_old_partitions should succeed");

    // Verify old partitions were dropped (only 3 recent should remain)
    let (final_count,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'tensorzero' AND tablename LIKE $1",
    )
    .bind(format!("{TEST_RETENTION_TABLE}_%"))
    .fetch_one(pool)
    .await
    .expect("Count query should succeed");

    assert_eq!(
        final_count, 3,
        "Should have 3 partitions after dropping old ones (30+ days old)"
    );

    // Verify the old partitions are gone
    for date in &old_dates {
        let partition_name = format!("{}_{}", TEST_RETENTION_TABLE, date.format("%Y_%m_%d"));
        let (exists,): (bool,) = sqlx::query_as(
            "SELECT EXISTS(SELECT 1 FROM pg_tables WHERE schemaname = 'tensorzero' AND tablename = $1)",
        )
        .bind(&partition_name)
        .fetch_one(pool)
        .await
        .expect("Existence check should succeed");

        assert!(
            !exists,
            "Old partition {partition_name} should have been dropped"
        );
    }

    // Verify recent partitions still exist
    for date in &recent_dates {
        let partition_name = format!("{}_{}", TEST_RETENTION_TABLE, date.format("%Y_%m_%d"));
        let (exists,): (bool,) = sqlx::query_as(
            "SELECT EXISTS(SELECT 1 FROM pg_tables WHERE schemaname = 'tensorzero' AND tablename = $1)",
        )
        .bind(&partition_name)
        .fetch_one(pool)
        .await
        .expect("Existence check should succeed");

        assert!(
            exists,
            "Recent partition {partition_name} should still exist"
        );
    }

    // Clean up
    let mut qb: QueryBuilder<sqlx::Postgres> =
        QueryBuilder::new("DROP TABLE IF EXISTS tensorzero.");
    qb.push(TEST_RETENTION_TABLE);
    qb.push(" CASCADE");
    qb.build()
        .execute(pool)
        .await
        .expect("Cleanup should succeed");

    sqlx::query("DELETE FROM tensorzero.retention_config WHERE key = $1")
        .bind(TEST_RETENTION_KEY)
        .execute(pool)
        .await
        .expect("Retention config cleanup should succeed");
}

const TEST_NO_RETENTION_TABLE: &str = "test_no_retention_table";
const TEST_MISSING_RETENTION_KEY: &str = "test_missing_retention_key";

/// Tests that `drop_old_partitions` does nothing when the retention key is not configured.
#[tokio::test]
async fn test_drop_old_partitions_skips_when_retention_not_configured() {
    let conn = get_test_postgres().await;
    let pool = conn.get_pool().expect("Pool should be available");

    // Ensure the retention key does not exist
    sqlx::query("DELETE FROM tensorzero.retention_config WHERE key = $1")
        .bind(TEST_MISSING_RETENTION_KEY)
        .execute(pool)
        .await
        .expect("Retention config cleanup should succeed");

    // Clean up any existing test table
    let mut qb: QueryBuilder<sqlx::Postgres> =
        QueryBuilder::new("DROP TABLE IF EXISTS tensorzero.");
    qb.push(TEST_NO_RETENTION_TABLE);
    qb.push(" CASCADE");
    qb.build()
        .execute(pool)
        .await
        .expect("Cleanup should succeed");

    // Create a partitioned table
    let mut qb: QueryBuilder<sqlx::Postgres> = QueryBuilder::new("CREATE TABLE tensorzero.");
    qb.push(TEST_NO_RETENTION_TABLE);
    qb.push(" (id UUID NOT NULL, created_at DATE NOT NULL, PRIMARY KEY (id, created_at)) PARTITION BY RANGE (created_at)");
    qb.build()
        .execute(pool)
        .await
        .expect("Table creation should succeed");

    // Create some old partitions that would be dropped if retention were configured
    let today = Utc::now().date_naive();
    let old_dates = [
        today - Days::new(60),
        today - Days::new(45),
        today - Days::new(31),
    ];

    for date in &old_dates {
        let partition_name = format!("{}_{}", TEST_NO_RETENTION_TABLE, date.format("%Y_%m_%d"));
        let next_date = *date + Days::new(1);

        let mut qb: QueryBuilder<sqlx::Postgres> =
            QueryBuilder::new("CREATE TABLE IF NOT EXISTS tensorzero.");
        qb.push(&partition_name);
        qb.push(" PARTITION OF tensorzero.");
        qb.push(TEST_NO_RETENTION_TABLE);
        qb.push(" FOR VALUES FROM ('");
        qb.push(date.to_string());
        qb.push("') TO ('");
        qb.push(next_date.to_string());
        qb.push("')");
        qb.build()
            .execute(pool)
            .await
            .expect("Partition creation should succeed");
    }

    // Verify all 3 partitions exist before calling drop_old_partitions
    let (initial_count,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'tensorzero' AND tablename LIKE $1",
    )
    .bind(format!("{TEST_NO_RETENTION_TABLE}_%"))
    .fetch_one(pool)
    .await
    .expect("Count query should succeed");

    assert_eq!(initial_count, 3, "Should have 3 partitions initially");

    // Call drop_old_partitions with a non-existent retention key
    sqlx::query("SELECT tensorzero.drop_old_partitions($1, $2)")
        .bind(TEST_NO_RETENTION_TABLE)
        .bind(TEST_MISSING_RETENTION_KEY)
        .execute(pool)
        .await
        .expect("drop_old_partitions should succeed even without retention config");

    // Verify all partitions still exist (nothing was dropped)
    let (final_count,): (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'tensorzero' AND tablename LIKE $1",
    )
    .bind(format!("{TEST_NO_RETENTION_TABLE}_%"))
    .fetch_one(pool)
    .await
    .expect("Count query should succeed");

    assert_eq!(
        final_count, 3,
        "All partitions should remain when retention key is not configured"
    );

    // Clean up
    let mut qb: QueryBuilder<sqlx::Postgres> =
        QueryBuilder::new("DROP TABLE IF EXISTS tensorzero.");
    qb.push(TEST_NO_RETENTION_TABLE);
    qb.push(" CASCADE");
    qb.build()
        .execute(pool)
        .await
        .expect("Cleanup should succeed");
}
