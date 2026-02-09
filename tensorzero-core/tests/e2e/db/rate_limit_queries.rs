//! Shared test logic for RateLimitQueries implementations (Postgres and Valkey).
//!
//! Each test function accepts a connection implementing `RateLimitQueries` and a `test_id`
//! to namespace keys and prevent interference between parallel tests.
//!
//! TODO(#5744): These tests currently run as part of the ClickHouse test job, but they
//! don't depend on ClickHouse. Refactor CI to run Postgres and Valkey tests separately.

use tensorzero_core::db::{ConsumeTicketsRequest, RateLimitQueries, ReturnTicketsRequest};
use tensorzero_core::rate_limiting::{ActiveRateLimitKey, RateLimitInterval};

/// Invokes a callback macro for each rate limit test.
/// This ensures both Postgres and Valkey test suites include all tests.
/// To add a new test, add it here and it will automatically be included in both backends.
macro_rules! invoke_rate_limit_tests {
    ($test_macro:ident) => {
        $test_macro!(test_atomic_multi_key_all_or_nothing);
        $test_macro!(test_atomic_consistency_under_load);
        $test_macro!(test_race_condition_no_over_consumption);
        $test_macro!(test_race_condition_interleaved_consume_return);
        $test_macro!(test_rate_limit_lifecycle);
        $test_macro!(test_capacity_boundaries);
        $test_macro!(test_refill_mechanics);
        $test_macro!(test_zero_refill_mechanics);
        $test_macro!(test_empty_operations);
        $test_macro!(test_new_bucket_behavior);
        $test_macro!(test_concurrent_stress);
        $test_macro!(test_consume_tickets_rejects_duplicate_keys);
        $test_macro!(test_return_tickets_rejects_duplicate_keys);
    };
}

// ===== POSTGRES TESTS =====

mod postgres {
    use sqlx::ConnectOptions;
    use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
    use tensorzero_core::db::postgres::{
        PostgresConnectionInfo, manual_run_postgres_migrations_with_url,
    };
    use uuid::Uuid;

    async fn setup_postgres(
        pool_opts: PgPoolOptions,
        conn_opts: PgConnectOptions,
    ) -> PostgresConnectionInfo {
        manual_run_postgres_migrations_with_url(conn_opts.to_url_lossy().as_ref())
            .await
            .unwrap();
        let pool = pool_opts.connect_with(conn_opts).await.unwrap();
        PostgresConnectionInfo::new_with_pool(pool)
    }

    macro_rules! postgres_rate_limit_test {
        ($test_name:ident) => {
            #[sqlx::test]
            async fn $test_name(pool_opts: PgPoolOptions, conn_opts: PgConnectOptions) {
                let conn = setup_postgres(pool_opts, conn_opts).await;
                super::$test_name(conn, &Uuid::now_v7().to_string()).await;
            }
        };
    }

    invoke_rate_limit_tests!(postgres_rate_limit_test);
}

// ===== VALKEY TESTS =====

mod valkey {
    use tensorzero_core::db::valkey::ValkeyConnectionInfo;
    use uuid::Uuid;

    async fn create_valkey_client() -> ValkeyConnectionInfo {
        let url =
            std::env::var("TENSORZERO_VALKEY_URL").expect("TENSORZERO_VALKEY_URL should be set");
        ValkeyConnectionInfo::new(&url, None)
            .await
            .expect("Failed to connect to Valkey")
    }

    macro_rules! valkey_rate_limit_test {
        ($test_name:ident) => {
            #[tokio::test]
            async fn $test_name() {
                let conn = create_valkey_client().await;
                super::$test_name(conn, &Uuid::now_v7().to_string()).await;
            }
        };
    }

    invoke_rate_limit_tests!(valkey_rate_limit_test);
}

// ===== HELPER FUNCTIONS =====

pub fn create_consume_request(
    key: &str,
    requested: u64,
    capacity: u64,
    refill_amount: u64,
    refill_interval: RateLimitInterval,
) -> ConsumeTicketsRequest {
    ConsumeTicketsRequest {
        key: ActiveRateLimitKey(key.to_string()),
        requested,
        capacity,
        refill_amount,
        refill_interval,
    }
}

pub fn create_return_request(
    key: &str,
    returned: u64,
    capacity: u64,
    refill_amount: u64,
    refill_interval: RateLimitInterval,
) -> ReturnTicketsRequest {
    ReturnTicketsRequest {
        key: ActiveRateLimitKey(key.to_string()),
        returned,
        capacity,
        refill_amount,
        refill_interval,
    }
}

fn test_key(test_id: &str, suffix: &str) -> String {
    format!("{test_id}_{suffix}")
}

// ===== ATOMIC BEHAVIOR TESTS =====

pub async fn test_atomic_multi_key_all_or_nothing(
    conn: impl RateLimitQueries + Clone,
    test_id: &str,
) {
    let key1 = test_key(test_id, "atomic_key1");
    let key2 = test_key(test_id, "atomic_key2");

    // First, consume some tokens from key1 to set up a scenario where key1 can succeed but key2 fails
    conn.consume_tickets(&[create_consume_request(
        &key1,
        50,
        100,
        10,
        RateLimitInterval::Minute,
    )])
    .await
    .unwrap();

    // Now create a batch where key1 can succeed (50 remaining) but key2 will fail (requesting more than capacity)
    let results = conn
        .consume_tickets(&[
            create_consume_request(&key1, 30, 100, 10, RateLimitInterval::Minute),
            create_consume_request(&key2, 150, 100, 10, RateLimitInterval::Minute), // Exceeds capacity - will fail
        ])
        .await
        .unwrap();

    // ALL requests should fail because it's atomic
    assert!(
        !results[0].success,
        "Key1 should fail due to atomic behavior even though it could succeed alone"
    );
    assert!(!results[1].success, "Key2 should fail as expected");

    // Verify key1 balance unchanged - no partial consumption
    let balance = conn
        .get_balance(&key1, 100, 10, RateLimitInterval::Minute)
        .await
        .unwrap();
    assert_eq!(
        balance, 50,
        "Key1 balance should be unchanged due to atomic rollback"
    );
}

pub async fn test_atomic_consistency_under_load(
    conn: impl RateLimitQueries + Clone + 'static,
    test_id: &str,
) {
    // Launch many concurrent multi-key requests where some will fail
    let handles: Vec<_> = (0..20)
        .map(|i| {
            let conn_clone = conn.clone();
            let test_id = test_id.to_string();

            #[expect(clippy::disallowed_methods)]
            tokio::spawn(async move {
                let requests = vec![
                    create_consume_request(
                        &format!("{test_id}_shared_key_{}", i % 5),
                        15,
                        100,
                        10,
                        RateLimitInterval::Minute,
                    ),
                    create_consume_request(
                        &format!("{test_id}_unique_key_{i}"),
                        if i % 3 == 0 { 150 } else { 20 },
                        100,
                        10,
                        RateLimitInterval::Minute,
                    ),
                ];
                conn_clone.consume_tickets(&requests).await.unwrap()
            })
        })
        .collect();

    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Verify atomic behavior: for each result pair, both should have same success status
    for result in results {
        assert_eq!(
            result[0].success, result[1].success,
            "Both keys in a batch should have same success status due to atomic behavior"
        );
    }
}

// ===== RACE CONDITION TESTS =====

pub async fn test_race_condition_no_over_consumption(
    conn: impl RateLimitQueries + Clone + 'static,
    test_id: &str,
) {
    let key = test_key(test_id, "race_test");

    // Launch 50 concurrent requests for 5 tokens each on a bucket with 100 capacity
    let handles: Vec<_> = (0..50)
        .map(|_| {
            let conn_clone = conn.clone();
            let key = key.clone();

            #[expect(clippy::disallowed_methods)]
            tokio::spawn(async move {
                conn_clone
                    .consume_tickets(&[create_consume_request(
                        &key,
                        5,
                        100,
                        10,
                        RateLimitInterval::Minute,
                    )])
                    .await
                    .unwrap()
            })
        })
        .collect();

    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .map(|mut v| v.pop().unwrap())
        .collect();

    let successful = results.iter().filter(|r| r.success).count();
    let total_consumed: u64 = results.iter().map(|r| r.tickets_consumed).sum();

    // Exactly 20 should succeed (20 * 5 = 100), no over-consumption
    assert_eq!(successful, 20, "Exactly 20 requests should succeed");
    assert_eq!(
        total_consumed, 100,
        "Total consumed should exactly equal capacity"
    );

    // Final balance should be 0
    let final_balance = conn
        .get_balance(&key, 100, 10, RateLimitInterval::Minute)
        .await
        .unwrap();
    assert_eq!(
        final_balance, 0,
        "Final balance should be 0 after consuming all tokens"
    );
}

pub async fn test_race_condition_interleaved_consume_return(
    conn: impl RateLimitQueries + Clone + 'static,
    test_id: &str,
) {
    let key = test_key(test_id, "interleaved_test");

    // Set up initial state
    conn.consume_tickets(&[create_consume_request(
        &key,
        50,
        100,
        10,
        RateLimitInterval::Minute,
    )])
    .await
    .unwrap();

    let mut consume_handles = Vec::new();
    let mut return_handles = Vec::new();

    // 15 concurrent consumers requesting 10 each
    for _ in 0..15 {
        let conn_clone = conn.clone();
        let key = key.clone();

        #[expect(clippy::disallowed_methods)]
        let handle = tokio::spawn(async move {
            conn_clone
                .consume_tickets(&[create_consume_request(
                    &key,
                    10,
                    100,
                    10,
                    RateLimitInterval::Minute,
                )])
                .await
                .unwrap()
        });
        consume_handles.push(handle);
    }

    // 10 concurrent returners returning 5 each
    for _ in 0..10 {
        let conn_clone = conn.clone();
        let key = key.clone();

        #[expect(clippy::disallowed_methods)]
        let handle = tokio::spawn(async move {
            conn_clone
                .return_tickets(vec![create_return_request(
                    &key,
                    5,
                    100,
                    10,
                    RateLimitInterval::Minute,
                )])
                .await
                .unwrap()
        });
        return_handles.push(handle);
    }

    // Wait for all operations
    let (_consume_results, _return_results) = futures::future::join(
        futures::future::join_all(consume_handles),
        futures::future::join_all(return_handles),
    )
    .await;

    // Final balance should be consistent and within bounds
    let final_balance = conn
        .get_balance(&key, 100, 10, RateLimitInterval::Minute)
        .await
        .unwrap();
    assert!(
        final_balance <= 100,
        "Balance should not exceed capacity: {final_balance}",
    );
}

// ===== CONSOLIDATED FUNCTIONAL TESTS =====

pub async fn test_rate_limit_lifecycle(conn: impl RateLimitQueries + Clone, test_id: &str) {
    let key = test_key(test_id, "lifecycle_test");

    // Phase 1: Initial consumption
    let results = conn
        .consume_tickets(&[create_consume_request(
            &key,
            60,
            100,
            10,
            RateLimitInterval::Minute,
        )])
        .await
        .unwrap();
    assert!(results[0].success);
    assert_eq!(results[0].tickets_consumed, 60);
    assert_eq!(results[0].tickets_remaining, 40);

    // Phase 2: Check balance
    let balance = conn
        .get_balance(&key, 100, 10, RateLimitInterval::Minute)
        .await
        .unwrap();
    assert_eq!(balance, 40);

    // Phase 3: Partial return
    let results = conn
        .return_tickets(vec![create_return_request(
            &key,
            20,
            100,
            10,
            RateLimitInterval::Minute,
        )])
        .await
        .unwrap();
    assert_eq!(results[0].balance, 60);

    // Phase 4: Consume at new balance
    let results = conn
        .consume_tickets(&[create_consume_request(
            &key,
            60,
            100,
            10,
            RateLimitInterval::Minute,
        )])
        .await
        .unwrap();
    assert!(results[0].success);
    assert_eq!(results[0].tickets_remaining, 0);

    // Phase 5: Should fail when empty
    let results = conn
        .consume_tickets(&[create_consume_request(
            &key,
            1,
            100,
            10,
            RateLimitInterval::Minute,
        )])
        .await
        .unwrap();
    assert!(!results[0].success);
    assert_eq!(results[0].tickets_consumed, 0);
}

pub async fn test_capacity_boundaries(conn: impl RateLimitQueries + Clone, test_id: &str) {
    // Test 1: Zero request (should always succeed)
    let results = conn
        .consume_tickets(&[create_consume_request(
            &test_key(test_id, "zero_test"),
            0,
            50,
            5,
            RateLimitInterval::Minute,
        )])
        .await
        .unwrap();
    assert!(results[0].success);
    assert_eq!(results[0].tickets_consumed, 0);
    assert_eq!(results[0].tickets_remaining, 50);

    // Test 2: Exactly at capacity
    let results = conn
        .consume_tickets(&[create_consume_request(
            &test_key(test_id, "capacity_test"),
            75,
            75,
            5,
            RateLimitInterval::Minute,
        )])
        .await
        .unwrap();
    assert!(results[0].success);
    assert_eq!(results[0].tickets_remaining, 0);

    // Test 3: Exceed capacity (should fail)
    let results = conn
        .consume_tickets(&[create_consume_request(
            &test_key(test_id, "exceed_test"),
            101,
            100,
            5,
            RateLimitInterval::Minute,
        )])
        .await
        .unwrap();
    assert!(!results[0].success);
    assert_eq!(results[0].tickets_consumed, 0);
    assert_eq!(results[0].tickets_remaining, 100);

    // Test 4: Return beyond capacity (should cap)
    let results = conn
        .return_tickets(vec![create_return_request(
            &test_key(test_id, "return_test"),
            150,
            100,
            5,
            RateLimitInterval::Minute,
        )])
        .await
        .unwrap();
    assert_eq!(results[0].balance, 100); // Capped at capacity
}

pub async fn test_refill_mechanics(conn: impl RateLimitQueries + Clone, test_id: &str) {
    let key = test_key(test_id, "refill_test");

    // Phase 1: Consume most tokens
    let results = conn
        .consume_tickets(&[create_consume_request(
            &key,
            40,
            100,
            30,
            RateLimitInterval::Second,
        )])
        .await
        .unwrap();
    assert_eq!(results[0].tickets_remaining, 60);

    // Phase 2: Wait for single refill (1 second + buffer)
    tokio::time::sleep(tokio::time::Duration::from_millis(1100)).await;
    let balance = conn
        .get_balance(&key, 100, 30, RateLimitInterval::Second)
        .await
        .unwrap();
    assert_eq!(balance, 90); // 60 + 30 refill

    // Phase 3: Wait for another refill and verify capping (1 more second)
    tokio::time::sleep(tokio::time::Duration::from_millis(1100)).await;
    let balance = conn
        .get_balance(&key, 100, 30, RateLimitInterval::Second)
        .await
        .unwrap();
    assert_eq!(balance, 100); // Should be capped at capacity
}

pub async fn test_zero_refill_mechanics(conn: impl RateLimitQueries + Clone, test_id: &str) {
    let key = test_key(test_id, "zero_refill");

    conn.consume_tickets(&[create_consume_request(
        &key,
        40,
        100,
        0,
        RateLimitInterval::Second,
    )])
    .await
    .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(1100)).await;
    let balance = conn
        .get_balance(&key, 100, 0, RateLimitInterval::Second)
        .await
        .unwrap();
    assert_eq!(balance, 60); // No refill should occur
}

// ===== EDGE CASES AND EMPTY OPERATIONS =====

pub async fn test_empty_operations(conn: impl RateLimitQueries + Clone, _test_id: &str) {
    // Empty consume requests
    let results = conn.consume_tickets(&[]).await.unwrap();
    assert_eq!(results.len(), 0);

    // Empty return requests
    let results = conn.return_tickets(vec![]).await.unwrap();
    assert_eq!(results.len(), 0);
}

pub async fn test_new_bucket_behavior(conn: impl RateLimitQueries + Clone, test_id: &str) {
    // New bucket starts at capacity
    let balance = conn
        .get_balance(
            &test_key(test_id, "new_bucket"),
            100,
            10,
            RateLimitInterval::Minute,
        )
        .await
        .unwrap();
    assert_eq!(balance, 100);

    // Returning to new bucket creates it at capacity (capped)
    let results = conn
        .return_tickets(vec![create_return_request(
            &test_key(test_id, "return_new"),
            30,
            100,
            10,
            RateLimitInterval::Minute,
        )])
        .await
        .unwrap();
    assert_eq!(results[0].balance, 100); // Created at capacity, return is capped
}

// ===== CONCURRENT STRESS TEST =====

pub async fn test_concurrent_stress(conn: impl RateLimitQueries + Clone + 'static, test_id: &str) {
    // High concurrency test with multiple keys
    let handles: Vec<_> = (0..100)
        .map(|i| {
            let conn_clone = conn.clone();
            let test_id = test_id.to_string();

            #[expect(clippy::disallowed_methods)]
            tokio::spawn(async move {
                conn_clone
                    .consume_tickets(&[create_consume_request(
                        &format!("{test_id}_stress_key_{}", i % 10),
                        15,
                        200,
                        10,
                        RateLimitInterval::Minute,
                    )])
                    .await
                    .unwrap()
            })
        })
        .collect();

    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .map(|mut v| v.pop().unwrap())
        .collect();

    // With 10 keys, each with 200 capacity, and 10 requests of 15 each per key:
    // Each key can handle 13 requests (13 * 15 = 195 < 200, 14 * 15 = 210 > 200)
    // So 10 * 10 = 100 total requests, all should succeed since we have enough capacity
    let successful = results.iter().filter(|r| r.success).count();
    assert_eq!(
        successful, 100,
        "All 100 requests should succeed with sufficient capacity"
    );
}

// ===== INPUT VALIDATION TESTS =====

pub async fn test_consume_tickets_rejects_duplicate_keys(
    conn: impl RateLimitQueries + Clone,
    test_id: &str,
) {
    let key = test_key(test_id, "dup_consume");

    // Attempt to consume with the same key twice in one request
    let result = conn
        .consume_tickets(&[
            create_consume_request(&key, 10, 100, 10, RateLimitInterval::Minute),
            create_consume_request(&key, 20, 100, 10, RateLimitInterval::Minute),
        ])
        .await;

    assert!(
        result.is_err(),
        "consume_tickets should reject duplicate keys"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Duplicate keys"),
        "Error message should mention duplicate keys, got: {err}"
    );
}

pub async fn test_return_tickets_rejects_duplicate_keys(
    conn: impl RateLimitQueries + Clone,
    test_id: &str,
) {
    let key = test_key(test_id, "dup_return");

    // Attempt to return with the same key twice in one request
    let result = conn
        .return_tickets(vec![
            create_return_request(&key, 10, 100, 10, RateLimitInterval::Minute),
            create_return_request(&key, 20, 100, 10, RateLimitInterval::Minute),
        ])
        .await;

    let err = match result {
        Ok(_) => panic!("return_tickets should reject duplicate keys"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("Duplicate keys"),
        "Error message should mention duplicate keys, got: {err}"
    );
}
