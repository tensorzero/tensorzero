use chrono::Duration;
use sqlx::PgPool;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::{RateLimitQueries, ReturnTicketsRequest};
use tensorzero_core::{db::ConsumeTicketsRequest, rate_limiting::ActiveRateLimitKey};

// ===== HELPER FUNCTIONS =====

fn create_consume_request(
    key: &str,
    requested: u64,
    capacity: u64,
    refill_amount: u64,
    refill_interval: Duration,
) -> ConsumeTicketsRequest {
    ConsumeTicketsRequest {
        key: ActiveRateLimitKey(key.to_string()),
        requested,
        capacity,
        refill_amount,
        refill_interval: refill_interval
            .try_into()
            .expect("Failed to convert Duration to PgInterval"),
    }
}

fn create_return_request(
    key: &str,
    returned: u64,
    capacity: u64,
    refill_amount: u64,
    refill_interval: Duration,
) -> ReturnTicketsRequest {
    ReturnTicketsRequest {
        key: ActiveRateLimitKey(key.to_string()),
        returned,
        capacity,
        refill_amount,
        refill_interval: refill_interval
            .try_into()
            .expect("Failed to convert Duration to PgInterval"),
    }
}

// ===== ATOMIC BEHAVIOR TESTS =====

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_atomic_multi_key_all_or_nothing(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    // First, consume some tokens from key1 to set up a scenario where key1 can succeed but key2 fails
    let setup_request = create_consume_request("key1", 50, 100, 10, Duration::seconds(60));
    conn.consume_tickets(&[setup_request]).await.unwrap();

    // Now create a batch where key1 can succeed (50 remaining) but key2 will fail (requesting more than capacity)
    let batch_requests = vec![
        create_consume_request("key1", 30, 100, 10, Duration::seconds(60)), // Should succeed if isolated
        create_consume_request("key2", 150, 100, 10, Duration::seconds(60)), // Will fail - exceeds capacity
    ];

    let results = conn.consume_tickets(&batch_requests).await.unwrap();

    // ALL requests should fail because it's atomic
    assert!(
        !results[0].success,
        "Key1 should fail due to atomic behavior even though it could succeed alone"
    );
    assert!(!results[1].success, "Key2 should fail as expected");

    // Verify key1 balance unchanged - no partial consumption
    let balance = conn
        .get_balance(
            "key1",
            100,
            10,
            Duration::seconds(60)
                .try_into()
                .expect("Failed to convert Duration"),
        )
        .await
        .unwrap();
    assert_eq!(
        balance, 50,
        "Key1 balance should be unchanged due to atomic rollback"
    );
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_atomic_consistency_under_load(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    // Launch many concurrent multi-key requests where some will fail
    let handles: Vec<_> = (0..20)
        .map(|i| {
            let conn_clone = conn.clone();
            // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
            #[expect(clippy::disallowed_methods)]
            tokio::spawn(async move {
                let requests = vec![
                    create_consume_request(
                        &format!("shared_key_{}", i % 5),
                        15,
                        100,
                        10,
                        Duration::seconds(60),
                    ),
                    create_consume_request(
                        &format!("unique_key_{i}"),
                        if i % 3 == 0 { 150 } else { 20 },
                        100,
                        10,
                        Duration::seconds(60),
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

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_race_condition_no_over_consumption(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);
    let key = "race_test";

    // Launch 50 concurrent requests for 5 tokens each on a bucket with 100 capacity
    let handles: Vec<_> = (0..50)
        .map(|_| {
            let conn_clone = conn.clone();
            // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
            #[expect(clippy::disallowed_methods)]
            tokio::spawn(async move {
                let request = create_consume_request(key, 5, 100, 10, Duration::seconds(60));
                conn_clone.consume_tickets(&[request]).await.unwrap()
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
        .get_balance(
            key,
            100,
            10,
            Duration::seconds(60)
                .try_into()
                .expect("Failed to convert Duration"),
        )
        .await
        .unwrap();
    assert_eq!(
        final_balance, 0,
        "Final balance should be 0 after consuming all tokens"
    );
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_race_condition_interleaved_consume_return(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);
    let key = "interleaved_test";

    // Set up initial state
    let setup = create_consume_request(key, 50, 100, 10, Duration::seconds(60));
    conn.consume_tickets(&[setup]).await.unwrap();

    let mut consume_handles = Vec::new();
    let mut return_handles = Vec::new();

    // 15 concurrent consumers requesting 10 each
    for _ in 0..15 {
        let conn_clone = conn.clone();
        // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
        #[expect(clippy::disallowed_methods)]
        let handle = tokio::spawn(async move {
            let request = create_consume_request(key, 10, 100, 10, Duration::seconds(60));
            conn_clone.consume_tickets(&[request]).await.unwrap()
        });
        consume_handles.push(handle);
    }

    // 10 concurrent returners returning 5 each
    for _ in 0..10 {
        let conn_clone = conn.clone();
        // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
        #[expect(clippy::disallowed_methods)]
        let handle = tokio::spawn(async move {
            let request = create_return_request(key, 5, 100, 10, Duration::seconds(60));
            conn_clone.return_tickets(vec![request]).await.unwrap()
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
        .get_balance(
            key,
            100,
            10,
            Duration::seconds(60)
                .try_into()
                .expect("Failed to convert Duration"),
        )
        .await
        .unwrap();
    assert!(
        final_balance <= 100,
        "Balance should not exceed capacity: {final_balance}",
    );
}

// ===== CONSOLIDATED FUNCTIONAL TESTS =====

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_rate_limit_lifecycle(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);
    let key = "lifecycle_test";

    // Phase 1: Initial consumption
    let consume1 = create_consume_request(key, 60, 100, 10, Duration::seconds(60));
    let results = conn.consume_tickets(&[consume1]).await.unwrap();
    assert!(results[0].success);
    assert_eq!(results[0].tickets_consumed, 60);
    assert_eq!(results[0].tickets_remaining, 40);

    // Phase 2: Check balance
    let balance = conn
        .get_balance(
            key,
            100,
            10,
            Duration::seconds(60)
                .try_into()
                .expect("Failed to convert Duration"),
        )
        .await
        .unwrap();
    assert_eq!(balance, 40);

    // Phase 3: Partial return
    let return_req = create_return_request(key, 20, 100, 10, Duration::seconds(60));
    let results = conn.return_tickets(vec![return_req]).await.unwrap();
    assert_eq!(results[0].balance, 60);

    // Phase 4: Consume at new balance
    let consume2 = create_consume_request(key, 60, 100, 10, Duration::seconds(60));
    let results = conn.consume_tickets(&[consume2]).await.unwrap();
    assert!(results[0].success);
    assert_eq!(results[0].tickets_remaining, 0);

    // Phase 5: Should fail when empty
    let consume3 = create_consume_request(key, 1, 100, 10, Duration::seconds(60));
    let results = conn.consume_tickets(&[consume3]).await.unwrap();
    assert!(!results[0].success);
    assert_eq!(results[0].tickets_consumed, 0);
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_capacity_boundaries(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    // Test 1: Zero request (should always succeed)
    let zero_req = create_consume_request("zero_test", 0, 50, 5, Duration::seconds(60));
    let results = conn.consume_tickets(&[zero_req]).await.unwrap();
    assert!(results[0].success);
    assert_eq!(results[0].tickets_consumed, 0);
    assert_eq!(results[0].tickets_remaining, 50);

    // Test 2: Exactly at capacity
    let at_capacity = create_consume_request("capacity_test", 75, 75, 5, Duration::seconds(60));
    let results = conn.consume_tickets(&[at_capacity]).await.unwrap();
    assert!(results[0].success);
    assert_eq!(results[0].tickets_remaining, 0);

    // Test 3: Exceed capacity (should fail)
    let exceed = create_consume_request("exceed_test", 101, 100, 5, Duration::seconds(60));
    let results = conn.consume_tickets(&[exceed]).await.unwrap();
    assert!(!results[0].success);
    assert_eq!(results[0].tickets_consumed, 0);
    assert_eq!(results[0].tickets_remaining, 100);

    // Test 4: Return beyond capacity (should cap)
    let return_beyond = create_return_request("return_test", 150, 100, 5, Duration::seconds(60));
    let results = conn.return_tickets(vec![return_beyond]).await.unwrap();
    assert_eq!(results[0].balance, 100); // Capped at capacity
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_refill_mechanics(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);
    let key = "refill_test";

    // Phase 1: Consume most tokens
    let consume = create_consume_request(key, 80, 100, 30, Duration::milliseconds(100));
    let results = conn.consume_tickets(&[consume]).await.unwrap();
    assert_eq!(results[0].tickets_remaining, 20);

    // Phase 2: Wait for single refill
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
    let balance = conn
        .get_balance(
            key,
            100,
            30,
            Duration::milliseconds(100)
                .try_into()
                .expect("Failed to convert Duration"),
        )
        .await
        .unwrap();
    assert_eq!(balance, 50); // 20 + 30 refill

    // Phase 3: Wait for multiple refills and verify capping
    tokio::time::sleep(tokio::time::Duration::from_millis(300)).await; // 3 more intervals
    let balance = conn
        .get_balance(
            key,
            100,
            30,
            Duration::milliseconds(100)
                .try_into()
                .expect("Failed to convert Duration"),
        )
        .await
        .unwrap();
    assert_eq!(balance, 100); // Should be capped at capacity

    // Phase 4: Test zero refill amount
    let zero_refill_key = "zero_refill";
    let consume_zero =
        create_consume_request(zero_refill_key, 40, 100, 0, Duration::milliseconds(100));
    conn.consume_tickets(&[consume_zero]).await.unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    let balance = conn
        .get_balance(
            zero_refill_key,
            100,
            0,
            Duration::milliseconds(100)
                .try_into()
                .expect("Failed to convert Duration"),
        )
        .await
        .unwrap();
    assert_eq!(balance, 60); // No refill should occur
}

// ===== EDGE CASES AND EMPTY OPERATIONS =====

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_empty_operations(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    // Empty consume requests
    let results = conn.consume_tickets(&[]).await.unwrap();
    assert_eq!(results.len(), 0);

    // Empty return requests
    let results = conn.return_tickets(vec![]).await.unwrap();
    assert_eq!(results.len(), 0);
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_new_bucket_behavior(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    // New bucket starts at capacity
    let balance = conn
        .get_balance(
            "new_bucket",
            100,
            10,
            Duration::seconds(60)
                .try_into()
                .expect("Failed to convert Duration"),
        )
        .await
        .unwrap();
    assert_eq!(balance, 100);

    // Returning to new bucket creates it at capacity (capped)
    let return_req = create_return_request("return_new", 30, 100, 10, Duration::seconds(60));
    let results = conn.return_tickets(vec![return_req]).await.unwrap();
    assert_eq!(results[0].balance, 100); // Created at capacity, return is capped
}

// ===== CONCURRENT STRESS TEST =====

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_concurrent_stress(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    // High concurrency test with multiple keys
    let handles: Vec<_> = (0..100)
        .map(|i| {
            let conn_clone = conn.clone();
            // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
            #[expect(clippy::disallowed_methods)]
            tokio::spawn(async move {
                let key = format!("stress_key_{}", i % 10); // 10 different keys
                let request = create_consume_request(&key, 15, 200, 10, Duration::seconds(60));
                conn_clone.consume_tickets(&[request]).await.unwrap()
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

// ===== INVALID INPUT TESTS =====

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_zero_refill_interval_exception(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    // Test zero interval throws exception
    let zero_interval_request =
        create_consume_request("zero_interval", 10, 100, 10, Duration::zero());
    let result = conn.consume_tickets(&[zero_interval_request]).await;
    assert!(result.is_err(), "Zero interval should cause an error");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Refill interval must be positive"),
        "Error should mention that refill interval must be positive"
    );

    // Test negative interval throws exception
    let negative_interval_request =
        create_consume_request("negative_interval", 10, 100, 10, Duration::seconds(-5));
    let result = conn.consume_tickets(&[negative_interval_request]).await;
    assert!(result.is_err(), "Negative interval should cause an error");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Refill interval must be positive"),
        "Error should mention that refill interval must be positive"
    );
}
