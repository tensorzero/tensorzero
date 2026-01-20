use tensorzero_core::db::valkey::ValkeyConnectionInfo;
use tensorzero_core::db::{ConsumeTicketsRequest, RateLimitQueries, ReturnTicketsRequest};
use tensorzero_core::rate_limiting::{ActiveRateLimitKey, RateLimitInterval};
use uuid::Uuid;

async fn create_valkey_client() -> ValkeyConnectionInfo {
    let url =
        std::env::var("TENSORZERO_VALKEY_URL").unwrap_or_else(|_| "redis://localhost:6379".into());
    ValkeyConnectionInfo::new(&url)
        .await
        .expect("Failed to connect to Valkey")
}

fn unique_key(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4())
}

// ===== ATOMIC BEHAVIOR TESTS =====

#[tokio::test]
async fn test_atomic_multi_key_all_or_nothing() {
    let conn = create_valkey_client().await;
    let key1 = unique_key("atomic_key1");
    let key2 = unique_key("atomic_key2");

    // First, consume some tokens from key1 to set up a scenario where key1 can succeed but key2 fails
    conn.consume_tickets(&[ConsumeTicketsRequest {
        key: ActiveRateLimitKey(key1.clone()),
        requested: 50,
        capacity: 100,
        refill_amount: 10,
        refill_interval: RateLimitInterval::Minute,
    }])
    .await
    .unwrap();

    // Now create a batch where key1 can succeed (50 remaining) but key2 will fail (requesting more than capacity)
    let results = conn
        .consume_tickets(&[
            ConsumeTicketsRequest {
                key: ActiveRateLimitKey(key1.clone()),
                requested: 30,
                capacity: 100,
                refill_amount: 10,
                refill_interval: RateLimitInterval::Minute,
            },
            ConsumeTicketsRequest {
                key: ActiveRateLimitKey(key2),
                requested: 150, // Exceeds capacity - will fail
                capacity: 100,
                refill_amount: 10,
                refill_interval: RateLimitInterval::Minute,
            },
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

#[tokio::test]
async fn test_atomic_consistency_under_load() {
    let conn = create_valkey_client().await;
    let test_id = Uuid::new_v4();

    // Launch many concurrent multi-key requests where some will fail
    let handles: Vec<_> = (0..20)
        .map(|i| {
            let conn_clone = conn.clone();

            #[expect(clippy::disallowed_methods)]
            tokio::spawn(async move {
                let requests = vec![
                    ConsumeTicketsRequest {
                        key: ActiveRateLimitKey(format!("shared_key_{test_id}_{}", i % 5)),
                        requested: 15,
                        capacity: 100,
                        refill_amount: 10,
                        refill_interval: RateLimitInterval::Minute,
                    },
                    ConsumeTicketsRequest {
                        key: ActiveRateLimitKey(format!("unique_key_{test_id}_{i}")),
                        requested: if i % 3 == 0 { 150 } else { 20 },
                        capacity: 100,
                        refill_amount: 10,
                        refill_interval: RateLimitInterval::Minute,
                    },
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

#[tokio::test]
async fn test_race_condition_no_over_consumption() {
    let conn = create_valkey_client().await;
    let key = unique_key("race_test");

    // Launch 50 concurrent requests for 5 tokens each on a bucket with 100 capacity
    let handles: Vec<_> = (0..50)
        .map(|_| {
            let conn_clone = conn.clone();
            let key = key.clone();

            #[expect(clippy::disallowed_methods)]
            tokio::spawn(async move {
                conn_clone
                    .consume_tickets(&[ConsumeTicketsRequest {
                        key: ActiveRateLimitKey(key),
                        requested: 5,
                        capacity: 100,
                        refill_amount: 10,
                        refill_interval: RateLimitInterval::Minute,
                    }])
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

#[tokio::test]
async fn test_race_condition_interleaved_consume_return() {
    let conn = create_valkey_client().await;
    let key = unique_key("interleaved_test");

    // Set up initial state
    conn.consume_tickets(&[ConsumeTicketsRequest {
        key: ActiveRateLimitKey(key.clone()),
        requested: 50,
        capacity: 100,
        refill_amount: 10,
        refill_interval: RateLimitInterval::Minute,
    }])
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
                .consume_tickets(&[ConsumeTicketsRequest {
                    key: ActiveRateLimitKey(key),
                    requested: 10,
                    capacity: 100,
                    refill_amount: 10,
                    refill_interval: RateLimitInterval::Minute,
                }])
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
                .return_tickets(vec![ReturnTicketsRequest {
                    key: ActiveRateLimitKey(key),
                    returned: 5,
                    capacity: 100,
                    refill_amount: 10,
                    refill_interval: RateLimitInterval::Minute,
                }])
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

#[tokio::test]
async fn test_rate_limit_lifecycle() {
    let conn = create_valkey_client().await;
    let key = unique_key("lifecycle_test");

    // Phase 1: Initial consumption
    let results = conn
        .consume_tickets(&[ConsumeTicketsRequest {
            key: ActiveRateLimitKey(key.clone()),
            requested: 60,
            capacity: 100,
            refill_amount: 10,
            refill_interval: RateLimitInterval::Minute,
        }])
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
        .return_tickets(vec![ReturnTicketsRequest {
            key: ActiveRateLimitKey(key.clone()),
            returned: 20,
            capacity: 100,
            refill_amount: 10,
            refill_interval: RateLimitInterval::Minute,
        }])
        .await
        .unwrap();
    assert_eq!(results[0].balance, 60);

    // Phase 4: Consume at new balance
    let results = conn
        .consume_tickets(&[ConsumeTicketsRequest {
            key: ActiveRateLimitKey(key.clone()),
            requested: 60,
            capacity: 100,
            refill_amount: 10,
            refill_interval: RateLimitInterval::Minute,
        }])
        .await
        .unwrap();
    assert!(results[0].success);
    assert_eq!(results[0].tickets_remaining, 0);

    // Phase 5: Should fail when empty
    let results = conn
        .consume_tickets(&[ConsumeTicketsRequest {
            key: ActiveRateLimitKey(key),
            requested: 1,
            capacity: 100,
            refill_amount: 10,
            refill_interval: RateLimitInterval::Minute,
        }])
        .await
        .unwrap();
    assert!(!results[0].success);
    assert_eq!(results[0].tickets_consumed, 0);
}

#[tokio::test]
async fn test_capacity_boundaries() {
    let conn = create_valkey_client().await;

    // Test 1: Zero request (should always succeed)
    let results = conn
        .consume_tickets(&[ConsumeTicketsRequest {
            key: ActiveRateLimitKey(unique_key("zero_test")),
            requested: 0,
            capacity: 50,
            refill_amount: 5,
            refill_interval: RateLimitInterval::Minute,
        }])
        .await
        .unwrap();
    assert!(results[0].success);
    assert_eq!(results[0].tickets_consumed, 0);
    assert_eq!(results[0].tickets_remaining, 50);

    // Test 2: Exactly at capacity
    let results = conn
        .consume_tickets(&[ConsumeTicketsRequest {
            key: ActiveRateLimitKey(unique_key("capacity_test")),
            requested: 75,
            capacity: 75,
            refill_amount: 5,
            refill_interval: RateLimitInterval::Minute,
        }])
        .await
        .unwrap();
    assert!(results[0].success);
    assert_eq!(results[0].tickets_remaining, 0);

    // Test 3: Exceed capacity (should fail)
    let results = conn
        .consume_tickets(&[ConsumeTicketsRequest {
            key: ActiveRateLimitKey(unique_key("exceed_test")),
            requested: 101,
            capacity: 100,
            refill_amount: 5,
            refill_interval: RateLimitInterval::Minute,
        }])
        .await
        .unwrap();
    assert!(!results[0].success);
    assert_eq!(results[0].tickets_consumed, 0);
    assert_eq!(results[0].tickets_remaining, 100);

    // Test 4: Return beyond capacity (should cap)
    let results = conn
        .return_tickets(vec![ReturnTicketsRequest {
            key: ActiveRateLimitKey(unique_key("return_test")),
            returned: 150,
            capacity: 100,
            refill_amount: 5,
            refill_interval: RateLimitInterval::Minute,
        }])
        .await
        .unwrap();
    assert_eq!(results[0].balance, 100); // Capped at capacity
}

#[tokio::test]
async fn test_refill_mechanics() {
    let conn = create_valkey_client().await;
    let key = unique_key("refill_test");

    // Phase 1: Consume most tokens
    let results = conn
        .consume_tickets(&[ConsumeTicketsRequest {
            key: ActiveRateLimitKey(key.clone()),
            requested: 40,
            capacity: 100,
            refill_amount: 30,
            refill_interval: RateLimitInterval::Second,
        }])
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

#[tokio::test]
async fn test_zero_refill_mechanics() {
    let conn = create_valkey_client().await;
    let key = unique_key("zero_refill");

    conn.consume_tickets(&[ConsumeTicketsRequest {
        key: ActiveRateLimitKey(key.clone()),
        requested: 40,
        capacity: 100,
        refill_amount: 0,
        refill_interval: RateLimitInterval::Second,
    }])
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

#[tokio::test]
async fn test_empty_operations() {
    let conn = create_valkey_client().await;

    // Empty consume requests
    let results = conn.consume_tickets(&[]).await.unwrap();
    assert_eq!(results.len(), 0);

    // Empty return requests
    let results = conn.return_tickets(vec![]).await.unwrap();
    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn test_new_bucket_behavior() {
    let conn = create_valkey_client().await;

    // New bucket starts at capacity
    let balance = conn
        .get_balance(
            &unique_key("new_bucket"),
            100,
            10,
            RateLimitInterval::Minute,
        )
        .await
        .unwrap();
    assert_eq!(balance, 100);

    // Returning to new bucket creates it at capacity (capped)
    let results = conn
        .return_tickets(vec![ReturnTicketsRequest {
            key: ActiveRateLimitKey(unique_key("return_new")),
            returned: 30,
            capacity: 100,
            refill_amount: 10,
            refill_interval: RateLimitInterval::Minute,
        }])
        .await
        .unwrap();
    assert_eq!(results[0].balance, 100); // Created at capacity, return is capped
}

// ===== CONCURRENT STRESS TEST =====

#[tokio::test]
async fn test_concurrent_stress() {
    let conn = create_valkey_client().await;
    let test_id = Uuid::new_v4();

    // High concurrency test with multiple keys
    let handles: Vec<_> = (0..100)
        .map(|i| {
            let conn_clone = conn.clone();

            #[expect(clippy::disallowed_methods)]
            tokio::spawn(async move {
                conn_clone
                    .consume_tickets(&[ConsumeTicketsRequest {
                        key: ActiveRateLimitKey(format!("stress_key_{test_id}_{}", i % 10)),
                        requested: 15,
                        capacity: 200,
                        refill_amount: 10,
                        refill_interval: RateLimitInterval::Minute,
                    }])
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

#[tokio::test]
async fn test_consume_tickets_rejects_duplicate_keys() {
    let conn = create_valkey_client().await;
    let key = unique_key("dup_consume");

    // Attempt to consume with the same key twice in one request
    let result = conn
        .consume_tickets(&[
            ConsumeTicketsRequest {
                key: ActiveRateLimitKey(key.clone()),
                requested: 10,
                capacity: 100,
                refill_amount: 10,
                refill_interval: RateLimitInterval::Minute,
            },
            ConsumeTicketsRequest {
                key: ActiveRateLimitKey(key.clone()),
                requested: 20,
                capacity: 100,
                refill_amount: 10,
                refill_interval: RateLimitInterval::Minute,
            },
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

#[tokio::test]
async fn test_return_tickets_rejects_duplicate_keys() {
    let conn = create_valkey_client().await;
    let key = unique_key("dup_return");

    // Attempt to return with the same key twice in one request
    let result = conn
        .return_tickets(vec![
            ReturnTicketsRequest {
                key: ActiveRateLimitKey(key.clone()),
                returned: 10,
                capacity: 100,
                refill_amount: 10,
                refill_interval: RateLimitInterval::Minute,
            },
            ReturnTicketsRequest {
                key: ActiveRateLimitKey(key.clone()),
                returned: 20,
                capacity: 100,
                refill_amount: 10,
                refill_interval: RateLimitInterval::Minute,
            },
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
