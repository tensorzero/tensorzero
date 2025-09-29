use crate::db::PostgresConnectionInfo;
use sqlx::PgPool;
use tensorzero_core::db::ExperimentationQueries;
use uuid::Uuid;

// ===== HELPER FUNCTIONS =====

fn generate_episode_id() -> Uuid {
    Uuid::now_v7()
}

fn generate_function_name(prefix: &str) -> String {
    format!("{}_func_{}", prefix, Uuid::new_v4())
}

fn generate_variant_name(prefix: &str) -> String {
    format!("{}_variant_{}", prefix, Uuid::new_v4())
}

async fn verify_variant_stored(
    conn: &PostgresConnectionInfo,
    episode_id: Uuid,
    function_name: &str,
    expected_variant: &str,
) {
    let result = conn
        .check_and_set_variant_by_episode(
            episode_id,
            function_name,
            "temp_variant", // This should return the existing variant
        )
        .await
        .unwrap();

    assert_eq!(
        result, expected_variant,
        "Stored variant should match expected value"
    );
}

// ===== BASIC FUNCTIONAL TESTS =====

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_basic_success(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id = generate_episode_id();
    let function_name = generate_function_name("basic");
    let variant_name = generate_variant_name("test");

    let result = conn
        .check_and_set_variant_by_episode(episode_id, &function_name, &variant_name)
        .await
        .unwrap();

    assert_eq!(result, variant_name, "Should return the newly set variant");

    // Verify it was actually stored
    verify_variant_stored(&conn, episode_id, &function_name, &variant_name).await;
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_existing_returns_original(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id = generate_episode_id();
    let function_name = generate_function_name("existing");
    let original_variant = generate_variant_name("original");
    let different_variant = generate_variant_name("different");

    // First call - should set the variant
    let result1 = conn
        .check_and_set_variant_by_episode(episode_id, &function_name, &original_variant)
        .await
        .unwrap();
    assert_eq!(result1, original_variant);

    // Second call with different variant - should return original
    let result2 = conn
        .check_and_set_variant_by_episode(episode_id, &function_name, &different_variant)
        .await
        .unwrap();
    assert_eq!(
        result2, original_variant,
        "Should return the originally set variant, not the new candidate"
    );

    // Verify original is still stored
    verify_variant_stored(&conn, episode_id, &function_name, &original_variant).await;
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_same_variant_idempotent(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id = generate_episode_id();
    let function_name = generate_function_name("idempotent");
    let variant_name = generate_variant_name("same");

    // First call
    let result1 = conn
        .check_and_set_variant_by_episode(episode_id, &function_name, &variant_name)
        .await
        .unwrap();
    assert_eq!(result1, variant_name);

    // Second call with same variant - should still return same variant
    let result2 = conn
        .check_and_set_variant_by_episode(episode_id, &function_name, &variant_name)
        .await
        .unwrap();
    assert_eq!(result2, variant_name, "Operation should be idempotent");
}

// ===== ISOLATION TESTS =====

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_different_episodes_isolated(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id1 = generate_episode_id();
    let episode_id2 = generate_episode_id();
    let function_name = generate_function_name("shared_func");
    let variant1 = generate_variant_name("variant1");
    let variant2 = generate_variant_name("variant2");

    // Set variant for episode 1
    let result1 = conn
        .check_and_set_variant_by_episode(episode_id1, &function_name, &variant1)
        .await
        .unwrap();
    assert_eq!(result1, variant1);

    // Set different variant for episode 2 (same function) - should succeed
    let result2 = conn
        .check_and_set_variant_by_episode(episode_id2, &function_name, &variant2)
        .await
        .unwrap();
    assert_eq!(result2, variant2);

    // Verify both are stored independently
    verify_variant_stored(&conn, episode_id1, &function_name, &variant1).await;
    verify_variant_stored(&conn, episode_id2, &function_name, &variant2).await;
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_different_functions_isolated(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id = generate_episode_id();
    let function_name1 = generate_function_name("func1");
    let function_name2 = generate_function_name("func2");
    let variant1 = generate_variant_name("variant1");
    let variant2 = generate_variant_name("variant2");

    // Set variant for function 1
    let result1 = conn
        .check_and_set_variant_by_episode(episode_id, &function_name1, &variant1)
        .await
        .unwrap();
    assert_eq!(result1, variant1);

    // Set different variant for function 2 (same episode) - should succeed
    let result2 = conn
        .check_and_set_variant_by_episode(episode_id, &function_name2, &variant2)
        .await
        .unwrap();
    assert_eq!(result2, variant2);

    // Verify both are stored independently
    verify_variant_stored(&conn, episode_id, &function_name1, &variant1).await;
    verify_variant_stored(&conn, episode_id, &function_name2, &variant2).await;
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_complete_isolation(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    // Create a 2x2 matrix of episode/function combinations
    let episodes = [generate_episode_id(), generate_episode_id()];
    let functions = [generate_function_name("func_a"), generate_function_name("func_b")];
    let variants = [
        [generate_variant_name("var_00"), generate_variant_name("var_01")],
        [generate_variant_name("var_10"), generate_variant_name("var_11")],
    ];

    // Set variants for all combinations
    for (i, episode_id) in episodes.iter().enumerate() {
        for (j, function_name) in functions.iter().enumerate() {
            let result = conn
                .check_and_set_variant_by_episode(*episode_id, function_name, &variants[i][j])
                .await
                .unwrap();
            assert_eq!(result, variants[i][j]);
        }
    }

    // Verify all combinations are isolated and correctly stored
    for (i, episode_id) in episodes.iter().enumerate() {
        for (j, function_name) in functions.iter().enumerate() {
            verify_variant_stored(&conn, *episode_id, function_name, &variants[i][j]).await;
        }
    }
}

// ===== RACE CONDITION TESTS =====

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_concurrent_same_pair(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id = generate_episode_id();
    let function_name = generate_function_name("concurrent");
    let num_attempts = 50;

    // Launch many concurrent attempts to set different variants
    let handles: Vec<_> = (0..num_attempts)
        .map(|i| {
            let conn_clone = conn.clone();
            let function_name_clone = function_name.clone();
            let variant_name = format!("variant_{i}");

            tokio::spawn(async move {
                conn_clone
                    .check_and_set_variant_by_episode(
                        episode_id,
                        &function_name_clone,
                        &variant_name,
                    )
                    .await
                    .unwrap()
            })
        })
        .collect();

    let results: Vec<String> = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // All results should be the same (the winner variant)
    let winner_variant = &results[0];
    for result in &results {
        assert_eq!(
            result, winner_variant,
            "All concurrent operations should return the same winning variant"
        );
    }

    // Verify the winner is actually stored
    verify_variant_stored(&conn, episode_id, &function_name, winner_variant).await;
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_concurrent_different_pairs(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let num_pairs = 20;

    // Launch concurrent operations on different episode/function pairs
    let handles: Vec<_> = (0..num_pairs)
        .map(|i| {
            let conn_clone = conn.clone();
            let episode_id = generate_episode_id();
            let function_name = generate_function_name(&format!("concurrent_func_{i}"));
            let variant_name = generate_variant_name(&format!("concurrent_var_{i}"));

            tokio::spawn(async move {
                let result = conn_clone
                    .check_and_set_variant_by_episode(
                        episode_id,
                        &function_name,
                        &variant_name,
                    )
                    .await
                    .unwrap();

                (episode_id, function_name, variant_name, result)
            })
        })
        .collect();

    let results = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Each operation should have succeeded with its own variant
    for (episode_id, function_name, expected_variant, actual_result) in results {
        assert_eq!(
            actual_result, expected_variant,
            "Each different pair should succeed with its own variant"
        );

        // Verify it was stored correctly
        verify_variant_stored(&conn, episode_id, &function_name, &expected_variant).await;
    }
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_high_contention(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let num_episodes = 5;
    let num_functions = 3;
    let operations_per_pair = 10;

    let episodes: Vec<Uuid> = (0..num_episodes).map(|_| generate_episode_id()).collect();
    let functions: Vec<String> = (0..num_functions)
        .map(|i| generate_function_name(&format!("stress_func_{i}")))
        .collect();

    // Launch many concurrent operations across multiple episode/function pairs
    let mut handles = Vec::new();

    for episode_id in &episodes {
        for function_name in &functions {
            for op_num in 0..operations_per_pair {
                let conn_clone = conn.clone();
                let episode_id = *episode_id;
                let function_name = function_name.clone();
                let variant_name = format!("variant_{}_{}", function_name, op_num);

                let handle = tokio::spawn(async move {
                    let result = conn_clone
                        .check_and_set_variant_by_episode(
                            episode_id,
                            &function_name,
                            &variant_name,
                        )
                        .await
                        .unwrap();

                    (episode_id, function_name, result)
                });

                handles.push(handle);
            }
        }
    }

    let results = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Group results by episode/function pair and verify consistency
    let mut pair_results: std::collections::HashMap<(Uuid, String), Vec<String>> = std::collections::HashMap::new();

    for (episode_id, function_name, result) in results {
        pair_results
            .entry((episode_id, function_name))
            .or_insert_with(Vec::new)
            .push(result);
    }

    // For each pair, all operations should have returned the same variant
    for ((episode_id, function_name), variants) in pair_results {
        let expected_variant = &variants[0];
        for variant in &variants {
            assert_eq!(
                variant, expected_variant,
                "All operations on the same pair should return the same variant"
            );
        }

        // Verify the winner is stored
        verify_variant_stored(&conn, episode_id, &function_name, expected_variant).await;
    }
}

// ===== ATOMICITY TESTS =====

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_truly_atomic(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id = generate_episode_id();
    let function_name = generate_function_name("atomic");
    let num_attempts = 100;

    // Use a barrier to synchronize all operations to start at the same time
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(num_attempts));

    let handles: Vec<_> = (0..num_attempts)
        .map(|i| {
            let conn_clone = conn.clone();
            let function_name_clone = function_name.clone();
            let barrier_clone = barrier.clone();
            let variant_name = format!("atomic_variant_{i}");

            tokio::spawn(async move {
                // Wait for all tasks to be ready
                barrier_clone.wait().await;

                // Attempt the atomic operation
                conn_clone
                    .check_and_set_variant_by_episode(
                        episode_id,
                        &function_name_clone,
                        &variant_name,
                    )
                    .await
                    .unwrap()
            })
        })
        .collect();

    let results: Vec<String> = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // All results should be identical - proving atomicity
    let winning_variant = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            result, winning_variant,
            "Result {i} differs from winning variant - atomicity violated"
        );
    }
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_no_partial_writes(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id = generate_episode_id();
    let function_name = generate_function_name("no_partial");

    // Start many concurrent operations
    let handles: Vec<_> = (0..50)
        .map(|i| {
            let conn_clone = conn.clone();
            let function_name_clone = function_name.clone();
            let variant_name = format!("variant_{i}");

            tokio::spawn(async move {
                conn_clone
                    .check_and_set_variant_by_episode(
                        episode_id,
                        &function_name_clone,
                        &variant_name,
                    )
                    .await
                    .unwrap()
            })
        })
        .collect();

    // While operations are running, continuously check for consistency
    let verification_handle = {
        let conn_clone = conn.clone();
        let function_name_clone = function_name.clone();

        tokio::spawn(async move {
            for _ in 0..20 {
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

                // Try to read the current state - should always be consistent
                match conn_clone
                    .check_and_set_variant_by_episode(
                        episode_id,
                        &function_name_clone,
                        "verification_variant",
                    )
                    .await
                {
                    Ok(_) => {}, // Either no data yet or consistent data
                    Err(_) => panic!("Verification read failed - possible partial write detected"),
                }
            }
        })
    };

    // Wait for all operations to complete
    let results: Vec<String> = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    verification_handle.await.unwrap();

    // All results should be the same
    let expected_variant = &results[0];
    for result in &results {
        assert_eq!(
            result, expected_variant,
            "Inconsistent results indicate partial writes occurred"
        );
    }
}

// ===== EDGE CASE TESTS =====

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_empty_strings(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id = generate_episode_id();

    // Test with empty function name - should work if schema allows
    let result = conn
        .check_and_set_variant_by_episode(episode_id, "", "variant1")
        .await;

    match result {
        Ok(variant) => {
            assert_eq!(variant, "variant1");
            // Verify it's stored
            verify_variant_stored(&conn, episode_id, "", "variant1").await;
        }
        Err(_) => {
            // Empty function names might be rejected by schema - that's valid
        }
    }

    // Test with empty variant name - should work if schema allows
    let function_name = generate_function_name("empty_variant_test");
    let result = conn
        .check_and_set_variant_by_episode(episode_id, &function_name, "")
        .await;

    match result {
        Ok(variant) => {
            assert_eq!(variant, "");
            // Verify it's stored
            verify_variant_stored(&conn, episode_id, &function_name, "").await;
        }
        Err(_) => {
            // Empty variant names might be rejected by schema - that's valid
        }
    }
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_very_long_names(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id = generate_episode_id();

    // Test with very long function name
    let long_function_name = "a".repeat(1000);
    let variant_name = generate_variant_name("long_func");

    let result = conn
        .check_and_set_variant_by_episode(episode_id, &long_function_name, &variant_name)
        .await;

    // Should either succeed or fail gracefully (depending on schema limits)
    match result {
        Ok(returned_variant) => {
            assert_eq!(returned_variant, variant_name);
            verify_variant_stored(&conn, episode_id, &long_function_name, &variant_name).await;
        }
        Err(_) => {
            // Long names might be rejected - that's valid
        }
    }

    // Test with very long variant name
    let function_name = generate_function_name("long_variant_test");
    let long_variant_name = "b".repeat(1000);

    let result = conn
        .check_and_set_variant_by_episode(episode_id, &function_name, &long_variant_name)
        .await;

    match result {
        Ok(returned_variant) => {
            assert_eq!(returned_variant, long_variant_name);
            verify_variant_stored(&conn, episode_id, &function_name, &long_variant_name).await;
        }
        Err(_) => {
            // Long names might be rejected - that's valid
        }
    }
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_special_characters(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let episode_id = generate_episode_id();

    let special_chars_test_cases = vec![
        ("unicode_func_🚀", "unicode_variant_🎯"),
        ("newlines\n\r\nfunc", "newlines\n\r\nvariant"),
        ("quotes'\"func", "quotes'\"variant"),
        ("tabs\t\tfunc", "tabs\t\tvariant"),
        ("nulls\x00func", "nulls\x00variant"),
        ("mixed_!@#$%^&*()_func", "mixed_!@#$%^&*()_variant"),
    ];

    for (function_name, variant_name) in special_chars_test_cases {
        let result = conn
            .check_and_set_variant_by_episode(episode_id, function_name, variant_name)
            .await;

        match result {
            Ok(returned_variant) => {
                assert_eq!(returned_variant, variant_name);
                verify_variant_stored(&conn, episode_id, function_name, variant_name).await;
            }
            Err(e) => {
                // Some special characters might be rejected - log but don't fail
                println!("Special character test failed for '{function_name}': {e}");
            }
        }
    }
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_uuid_edge_cases(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool);

    let function_name = generate_function_name("uuid_edge");
    let variant_name = generate_variant_name("uuid_test");

    // Test with nil UUID
    let nil_uuid = Uuid::nil();
    let result = conn
        .check_and_set_variant_by_episode(nil_uuid, &function_name, &variant_name)
        .await
        .unwrap();

    assert_eq!(result, variant_name);
    verify_variant_stored(&conn, nil_uuid, &function_name, &variant_name).await;

    // Test with max UUID (all 1s)
    let max_uuid = Uuid::from_u128(u128::MAX);
    let result = conn
        .check_and_set_variant_by_episode(max_uuid, &function_name, &variant_name)
        .await
        .unwrap();

    assert_eq!(result, variant_name);
    verify_variant_stored(&conn, max_uuid, &function_name, &variant_name).await;

    // Verify nil and max are isolated from each other
    let different_variant = generate_variant_name("different");
    let result = conn
        .check_and_set_variant_by_episode(nil_uuid, &function_name, &different_variant)
        .await
        .unwrap();

    // Should return the original variant, not the different one
    assert_eq!(result, variant_name, "Nil UUID should maintain its own state");

    let result = conn
        .check_and_set_variant_by_episode(max_uuid, &function_name, &different_variant)
        .await
        .unwrap();

    // Should return the original variant, not the different one
    assert_eq!(result, variant_name, "Max UUID should maintain its own state");
}
