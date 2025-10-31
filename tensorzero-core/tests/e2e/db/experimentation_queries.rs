use sqlx::PgPool;
use std::time::Instant;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::ExperimentationQueries;
use uuid::Uuid;

// ===== HELPER FUNCTIONS =====

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
    // TODO: once we implement a read for the variant_by_episode table we should
    // replace this with something simpler.
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

// ===== CONSOLIDATED TESTS =====

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_basic_functionality(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    let episode_id = Uuid::now_v7();
    let function_name = generate_function_name("basic");
    let variant_name = generate_variant_name("test");
    let different_variant = generate_variant_name("different");

    // Test 1: Basic success - first call should set the variant
    let result = conn
        .check_and_set_variant_by_episode(episode_id, &function_name, &variant_name)
        .await
        .unwrap();
    assert_eq!(result, variant_name, "Should return the newly set variant");

    // Test 2: Existing returns original - second call with different variant should return original
    let result = conn
        .check_and_set_variant_by_episode(episode_id, &function_name, &different_variant)
        .await
        .unwrap();
    assert_eq!(
        result, variant_name,
        "Should return the originally set variant"
    );

    // Test 3: Idempotent - same variant should still work
    let result = conn
        .check_and_set_variant_by_episode(episode_id, &function_name, &variant_name)
        .await
        .unwrap();
    assert_eq!(result, variant_name, "Operation should be idempotent");

    // Verify final state
    verify_variant_stored(&conn, episode_id, &function_name, &variant_name).await;
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_isolation(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    // Create comprehensive test matrix: 3 episodes × 3 functions = 9 combinations
    let episodes: Vec<Uuid> = (0..3).map(|_| Uuid::now_v7()).collect();
    let functions: Vec<String> = (0..3)
        .map(|i| generate_function_name(&format!("isolation_{i}")))
        .collect();

    // Create unique variants for each combination
    let mut expected_variants = Vec::new();
    for i in 0..3 {
        let mut row = Vec::new();
        for j in 0..3 {
            row.push(generate_variant_name(&format!("var_{i}_{j}")));
        }
        expected_variants.push(row);
    }

    // Set variants for all combinations
    for (i, &episode_id) in episodes.iter().enumerate() {
        for (j, function_name) in functions.iter().enumerate() {
            let result = conn
                .check_and_set_variant_by_episode(
                    episode_id,
                    function_name,
                    &expected_variants[i][j],
                )
                .await
                .unwrap();
            assert_eq!(result, expected_variants[i][j]);
        }
    }

    // Verify isolation: each combination should maintain its unique variant
    for (i, &episode_id) in episodes.iter().enumerate() {
        for (j, function_name) in functions.iter().enumerate() {
            verify_variant_stored(&conn, episode_id, function_name, &expected_variants[i][j]).await;
        }
    }

    // Test cross-contamination: trying to set different variants should return existing ones
    for (i, &episode_id) in episodes.iter().enumerate() {
        for (j, function_name) in functions.iter().enumerate() {
            let different_variant = generate_variant_name("contamination_test");
            let result = conn
                .check_and_set_variant_by_episode(episode_id, function_name, &different_variant)
                .await
                .unwrap();
            assert_eq!(
                result, expected_variants[i][j],
                "Should return original variant, not contamination attempt"
            );
        }
    }
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_concurrency_and_atomicity(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    let episode_id = Uuid::now_v7();
    let function_name = generate_function_name("concurrency");
    let num_attempts = 100;

    // Use barrier to ensure maximum contention
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(num_attempts));

    let handles: Vec<_> = (0..num_attempts)
        .map(|i| {
            let conn_clone = conn.clone();
            let function_name_clone = function_name.clone();
            let barrier_clone = barrier.clone();
            let variant_name = format!("concurrent_variant_{i}");

            // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
            #[expect(clippy::disallowed_methods)]
            tokio::spawn(async move {
                // Wait for all tasks to be ready for maximum contention
                barrier_clone.wait().await;

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

    // Verify atomicity: all results must be identical
    let winning_variant = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            result, winning_variant,
            "Result {i} differs from winning variant - atomicity violated at high contention"
        );
    }

    // Verify the winner is actually stored
    verify_variant_stored(&conn, episode_id, &function_name, winning_variant).await;
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_edge_case_values(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    let episode_id = Uuid::now_v7();

    // Create long strings first to avoid temporary value issues
    let long_function_name = "a".repeat(1000);
    let long_variant_name = "b".repeat(1000);

    // Test cases: (function_name, variant_name, should_succeed, description)
    let test_cases = vec![
        // Empty strings
        ("", "variant1", None, "empty function name"),
        ("func1", "", None, "empty variant name"),
        // Very long names
        (
            &long_function_name,
            "variant2",
            None,
            "very long function name",
        ),
        ("func2", &long_variant_name, None, "very long variant name"),
        // Special characters
        (
            "unicode_func_test",
            "unicode_variant_test",
            Some("unicode_variant_test"),
            "unicode characters",
        ),
        (
            "newlines\n\r\nfunc",
            "newlines\n\r\nvariant",
            Some("newlines\n\r\nvariant"),
            "newline characters",
        ),
        (
            "quotes'\"func",
            "quotes'\"variant",
            Some("quotes'\"variant"),
            "quote characters",
        ),
        (
            "tabs\t\tfunc",
            "tabs\t\tvariant",
            Some("tabs\t\tvariant"),
            "tab characters",
        ),
        (
            "mixed_!@#$%^&*()_func",
            "mixed_!@#$%^&*()_variant",
            Some("mixed_!@#$%^&*()_variant"),
            "mixed special characters",
        ),
    ];

    for (function_name, variant_name, expected_success, description) in test_cases {
        let result = conn
            .check_and_set_variant_by_episode(episode_id, function_name, variant_name)
            .await;

        match (result, expected_success) {
            (Ok(returned_variant), Some(expected)) => {
                assert_eq!(
                    returned_variant, expected,
                    "Mismatch for test case: {description}"
                );
                verify_variant_stored(&conn, episode_id, function_name, expected).await;
            }
            (Ok(returned_variant), None) => {
                // Unexpectedly succeeded - verify it worked correctly
                assert_eq!(
                    returned_variant, variant_name,
                    "Unexpected success for: {description}"
                );
                verify_variant_stored(&conn, episode_id, function_name, variant_name).await;
            }
            (Err(_), Some(_)) => {
                panic!("Expected success but got error for test case: {description}");
            }
            (Err(_), None) => {
                // Expected failure - this is fine
                // Edge case correctly rejected
            }
        }
    }

    // UUID edge cases
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

    // Test with max UUID
    let max_uuid = Uuid::from_u128(u128::MAX);
    let result = conn
        .check_and_set_variant_by_episode(max_uuid, &function_name, &variant_name)
        .await
        .unwrap();
    assert_eq!(result, variant_name);
    verify_variant_stored(&conn, max_uuid, &function_name, &variant_name).await;
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_stress_test(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    let num_episodes = 5;
    let num_functions = 4;
    let operations_per_pair = 20;
    let _total_operations = num_episodes * num_functions * operations_per_pair;

    let episodes: Vec<Uuid> = (0..num_episodes).map(|_| Uuid::now_v7()).collect();
    let functions: Vec<String> = (0..num_functions)
        .map(|i| generate_function_name(&format!("stress_func_{i}")))
        .collect();

    // Starting stress test with total_operations
    let start_time = Instant::now();

    // Launch all operations concurrently
    let mut handles = Vec::new();

    for (ep_idx, &episode_id) in episodes.iter().enumerate() {
        for (fn_idx, function_name) in functions.iter().enumerate() {
            for op_num in 0..operations_per_pair {
                let conn_clone = conn.clone();
                let function_name = function_name.clone();
                let variant_name = format!("stress_var_{ep_idx}_{fn_idx}_{op_num}");

                // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
                #[expect(clippy::disallowed_methods)]
                let handle = tokio::spawn(async move {
                    let result = conn_clone
                        .check_and_set_variant_by_episode(episode_id, &function_name, &variant_name)
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

    let _duration = start_time.elapsed();
    // Stress test completed

    // Group and verify results
    let mut pair_results: std::collections::HashMap<(Uuid, String), Vec<String>> =
        std::collections::HashMap::new();

    for (episode_id, function_name, result) in results {
        pair_results
            .entry((episode_id, function_name))
            .or_default()
            .push(result);
    }

    // Verify each pair has consistent results
    assert_eq!(
        pair_results.len(),
        num_episodes * num_functions,
        "Should have results for all pairs"
    );

    for ((episode_id, function_name), variants) in pair_results {
        assert_eq!(
            variants.len(),
            operations_per_pair,
            "Should have all operations for pair"
        );

        let expected_variant = &variants[0];
        for variant in &variants {
            assert_eq!(
                variant, expected_variant,
                "All operations on same pair should return same variant"
            );
        }

        verify_variant_stored(&conn, episode_id, &function_name, expected_variant).await;
    }
}

#[sqlx::test(migrations = "src/db/postgres/migrations")]
async fn test_cas_failure_recovery(pool: PgPool) {
    let conn = PostgresConnectionInfo::new_with_pool(pool, None);

    let episode_id = Uuid::now_v7();
    let function_name = generate_function_name("failure_recovery");
    let original_variant = generate_variant_name("original");

    // Set initial variant
    let result = conn
        .check_and_set_variant_by_episode(episode_id, &function_name, &original_variant)
        .await
        .unwrap();
    assert_eq!(result, original_variant);

    // Simulate recovery scenario: multiple attempts should all return the same value
    let recovery_attempts = 10;
    for i in 0..recovery_attempts {
        let attempt_variant = format!("recovery_attempt_{i}");
        let result = conn
            .check_and_set_variant_by_episode(episode_id, &function_name, &attempt_variant)
            .await
            .unwrap();
        assert_eq!(
            result, original_variant,
            "Recovery attempt {i} should return original variant"
        );
    }

    // Final verification
    verify_variant_stored(&conn, episode_id, &function_name, &original_variant).await;
}
