//! E2E tests for DICLQueries implementations.

use chrono::Utc;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::db::{DICLQueries, StoredDICLExample};
use uuid::Uuid;

// ===== HELPERS =====

fn make_dicl_example(
    function_name: &str,
    variant_name: &str,
    namespace: &str,
    input: &str,
    output: &str,
    embedding: Vec<f32>,
) -> StoredDICLExample {
    StoredDICLExample {
        id: Uuid::now_v7(),
        function_name: function_name.to_string(),
        variant_name: variant_name.to_string(),
        namespace: namespace.to_string(),
        input: input.to_string(),
        output: output.to_string(),
        embedding,
        created_at: Utc::now(),
    }
}

// ===== INSERT + HAS_DICL_EXAMPLES TESTS =====

async fn test_has_dicl_examples_returns_false_for_empty(
    conn: impl DICLQueries + TestDatabaseHelpers,
) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());
    let variant_name = "test_var";

    let exists = conn
        .has_dicl_examples(&function_name, variant_name)
        .await
        .unwrap();
    assert!(
        !exists,
        "Should return false when no examples exist for the function/variant"
    );
}
make_db_test!(test_has_dicl_examples_returns_false_for_empty);

async fn test_insert_and_has_dicl_examples(conn: impl DICLQueries + TestDatabaseHelpers) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());
    let variant_name = "test_var";

    let example = make_dicl_example(
        &function_name,
        variant_name,
        "",
        r#"{"messages":[]}"#,
        "hello",
        vec![1.0, 0.0, 0.0],
    );

    conn.insert_dicl_example(&example).await.unwrap();
    conn.flush_pending_writes().await;

    let exists = conn
        .has_dicl_examples(&function_name, variant_name)
        .await
        .unwrap();
    assert!(
        exists,
        "Should return true after inserting an example for the function/variant"
    );
}
make_db_test!(test_insert_and_has_dicl_examples);

// ===== INSERT_DICL_EXAMPLES (BATCH) =====

async fn test_insert_dicl_examples_batch(conn: impl DICLQueries + TestDatabaseHelpers) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());
    let variant_name = "test_var";

    let examples: Vec<StoredDICLExample> = (0..5)
        .map(|i| {
            make_dicl_example(
                &function_name,
                variant_name,
                "",
                &format!(r#"{{"messages":[{i}]}}"#),
                &format!("output_{i}"),
                vec![i as f32, 0.0, 0.0],
            )
        })
        .collect();

    let rows = conn.insert_dicl_examples(&examples).await.unwrap();
    assert_eq!(rows, 5, "Should report 5 rows inserted");

    conn.flush_pending_writes().await;

    let exists = conn
        .has_dicl_examples(&function_name, variant_name)
        .await
        .unwrap();
    assert!(exists, "Should have examples after batch insert");
}
make_db_test!(test_insert_dicl_examples_batch);

async fn test_insert_dicl_examples_empty(conn: impl DICLQueries + TestDatabaseHelpers) {
    let rows = conn.insert_dicl_examples(&[]).await.unwrap();
    assert_eq!(rows, 0, "Should return 0 for empty batch insert");
}
make_db_test!(test_insert_dicl_examples_empty);

async fn test_insert_dicl_examples_special_characters(
    conn: impl DICLQueries + TestDatabaseHelpers,
) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());
    let variant_name = "test_var";

    // Input/output with quotes, newlines, and unicode characters
    let example = make_dicl_example(
        &function_name,
        variant_name,
        "",
        r#"{"messages":[{"role":"user","content":"input with \"quotes\" and \n newlines"}]}"#,
        "output with special chars: àáâã",
        vec![1.0, 0.0, 0.0],
    );

    conn.insert_dicl_example(&example).await.unwrap();
    conn.flush_pending_writes().await;

    let results = conn
        .get_similar_dicl_examples(&function_name, variant_name, &[1.0, 0.0, 0.0], 1)
        .await
        .unwrap();

    assert_eq!(results.len(), 1, "Should return the inserted example");
    assert!(
        results[0].input.contains(r#"\"quotes\""#),
        "Input should preserve escaped quotes: {}",
        results[0].input
    );
    assert!(
        results[0].input.contains(r"\n"),
        "Input should preserve escaped newlines: {}",
        results[0].input
    );
    assert!(
        results[0].output.contains("àáâã"),
        "Output should preserve unicode characters: {}",
        results[0].output
    );
}
make_db_test!(test_insert_dicl_examples_special_characters);

async fn test_has_dicl_examples_isolates_by_function_and_variant(
    conn: impl DICLQueries + TestDatabaseHelpers,
) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());
    let variant_name = format!("test_var_{}", Uuid::now_v7());
    let other_variant = format!("other_var_{}", Uuid::now_v7());
    let other_function = format!("other_fn_{}", Uuid::now_v7());

    // Insert examples for function_name + variant_name
    let examples = vec![
        make_dicl_example(
            &function_name,
            &variant_name,
            "",
            r#"{"messages":["test"]}"#,
            "output",
            vec![1.0, 0.0, 0.0],
        ),
        make_dicl_example(
            &function_name,
            &variant_name,
            "",
            r#"{"messages":["test2"]}"#,
            "output2",
            vec![0.0, 1.0, 0.0],
        ),
    ];
    conn.insert_dicl_examples(&examples).await.unwrap();
    conn.flush_pending_writes().await;

    // Same function + variant should return true
    let exists = conn
        .has_dicl_examples(&function_name, &variant_name)
        .await
        .unwrap();
    assert!(
        exists,
        "Should return true for the function/variant with inserted examples"
    );

    // Same function + different variant should return false
    let other_variant_exists = conn
        .has_dicl_examples(&function_name, &other_variant)
        .await
        .unwrap();
    assert!(
        !other_variant_exists,
        "Should return false for a different variant name"
    );

    // Different function + same variant should return false
    let other_function_exists = conn
        .has_dicl_examples(&other_function, &variant_name)
        .await
        .unwrap();
    assert!(
        !other_function_exists,
        "Should return false for a different function name"
    );
}
make_db_test!(test_has_dicl_examples_isolates_by_function_and_variant);

// ===== GET_SIMILAR_DICL_EXAMPLES TESTS =====

async fn test_get_similar_dicl_examples_returns_sorted_by_distance(
    conn: impl DICLQueries + TestDatabaseHelpers,
) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());
    let variant_name = "test_var";

    // Insert examples with known embeddings
    // Query embedding will be [1, 0, 0], so cosine distance to [1,0,0] < [0.7,0.7,0] < [0,1,0]
    let examples = vec![
        make_dicl_example(
            &function_name,
            variant_name,
            "",
            r#"{"messages":["close"]}"#,
            "close_output",
            vec![1.0, 0.0, 0.0],
        ),
        make_dicl_example(
            &function_name,
            variant_name,
            "",
            r#"{"messages":["mid"]}"#,
            "mid_output",
            vec![0.7, 0.7, 0.0],
        ),
        make_dicl_example(
            &function_name,
            variant_name,
            "",
            r#"{"messages":["far"]}"#,
            "far_output",
            vec![0.0, 1.0, 0.0],
        ),
    ];

    conn.insert_dicl_examples(&examples).await.unwrap();
    conn.flush_pending_writes().await;

    let results = conn
        .get_similar_dicl_examples(&function_name, variant_name, &[1.0, 0.0, 0.0], 3)
        .await
        .unwrap();

    assert_eq!(results.len(), 3, "Should return all 3 examples");
    assert_eq!(
        results[0].output, "close_output",
        "Closest example should be first (identical embedding)"
    );
    assert_eq!(
        results[2].output, "far_output",
        "Farthest example should be last (orthogonal embedding)"
    );

    // Verify distances are in ascending order
    for i in 1..results.len() {
        assert!(
            results[i].cosine_distance >= results[i - 1].cosine_distance,
            "Results should be sorted by cosine distance ascending"
        );
    }
}
make_db_test!(test_get_similar_dicl_examples_returns_sorted_by_distance);

async fn test_get_similar_dicl_examples_respects_limit(
    conn: impl DICLQueries + TestDatabaseHelpers,
) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());
    let variant_name = "test_var";

    let examples: Vec<StoredDICLExample> = (0..5)
        .map(|i| {
            make_dicl_example(
                &function_name,
                variant_name,
                "",
                &format!(r#"{{"messages":[{i}]}}"#),
                &format!("output_{i}"),
                vec![i as f32, 1.0, 0.0],
            )
        })
        .collect();

    conn.insert_dicl_examples(&examples).await.unwrap();
    conn.flush_pending_writes().await;

    let results = conn
        .get_similar_dicl_examples(&function_name, variant_name, &[1.0, 0.0, 0.0], 2)
        .await
        .unwrap();

    assert_eq!(results.len(), 2, "Should respect the limit of 2");
}
make_db_test!(test_get_similar_dicl_examples_respects_limit);

async fn test_get_similar_dicl_examples_filters_by_function_and_variant(
    conn: impl DICLQueries + TestDatabaseHelpers,
) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());
    let other_function = format!("other_fn_{}", Uuid::now_v7());
    let variant_name = "test_var";

    // Insert for our function
    let example = make_dicl_example(
        &function_name,
        variant_name,
        "",
        r#"{"messages":["ours"]}"#,
        "our_output",
        vec![1.0, 0.0, 0.0],
    );
    conn.insert_dicl_example(&example).await.unwrap();

    // Insert for a different function
    let other_example = make_dicl_example(
        &other_function,
        variant_name,
        "",
        r#"{"messages":["other"]}"#,
        "other_output",
        vec![1.0, 0.0, 0.0],
    );
    conn.insert_dicl_example(&other_example).await.unwrap();

    conn.flush_pending_writes().await;

    let results = conn
        .get_similar_dicl_examples(&function_name, variant_name, &[1.0, 0.0, 0.0], 10)
        .await
        .unwrap();

    assert_eq!(
        results.len(),
        1,
        "Should only return examples matching the function name"
    );
    assert_eq!(results[0].output, "our_output");
}
make_db_test!(test_get_similar_dicl_examples_filters_by_function_and_variant);

async fn test_get_similar_dicl_examples_empty_result(conn: impl DICLQueries + TestDatabaseHelpers) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());

    let results = conn
        .get_similar_dicl_examples(&function_name, "nonexistent", &[1.0, 0.0], 5)
        .await
        .unwrap();

    assert!(
        results.is_empty(),
        "Should return empty vec when no examples match"
    );
}
make_db_test!(test_get_similar_dicl_examples_empty_result);

// ===== DELETE_DICL_EXAMPLES TESTS =====

async fn test_delete_dicl_examples_all_for_variant(conn: impl DICLQueries + TestDatabaseHelpers) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());
    let variant_name = "test_var";

    let examples: Vec<StoredDICLExample> = (0..3)
        .map(|i| {
            make_dicl_example(
                &function_name,
                variant_name,
                &format!("ns_{i}"),
                &format!(r#"{{"messages":[{i}]}}"#),
                &format!("output_{i}"),
                vec![i as f32, 0.0, 0.0],
            )
        })
        .collect();

    conn.insert_dicl_examples(&examples).await.unwrap();
    conn.flush_pending_writes().await;

    // Delete all examples for this function/variant (no namespace filter)
    let deleted = conn
        .delete_dicl_examples(&function_name, variant_name, None)
        .await
        .unwrap();
    assert_eq!(deleted, 3, "Should report deleting all 3 examples");

    conn.flush_pending_writes().await;

    let exists = conn
        .has_dicl_examples(&function_name, variant_name)
        .await
        .unwrap();
    assert!(
        !exists,
        "Should have no examples after deleting all for the variant"
    );
}
make_db_test!(test_delete_dicl_examples_all_for_variant);

async fn test_delete_dicl_examples_by_namespace(conn: impl DICLQueries + TestDatabaseHelpers) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());
    let variant_name = "test_var";

    // Insert examples in two different namespaces
    let ns_a = make_dicl_example(
        &function_name,
        variant_name,
        "namespace_a",
        r#"{"messages":["a"]}"#,
        "output_a",
        vec![1.0, 0.0, 0.0],
    );
    let ns_b = make_dicl_example(
        &function_name,
        variant_name,
        "namespace_b",
        r#"{"messages":["b"]}"#,
        "output_b",
        vec![0.0, 1.0, 0.0],
    );

    conn.insert_dicl_examples(&[ns_a, ns_b]).await.unwrap();
    conn.flush_pending_writes().await;

    // Delete only namespace_a
    let deleted = conn
        .delete_dicl_examples(&function_name, variant_name, Some("namespace_a"))
        .await
        .unwrap();
    assert_eq!(
        deleted, 1,
        "Should report deleting 1 example from namespace_a"
    );

    conn.flush_pending_writes().await;

    // namespace_b examples should still exist
    let exists = conn
        .has_dicl_examples(&function_name, variant_name)
        .await
        .unwrap();
    assert!(
        exists,
        "Should still have examples from namespace_b after deleting namespace_a"
    );

    // Query should return only the namespace_b example
    let results = conn
        .get_similar_dicl_examples(&function_name, variant_name, &[0.0, 1.0, 0.0], 10)
        .await
        .unwrap();
    assert_eq!(
        results.len(),
        1,
        "Should have exactly 1 example remaining after namespace delete"
    );
    assert_eq!(results[0].output, "output_b");
}
make_db_test!(test_delete_dicl_examples_by_namespace);

async fn test_delete_dicl_examples_returns_zero_when_none_exist(
    conn: impl DICLQueries + TestDatabaseHelpers,
) {
    let function_name = format!("test_fn_{}", Uuid::now_v7());

    let deleted = conn
        .delete_dicl_examples(&function_name, "nonexistent", None)
        .await
        .unwrap();
    assert_eq!(deleted, 0, "Should return 0 when no examples match");
}
make_db_test!(test_delete_dicl_examples_returns_zero_when_none_exist);
