//! E2E tests for inference count Postgres queries.
//!
//! These tests require:
//! 1. TENSORZERO_POSTGRES_URL environment variable set
//! 2. Postgres database with fixtures loaded via ui/fixtures/load_fixtures_postgres.sh

use tensorzero_core::db::inference_count::{CountInferencesParams, InferenceCountQueries};
use tensorzero_core::db::postgres::test_helpers::get_postgres;
use tensorzero_core::function::FunctionConfigType;

#[tokio::test]
async fn test_count_inferences_for_chat_function() {
    let postgres = get_postgres().await;

    let params = CountInferencesParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let count = postgres
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // The fixture data has 804 inferences for write_haiku
    assert_eq!(
        count, 804,
        "Expected 804 inferences for write_haiku, got {count}"
    );
}

#[tokio::test]
async fn test_count_inferences_for_json_function() {
    let postgres = get_postgres().await;

    let params = CountInferencesParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        variant_name: None,
    };

    let count = postgres
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // The fixture data has 604 inferences for extract_entities
    assert_eq!(
        count, 604,
        "Expected 604 inferences for extract_entities, got {count}"
    );
}

#[tokio::test]
async fn test_count_inferences_for_chat_function_with_variant() {
    let postgres = get_postgres().await;

    let params = CountInferencesParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        variant_name: Some("initial_prompt_gpt4o_mini"),
    };

    let count = postgres
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // The fixture data has 649 inferences for write_haiku with variant initial_prompt_gpt4o_mini
    assert_eq!(
        count, 649,
        "Expected 649 inferences for write_haiku/initial_prompt_gpt4o_mini, got {count}"
    );
}

#[tokio::test]
async fn test_count_inferences_for_json_function_with_variant() {
    let postgres = get_postgres().await;

    let params = CountInferencesParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        variant_name: Some("gpt4o_initial_prompt"),
    };

    let count = postgres
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // The fixture data has 132 inferences for extract_entities with variant gpt4o_initial_prompt
    assert_eq!(
        count, 132,
        "Expected 132 inferences for extract_entities/gpt4o_initial_prompt, got {count}"
    );
}

#[tokio::test]
async fn test_count_inferences_for_nonexistent_function() {
    let postgres = get_postgres().await;

    let params = CountInferencesParams {
        function_name: "nonexistent_function",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let count = postgres
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // Should return 0 for nonexistent function
    assert_eq!(count, 0, "Expected 0 for nonexistent function");
}

#[tokio::test]
async fn test_count_inferences_for_nonexistent_variant() {
    let postgres = get_postgres().await;

    let params = CountInferencesParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        variant_name: Some("nonexistent_variant"),
    };

    let count = postgres
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // Should return 0 for nonexistent variant
    assert_eq!(count, 0, "Expected 0 for nonexistent variant");
}

#[tokio::test]
async fn test_count_inferences_by_variant_for_chat_function() {
    let postgres = get_postgres().await;

    let params = CountInferencesParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let rows = postgres.count_inferences_by_variant(params).await.unwrap();

    // There should be exactly 2 variants for write_haiku
    assert_eq!(
        rows.len(),
        2,
        "Expected 2 variants for write_haiku, got {}",
        rows.len()
    );

    // The sum of all variant counts should match the total count
    let total_from_variants: u64 = rows.iter().map(|r| r.inference_count).sum();
    assert_eq!(
        total_from_variants, 804,
        "Sum of variant counts ({total_from_variants}) should equal 804"
    );

    // Results should be ordered by inference_count DESC
    for i in 1..rows.len() {
        assert!(
            rows[i - 1].inference_count >= rows[i].inference_count,
            "Results should be ordered by inference_count DESC"
        );
    }

    // Each row should have a valid last_used timestamp in RFC 3339 format
    let rfc3339_millis_regex =
        regex::Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$").unwrap();
    for row in &rows {
        assert!(
            rfc3339_millis_regex.is_match(&row.last_used_at),
            "last_used should be in RFC 3339 format, got: {} for variant {}",
            row.last_used_at,
            row.variant_name
        );
    }
}

#[tokio::test]
async fn test_count_inferences_by_variant_for_json_function() {
    let postgres = get_postgres().await;

    let params = CountInferencesParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        variant_name: None,
    };

    let rows = postgres.count_inferences_by_variant(params).await.unwrap();

    // There should be at least 2 variants for extract_entities
    assert!(
        rows.len() >= 2,
        "Expected at least 2 variants for extract_entities, got {}",
        rows.len()
    );

    // Verify expected variant is present
    let variant_names: Vec<&str> = rows.iter().map(|r| r.variant_name.as_str()).collect();
    assert!(
        variant_names.contains(&"gpt4o_initial_prompt"),
        "Expected gpt4o_initial_prompt variant to be present"
    );
}

#[tokio::test]
async fn test_count_inferences_by_variant_for_nonexistent_function() {
    let postgres = get_postgres().await;

    let params = CountInferencesParams {
        function_name: "nonexistent_function",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let rows = postgres.count_inferences_by_variant(params).await.unwrap();

    // Should return empty for nonexistent function
    assert!(
        rows.is_empty(),
        "Expected empty result for nonexistent function"
    );
}

// TODO(#5691): test `count_inferences_with_feedback` and `count_inferences_with_demonstration_feedback`

/// Test list_functions_with_inference_count returns expected functions
#[tokio::test]
async fn test_list_functions_with_inference_count() {
    let postgres = get_postgres().await;

    let rows = postgres
        .list_functions_with_inference_count()
        .await
        .unwrap();

    // Should return multiple functions
    assert!(
        rows.len() >= 2,
        "Expected at least 2 functions, got {}",
        rows.len()
    );

    // write_haiku is a chat function, extract_entities is a json function
    // Both should be present in the results
    let write_haiku = rows.iter().find(|r| r.function_name == "write_haiku");
    let extract_entities = rows.iter().find(|r| r.function_name == "extract_entities");

    // Verify inference_counts match expected values from fixtures
    let write_haiku = write_haiku.expect("Chat function write_haiku should be present");
    let extract_entities =
        extract_entities.expect("Json function extract_entities should be present");

    assert_eq!(
        write_haiku.inference_count, 804,
        "Expected 804 inferences for write_haiku, got {}",
        write_haiku.inference_count
    );
    assert_eq!(
        extract_entities.inference_count, 604,
        "Expected 604 inferences for extract_entities, got {}",
        extract_entities.inference_count
    );

    // Results should be ordered by last_inference_timestamp DESC
    for i in 1..rows.len() {
        assert!(
            rows[i - 1].last_inference_timestamp >= rows[i].last_inference_timestamp,
            "Results should be ordered by last_inference_timestamp DESC"
        );
    }

    // Each row should have a positive inference_count
    for row in &rows {
        assert!(
            row.inference_count > 0,
            "Each function should have at least one inference, {} has inference_count {}",
            row.function_name,
            row.inference_count
        );
    }
}
