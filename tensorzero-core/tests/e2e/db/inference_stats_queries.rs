//! E2E tests for inference statistics ClickHouse queries.

use tensorzero_core::db::clickhouse::{
    inference_stats::CountInferencesParams, test_helpers::get_clickhouse,
};
use tensorzero_core::function::FunctionConfigType;

#[tokio::test]
async fn test_count_inferences_for_chat_function() {
    let clickhouse = get_clickhouse().await;

    let params = CountInferencesParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let count = clickhouse
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // The test data has at least 804 inferences for write_haiku (base fixture data)
    // The count may be higher due to other tests adding more inferences
    assert!(
        count >= 804,
        "Expected at least 804 inferences for write_haiku, got {count}"
    );
}

#[tokio::test]
async fn test_count_inferences_for_json_function() {
    let clickhouse = get_clickhouse().await;

    let params = CountInferencesParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        variant_name: None,
    };

    let count = clickhouse
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // The test data has at least 604 inferences for extract_entities (base fixture data)
    // The count may be higher due to other tests adding more inferences
    assert!(
        count >= 604,
        "Expected at least 604 inferences for extract_entities, got {count}"
    );
}

#[tokio::test]
async fn test_count_inferences_for_chat_function_with_variant() {
    let clickhouse = get_clickhouse().await;

    let params = CountInferencesParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        variant_name: Some("initial_prompt_gpt4o_mini"),
    };

    let count = clickhouse
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // The test data has at least 649 inferences for write_haiku with variant initial_prompt_gpt4o_mini
    assert!(
        count >= 649,
        "Expected at least 649 inferences for write_haiku/initial_prompt_gpt4o_mini, got {count}"
    );
}

#[tokio::test]
async fn test_count_inferences_for_json_function_with_variant() {
    let clickhouse = get_clickhouse().await;

    let params = CountInferencesParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        variant_name: Some("gpt4o_initial_prompt"),
    };

    let count = clickhouse
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // The test data has at least 132 inferences for extract_entities with variant gpt4o_initial_prompt
    assert!(
        count >= 132,
        "Expected at least 132 inferences for extract_entities/gpt4o_initial_prompt, got {count}"
    );
}

#[tokio::test]
async fn test_count_inferences_for_nonexistent_function() {
    let clickhouse = get_clickhouse().await;

    let params = CountInferencesParams {
        function_name: "nonexistent_function",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let count = clickhouse
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // Should return 0 for nonexistent function
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_count_inferences_for_nonexistent_variant() {
    let clickhouse = get_clickhouse().await;

    let params = CountInferencesParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        variant_name: Some("nonexistent_variant"),
    };

    let count = clickhouse
        .count_inferences_for_function(params)
        .await
        .unwrap();

    // Should return 0 for nonexistent variant
    assert_eq!(count, 0);
}
