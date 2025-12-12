//! E2E tests for inference statistics ClickHouse queries.

use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::inference_stats::{
    CountInferencesParams, CountInferencesWithDemonstrationFeedbacksParams,
    CountInferencesWithFeedbackParams, InferenceStatsQueries,
};
use tensorzero_core::{
    config::{MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType},
    function::FunctionConfigType,
};

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

#[tokio::test]
async fn test_count_inferences_by_variant_for_chat_function() {
    let clickhouse = get_clickhouse().await;

    let params = CountInferencesParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let rows = clickhouse
        .count_inferences_by_variant(params)
        .await
        .unwrap();

    // There should be at least 2 variants for write_haiku
    assert!(
        rows.len() >= 2,
        "Expected at least 2 variants for write_haiku, got {}",
        rows.len()
    );

    // The sum of all variant counts should match the total count
    let total_from_variants: u64 = rows.iter().map(|r| r.inference_count).sum();

    let total_params = CountInferencesParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };
    let total_count = clickhouse
        .count_inferences_for_function(total_params)
        .await
        .unwrap();

    assert_eq!(
        total_from_variants, total_count,
        "Sum of variant counts ({total_from_variants}) should equal total count ({total_count})"
    );

    // Results should be ordered by inference_count DESC
    for i in 1..rows.len() {
        assert!(
            rows[i - 1].inference_count >= rows[i].inference_count,
            "Results should be ordered by inference_count DESC"
        );
    }

    // Each row should have a valid last_used timestamp
    for row in &rows {
        assert!(
            !row.last_used_at.is_empty(),
            "last_used should not be empty for variant {}",
            row.variant_name
        );
        // Verify it's a valid ISO 8601 format
        assert!(
            row.last_used_at.contains('T') && row.last_used_at.ends_with('Z'),
            "last_used should be in ISO 8601 format, got: {}",
            row.last_used_at
        );
    }
}

#[tokio::test]
async fn test_count_inferences_by_variant_for_json_function() {
    let clickhouse = get_clickhouse().await;

    let params = CountInferencesParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        variant_name: None,
    };

    let rows = clickhouse
        .count_inferences_by_variant(params)
        .await
        .unwrap();

    // There should be at least 2 variants for extract_entities
    assert!(
        rows.len() >= 2,
        "Expected at least 2 variants for extract_entities, got {}",
        rows.len()
    );

    // Verify expected variants are present
    let variant_names: Vec<&str> = rows.iter().map(|r| r.variant_name.as_str()).collect();
    assert!(
        variant_names.contains(&"gpt4o_initial_prompt"),
        "Expected gpt4o_initial_prompt variant to be present"
    );
}

#[tokio::test]
async fn test_count_inferences_by_variant_for_nonexistent_function() {
    let clickhouse = get_clickhouse().await;

    let params = CountInferencesParams {
        function_name: "nonexistent_function",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let rows = clickhouse
        .count_inferences_by_variant(params)
        .await
        .unwrap();

    // Should return empty for nonexistent function
    assert!(rows.is_empty());
}

/// Test counting feedbacks for a float metric
#[tokio::test]
async fn test_count_feedbacks_for_float_metric() {
    let clickhouse = get_clickhouse().await;

    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
    };

    let params = CountInferencesWithFeedbackParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        metric_name: "haiku_rating",
        metric_config: &metric_config,
        metric_threshold: None,
    };

    let count = clickhouse
        .count_inferences_with_feedback(params)
        .await
        .unwrap();

    // The test database should have some haiku_rating feedbacks
    assert!(
        count > 0,
        "Should have feedbacks for haiku_rating metric on write_haiku"
    );
}

/// Test counting feedbacks for a boolean metric
#[tokio::test]
async fn test_count_feedbacks_for_boolean_metric() {
    let clickhouse = get_clickhouse().await;

    let metric_config = MetricConfig {
        r#type: MetricConfigType::Boolean,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
    };

    let params = CountInferencesWithFeedbackParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        metric_name: "exact_match",
        metric_config: &metric_config,
        metric_threshold: None,
    };

    // Query executes successfully, count may be 0 if no feedbacks exist
    let count = clickhouse
        .count_inferences_with_feedback(params)
        .await
        .unwrap();

    assert!(
        count > 0,
        "Should have feedbacks for exact_match metric on extract_entities"
    );
}

/// Test counting inferences with feedback meeting threshold for a float metric
#[tokio::test]
async fn test_count_inferences_with_threshold_float_metric() {
    let clickhouse = get_clickhouse().await;

    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
    };

    // First get total feedbacks
    let total_params = CountInferencesWithFeedbackParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        metric_name: "haiku_rating",
        metric_config: &metric_config,
        metric_threshold: None,
    };

    let total_feedbacks = clickhouse
        .count_inferences_with_feedback(total_params)
        .await
        .unwrap();

    // Then get with threshold
    let threshold_params = CountInferencesWithFeedbackParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        metric_name: "haiku_rating",
        metric_config: &metric_config,
        metric_threshold: Some(0.5),
    };

    let threshold_count = clickhouse
        .count_inferences_with_feedback(threshold_params)
        .await
        .unwrap();

    // Threshold count should be <= total feedbacks
    assert!(
        threshold_count < total_feedbacks,
        "Threshold count should be < total feedbacks"
    );
}

/// Test counting inferences with feedback meeting threshold for a boolean metric (optimize max)
#[tokio::test]
async fn test_count_inferences_with_threshold_boolean_metric_max() {
    let clickhouse = get_clickhouse().await;

    let metric_config = MetricConfig {
        r#type: MetricConfigType::Boolean,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
    };

    // First get total feedbacks
    let total_params = CountInferencesWithFeedbackParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        metric_name: "exact_match",
        metric_config: &metric_config,
        metric_threshold: None,
    };

    let total_feedbacks = clickhouse
        .count_inferences_with_feedback(total_params)
        .await
        .unwrap();

    // Then get with threshold (value = 1 for max optimization)
    let threshold_params = CountInferencesWithFeedbackParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        metric_name: "exact_match",
        metric_config: &metric_config,
        metric_threshold: Some(0.0), // Not used for boolean, but required
    };

    let threshold_count = clickhouse
        .count_inferences_with_feedback(threshold_params)
        .await
        .unwrap();

    // Threshold count should be <= total feedbacks
    assert!(
        threshold_count < total_feedbacks,
        "Threshold count should be < total feedbacks"
    );
}

/// Test counting demonstration feedbacks for a function
#[tokio::test]
async fn test_count_demonstration_feedbacks() {
    let clickhouse = get_clickhouse().await;

    let params = CountInferencesWithDemonstrationFeedbacksParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
    };

    let count = clickhouse
        .count_inferences_with_demonstration_feedback(params)
        .await
        .unwrap();

    assert!(count > 0, "Should have demonstrations for extract_entities");
}

/// Test with episode-level metric
#[tokio::test]
async fn test_count_feedbacks_for_episode_level_boolean_metric() {
    let clickhouse = get_clickhouse().await;

    let metric_config = MetricConfig {
        r#type: MetricConfigType::Boolean,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Episode,
    };

    let params = CountInferencesWithFeedbackParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        metric_name: "exact_match_episode",
        metric_config: &metric_config,
        metric_threshold: None,
    };

    let count = clickhouse
        .count_inferences_with_feedback(params)
        .await
        .unwrap();

    // We're verifying the query executes successfully with episode-level join
    assert!(
        count > 0,
        "Should have feedbacks for boolean metric exact_match_episode on extract_entities"
    );
}

#[tokio::test]
async fn test_count_feedbacks_for_episode_level_float_metric() {
    let clickhouse = get_clickhouse().await;

    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Episode,
    };

    let params = CountInferencesWithFeedbackParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        metric_name: "jaccard_similarity_episode",
        metric_config: &metric_config,
        metric_threshold: None,
    };

    let count = clickhouse
        .count_inferences_with_feedback(params)
        .await
        .unwrap();

    // We're verifying the query executes successfully with episode-level join
    assert!(
        count > 0,
        "Should have feedbacks for float metric jaccard_similarity_episode on extract_entities"
    );
}
