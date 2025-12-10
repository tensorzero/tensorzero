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
