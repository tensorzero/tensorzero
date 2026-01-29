//! Shared test logic for InferenceQueries implementations (ClickHouse and Postgres).
//!
//! Each test function accepts a connection implementing `InferenceQueries`.
//! Tests use `>=` assertions since ClickHouse may accumulate data from other tests.

use std::path::Path;

use tensorzero_core::db::clickhouse::query_builder::InferenceFilter;
use tensorzero_core::db::inferences::{
    CountInferencesForFunctionParams, CountInferencesParams, InferenceQueries,
};
use tensorzero_core::endpoints::stored_inferences::v1::types::{
    BooleanMetricFilter, DemonstrationFeedbackFilter, FloatComparisonOperator, FloatMetricFilter,
};
use tensorzero_core::{
    config::{Config, ConfigFileGlob},
    function::FunctionConfigType,
};

async fn get_e2e_config() -> Config {
    Config::load_from_path_optional_verify_credentials(
        &ConfigFileGlob::new_from_path(Path::new("tests/e2e/config/tensorzero.*.toml")).unwrap(),
        false,
    )
    .await
    .unwrap()
    .into_config_without_writing_for_tests()
}

// ===== SHARED TEST IMPLEMENTATIONS =====
// These tests use InferenceQueries::count_inferences which works with both backends.

async fn test_count_inferences_for_chat_function(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let params = CountInferencesParams {
        function_name: Some("write_haiku"),
        ..Default::default()
    };

    let count = conn.count_inferences(&config, &params).await.unwrap();

    // The fixture data has 804 inferences for write_haiku
    // ClickHouse may have more due to other tests
    assert!(
        count >= 804,
        "Expected at least 804 inferences for write_haiku, got {count}"
    );
}
make_db_test!(test_count_inferences_for_chat_function);

async fn test_count_inferences_for_json_function(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let params = CountInferencesParams {
        function_name: Some("extract_entities"),
        ..Default::default()
    };

    let count = conn.count_inferences(&config, &params).await.unwrap();

    // The fixture data has 604 inferences for extract_entities
    assert!(
        count >= 604,
        "Expected at least 604 inferences for extract_entities, got {count}"
    );
}
make_db_test!(test_count_inferences_for_json_function);

async fn test_count_inferences_for_chat_function_with_variant(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let params = CountInferencesParams {
        function_name: Some("write_haiku"),
        variant_name: Some("initial_prompt_gpt4o_mini"),
        ..Default::default()
    };

    let count = conn.count_inferences(&config, &params).await.unwrap();

    // The fixture data has 649 inferences for write_haiku with variant initial_prompt_gpt4o_mini
    assert!(
        count >= 649,
        "Expected at least 649 inferences for write_haiku/initial_prompt_gpt4o_mini, got {count}"
    );
}
make_db_test!(test_count_inferences_for_chat_function_with_variant);

async fn test_count_inferences_for_json_function_with_variant(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let params = CountInferencesParams {
        function_name: Some("extract_entities"),
        variant_name: Some("gpt4o_initial_prompt"),
        ..Default::default()
    };

    let count = conn.count_inferences(&config, &params).await.unwrap();

    // The fixture data has 132 inferences for extract_entities with variant gpt4o_initial_prompt
    assert!(
        count >= 132,
        "Expected at least 132 inferences for extract_entities/gpt4o_initial_prompt, got {count}"
    );
}
make_db_test!(test_count_inferences_for_json_function_with_variant);

async fn test_count_inferences_for_nonexistent_function(conn: impl InferenceQueries) {
    let params = CountInferencesForFunctionParams {
        function_name: "nonexistent_function",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let count = conn.count_inferences_for_function(params).await.unwrap();

    assert_eq!(count, 0, "Expected 0 for nonexistent function");
}
make_db_test!(test_count_inferences_for_nonexistent_function);

async fn test_count_inferences_for_nonexistent_variant(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let params = CountInferencesParams {
        function_name: Some("write_haiku"),
        variant_name: Some("nonexistent_variant"),
        ..Default::default()
    };

    let count = conn.count_inferences(&config, &params).await.unwrap();

    assert_eq!(count, 0, "Expected 0 for nonexistent variant");
}
make_db_test!(test_count_inferences_for_nonexistent_variant);

// These tests use InferenceQueries methods that provide grouping functionality.

async fn test_count_inferences_by_variant_for_chat_function(conn: impl InferenceQueries + Clone) {
    let config = get_e2e_config().await;
    let params = CountInferencesForFunctionParams {
        function_name: "write_haiku",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let rows = conn.count_inferences_by_variant(params).await.unwrap();

    // There should be at least 2 variants for write_haiku
    assert!(
        rows.len() >= 2,
        "Expected at least 2 variants for write_haiku, got {}",
        rows.len()
    );

    // The sum of all variant counts should match the total count from InferenceQueries
    let total_from_variants: u64 = rows.iter().map(|r| r.inference_count).sum();

    let total_params = CountInferencesParams {
        function_name: Some("write_haiku"),
        ..Default::default()
    };
    let total_count = conn.count_inferences(&config, &total_params).await.unwrap();

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
make_db_test!(test_count_inferences_by_variant_for_chat_function);

async fn test_count_inferences_by_variant_for_json_function(conn: impl InferenceQueries) {
    let params = CountInferencesForFunctionParams {
        function_name: "extract_entities",
        function_type: FunctionConfigType::Json,
        variant_name: None,
    };

    let rows = conn.count_inferences_by_variant(params).await.unwrap();

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
make_db_test!(test_count_inferences_by_variant_for_json_function);

async fn test_count_inferences_by_variant_for_nonexistent_function(conn: impl InferenceQueries) {
    let params = CountInferencesForFunctionParams {
        function_name: "nonexistent_function",
        function_type: FunctionConfigType::Chat,
        variant_name: None,
    };

    let rows = conn.count_inferences_by_variant(params).await.unwrap();

    assert!(
        rows.is_empty(),
        "Expected empty result for nonexistent function"
    );
}
make_db_test!(test_count_inferences_by_variant_for_nonexistent_function);

async fn test_list_functions_with_inference_count(conn: impl InferenceQueries) {
    let rows = conn.list_functions_with_inference_count().await.unwrap();

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

    // Verify inference_counts are reasonable based on test fixtures
    let write_haiku = write_haiku.expect("Chat function write_haiku should be present");
    let extract_entities =
        extract_entities.expect("Json function extract_entities should be present");
    assert!(
        write_haiku.inference_count >= 804,
        "Expected at least 804 inferences for write_haiku, got {}",
        write_haiku.inference_count
    );
    assert!(
        extract_entities.inference_count >= 604,
        "Expected at least 604 inferences for extract_entities, got {}",
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
make_db_test!(test_list_functions_with_inference_count);

// ===== FEEDBACK FILTER TESTS =====
// These tests use InferenceQueries::count_inferences with metric filters.

async fn test_count_feedbacks_for_float_metric(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::FloatMetric(FloatMetricFilter {
        metric_name: "haiku_rating".to_string(),
        value: 0.0,
        comparison_operator: FloatComparisonOperator::GreaterThanOrEqual,
    });
    let params = CountInferencesParams {
        function_name: Some("write_haiku"),
        filters: Some(&filter),
        ..Default::default()
    };

    let count = conn.count_inferences(&config, &params).await.unwrap();

    assert!(
        count > 0,
        "Should have feedbacks for haiku_rating metric on write_haiku"
    );
}
make_db_test!(test_count_feedbacks_for_float_metric);

async fn test_count_feedbacks_for_boolean_metric(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::BooleanMetric(BooleanMetricFilter {
        metric_name: "exact_match".to_string(),
        value: true,
    });
    let params = CountInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter),
        ..Default::default()
    };

    let count = conn.count_inferences(&config, &params).await.unwrap();

    assert!(
        count > 0,
        "Should have feedbacks for exact_match metric on extract_entities"
    );
}
make_db_test!(test_count_feedbacks_for_boolean_metric);

async fn test_count_inferences_with_threshold_float_metric(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    // First get total feedbacks (any haiku_rating >= 0)
    let filter_all = InferenceFilter::FloatMetric(FloatMetricFilter {
        metric_name: "haiku_rating".to_string(),
        value: 0.0,
        comparison_operator: FloatComparisonOperator::GreaterThanOrEqual,
    });
    let params_all = CountInferencesParams {
        function_name: Some("write_haiku"),
        filters: Some(&filter_all),
        ..Default::default()
    };

    let total_feedbacks = conn.count_inferences(&config, &params_all).await.unwrap();

    // Then get with threshold (haiku_rating > 0.5)
    let filter_threshold = InferenceFilter::FloatMetric(FloatMetricFilter {
        metric_name: "haiku_rating".to_string(),
        value: 0.5,
        comparison_operator: FloatComparisonOperator::GreaterThan,
    });
    let params_threshold = CountInferencesParams {
        function_name: Some("write_haiku"),
        filters: Some(&filter_threshold),
        ..Default::default()
    };

    let threshold_count = conn
        .count_inferences(&config, &params_threshold)
        .await
        .unwrap();

    // Threshold count should be < total feedbacks
    assert!(
        threshold_count < total_feedbacks,
        "Threshold count ({threshold_count}) should be < total feedbacks ({total_feedbacks})"
    );
}
make_db_test!(test_count_inferences_with_threshold_float_metric);

async fn test_count_inferences_with_threshold_boolean_metric_max(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    // Get count of exact_match = true
    let filter_true = InferenceFilter::BooleanMetric(BooleanMetricFilter {
        metric_name: "exact_match".to_string(),
        value: true,
    });
    let params_true = CountInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter_true),
        ..Default::default()
    };

    let true_count = conn.count_inferences(&config, &params_true).await.unwrap();

    // Get count of exact_match = false
    let filter_false = InferenceFilter::BooleanMetric(BooleanMetricFilter {
        metric_name: "exact_match".to_string(),
        value: false,
    });
    let params_false = CountInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter_false),
        ..Default::default()
    };

    let false_count = conn.count_inferences(&config, &params_false).await.unwrap();

    // Both should have some counts (showing the filter works)
    assert!(
        true_count > 0,
        "Should have some exact_match = true feedbacks"
    );
    assert!(
        false_count > 0,
        "Should have some exact_match = false feedbacks"
    );
    // true_count should be less than total (true + false)
    assert!(
        true_count < true_count + false_count,
        "true_count ({true_count}) should be < total ({}) since there are false values",
        true_count + false_count
    );
}
make_db_test!(test_count_inferences_with_threshold_boolean_metric_max);

async fn test_count_demonstration_feedbacks(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::DemonstrationFeedback(DemonstrationFeedbackFilter {
        has_demonstration: true,
    });
    let params = CountInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter),
        ..Default::default()
    };

    let count = conn.count_inferences(&config, &params).await.unwrap();

    assert!(count > 0, "Should have demonstrations for extract_entities");
}
make_db_test!(test_count_demonstration_feedbacks);

async fn test_count_feedbacks_for_episode_level_boolean_metric(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::BooleanMetric(BooleanMetricFilter {
        metric_name: "exact_match_episode".to_string(),
        value: true,
    });
    let params = CountInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter),
        ..Default::default()
    };

    let count = conn.count_inferences(&config, &params).await.unwrap();

    // We're verifying the query executes successfully with episode-level join
    assert!(
        count > 0,
        "Should have feedbacks for boolean metric exact_match_episode on extract_entities"
    );
}
make_db_test!(test_count_feedbacks_for_episode_level_boolean_metric);

async fn test_count_feedbacks_for_episode_level_float_metric(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::FloatMetric(FloatMetricFilter {
        metric_name: "jaccard_similarity_episode".to_string(),
        value: 0.0,
        comparison_operator: FloatComparisonOperator::GreaterThanOrEqual,
    });
    let params = CountInferencesParams {
        function_name: Some("extract_entities"),
        filters: Some(&filter),
        ..Default::default()
    };

    let count = conn.count_inferences(&config, &params).await.unwrap();

    // We're verifying the query executes successfully with episode-level join
    assert!(
        count > 0,
        "Should have feedbacks for float metric jaccard_similarity_episode on extract_entities"
    );
}
make_db_test!(test_count_feedbacks_for_episode_level_float_metric);
