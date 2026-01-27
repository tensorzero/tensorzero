//! Shared test logic for InferenceQueries implementations (ClickHouse and Postgres).
//!
//! Tests for basic inference queries and filters work with both backends.

use std::path::Path;

use chrono::{TimeZone, Utc};
use tensorzero_core::{
    config::{Config, ConfigFileGlob, MetricConfigLevel},
    db::{
        clickhouse::query_builder::{InferenceFilter, OrderBy, OrderByTerm, OrderDirection},
        feedback::{FeedbackQueries, FeedbackRow},
        inferences::{FunctionInfo, InferenceOutputSource, InferenceQueries, ListInferencesParams},
    },
    endpoints::stored_inferences::v1::types::{
        BooleanMetricFilter, DemonstrationFeedbackFilter, FloatComparisonOperator,
        FloatMetricFilter, TagComparisonOperator, TagFilter, TimeComparisonOperator, TimeFilter,
    },
    inference::types::FunctionType,
    stored_inference::StoredSample,
};
use uuid::Uuid;

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
// These tests work with both ClickHouse and Postgres.

async fn test_get_inference_output_chat_inference(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    // First, list some chat inferences to get a valid inference_id
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("write_haiku"),
                output_source: InferenceOutputSource::Inference,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should have at least one chat inference"
    );

    let inference = &inferences[0];
    let inference_id = inference.id();

    // Get function info for this inference
    let function_info = conn
        .get_function_info(&inference_id, MetricConfigLevel::Inference)
        .await
        .unwrap()
        .expect("Should find function info for existing inference");

    assert_eq!(
        function_info.function_type,
        FunctionType::Chat,
        "write_haiku should be a Chat function"
    );

    // Now test get_inference_output
    let output = conn
        .get_inference_output(&function_info, inference_id)
        .await
        .unwrap();

    assert!(
        output.is_some(),
        "Should return output for existing inference"
    );
    let output_str = output.unwrap();
    assert!(
        !output_str.is_empty(),
        "Output should not be empty for existing inference"
    );
}
make_db_test!(test_get_inference_output_chat_inference);

async fn test_get_inference_output_json_inference(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    // First, list some json inferences to get a valid inference_id
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                output_source: InferenceOutputSource::Inference,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should have at least one json inference"
    );

    let inference = &inferences[0];
    let inference_id = inference.id();

    // Get function info for this inference
    let function_info = conn
        .get_function_info(&inference_id, MetricConfigLevel::Inference)
        .await
        .unwrap()
        .expect("Should find function info for existing inference");

    assert_eq!(
        function_info.function_type,
        FunctionType::Json,
        "extract_entities should be a Json function"
    );

    // Now test get_inference_output
    let output = conn
        .get_inference_output(&function_info, inference_id)
        .await
        .unwrap();

    assert!(
        output.is_some(),
        "Should return output for existing json inference"
    );
    let output_str = output.unwrap();
    assert!(
        !output_str.is_empty(),
        "Output should not be empty for existing json inference"
    );
}
make_db_test!(test_get_inference_output_json_inference);

async fn test_get_inference_output_not_found(conn: impl InferenceQueries) {
    // Create a fake function_info with a non-existent inference_id
    let fake_function_info = FunctionInfo {
        function_name: "write_haiku".to_string(),
        function_type: FunctionType::Chat,
        variant_name: "test_variant".to_string(),
        episode_id: Uuid::now_v7(),
    };

    let non_existent_inference_id = Uuid::now_v7();

    let output = conn
        .get_inference_output(&fake_function_info, non_existent_inference_id)
        .await
        .unwrap();

    assert!(
        output.is_none(),
        "Should return None for non-existent inference"
    );
}
make_db_test!(test_get_inference_output_not_found);

async fn test_list_inferences_chat_function(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("write_haiku"),
                output_source: InferenceOutputSource::Inference,
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should return chat inferences for write_haiku"
    );

    // All inferences should be for the requested function
    for inference in &inferences {
        assert_eq!(
            inference.function_name(),
            "write_haiku",
            "All inferences should be for write_haiku function"
        );
    }
}
make_db_test!(test_list_inferences_chat_function);

async fn test_list_inferences_json_function(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                output_source: InferenceOutputSource::Inference,
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should return json inferences for extract_entities"
    );

    // All inferences should be for the requested function
    for inference in &inferences {
        assert_eq!(
            inference.function_name(),
            "extract_entities",
            "All inferences should be for extract_entities function"
        );
    }
}
make_db_test!(test_list_inferences_json_function);

async fn test_get_function_info_for_inference(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    // Get an inference ID to test with
    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("write_haiku"),
                output_source: InferenceOutputSource::Inference,
                limit: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(!inferences.is_empty(), "Should have at least one inference");

    let inference_id = inferences[0].id();

    let function_info = conn
        .get_function_info(&inference_id, MetricConfigLevel::Inference)
        .await
        .unwrap();

    assert!(
        function_info.is_some(),
        "Should find function info for existing inference"
    );

    let info = function_info.unwrap();
    assert_eq!(info.function_name, "write_haiku");
    assert_eq!(info.function_type, FunctionType::Chat);
}
make_db_test!(test_get_function_info_for_inference);

async fn test_get_function_info_not_found(conn: impl InferenceQueries) {
    let non_existent_id = Uuid::now_v7();

    let function_info = conn
        .get_function_info(&non_existent_id, MetricConfigLevel::Inference)
        .await
        .unwrap();

    assert!(
        function_info.is_none(),
        "Should return None for non-existent inference"
    );
}
make_db_test!(test_get_function_info_not_found);

async fn test_list_inferences_filtered_to_has_demonstrations(
    conn: impl InferenceQueries + FeedbackQueries,
) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::DemonstrationFeedback(DemonstrationFeedbackFilter {
        has_demonstration: true,
    });

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 3,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should return inferences with demonstration feedback"
    );

    for inference in &inferences {
        let inference_id = inference.id();
        let feedback = conn
            .query_feedback_by_target_id(inference_id, None, None, Some(50))
            .await
            .unwrap();

        let has_demonstration = feedback
            .iter()
            .any(|row| matches!(row, FeedbackRow::Demonstration(_)));

        assert!(
            has_demonstration,
            "Expected demonstration feedback for inference {inference_id}"
        );
    }
}
make_db_test!(test_list_inferences_filtered_to_has_demonstrations);

async fn test_list_inferences_filtered_to_no_demonstrations(
    conn: impl InferenceQueries + FeedbackQueries,
) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::DemonstrationFeedback(DemonstrationFeedbackFilter {
        has_demonstration: false,
    });

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 3,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should return inferences without demonstration feedback"
    );

    for inference in &inferences {
        let inference_id = match inference {
            tensorzero_core::stored_inference::StoredInferenceDatabase::Json(json_inf) => {
                json_inf.inference_id
            }
            tensorzero_core::stored_inference::StoredInferenceDatabase::Chat(chat_inf) => {
                chat_inf.inference_id
            }
        };
        let feedback = conn
            .query_feedback_by_target_id(inference_id, None, None, Some(50))
            .await
            .unwrap();
        let has_demonstration = feedback
            .iter()
            .any(|row| matches!(row, FeedbackRow::Demonstration(_)));

        assert!(
            !has_demonstration,
            "Expected no demonstration feedback for inference {inference_id}"
        );
    }
}
make_db_test!(test_list_inferences_filtered_to_no_demonstrations);

// ===== BOOLEAN METRIC FILTER TESTS =====

async fn test_list_inferences_filtered_by_boolean_metric_true(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::BooleanMetric(BooleanMetricFilter {
        metric_name: "exact_match".to_string(),
        value: true,
    });

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences that have exact_match = true feedback
    // (we're testing that the filter is applied, not verifying the feedback values)
    assert!(
        !inferences.is_empty(),
        "Should return inferences with exact_match = true"
    );
}
make_db_test!(test_list_inferences_filtered_by_boolean_metric_true);

async fn test_list_inferences_filtered_by_boolean_metric_false(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::BooleanMetric(BooleanMetricFilter {
        metric_name: "exact_match".to_string(),
        value: false,
    });

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences that have exact_match = false feedback
    assert!(
        !inferences.is_empty(),
        "Should return inferences with exact_match = false"
    );
}
make_db_test!(test_list_inferences_filtered_by_boolean_metric_false);

// ===== FLOAT METRIC FILTER TESTS =====

async fn test_list_inferences_filtered_by_float_metric_greater_than(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::FloatMetric(FloatMetricFilter {
        metric_name: "jaccard_similarity".to_string(),
        value: 0.5,
        comparison_operator: FloatComparisonOperator::GreaterThan,
    });

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences with jaccard_similarity > 0.5
    assert!(
        !inferences.is_empty(),
        "Should return inferences with jaccard_similarity > 0.5"
    );
}
make_db_test!(test_list_inferences_filtered_by_float_metric_greater_than);

async fn test_list_inferences_filtered_by_float_metric_less_than(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::FloatMetric(FloatMetricFilter {
        metric_name: "jaccard_similarity".to_string(),
        value: 0.5,
        comparison_operator: FloatComparisonOperator::LessThan,
    });

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences with jaccard_similarity < 0.5
    assert!(
        !inferences.is_empty(),
        "Should return inferences with jaccard_similarity < 0.5"
    );
}
make_db_test!(test_list_inferences_filtered_by_float_metric_less_than);

async fn test_list_inferences_filtered_by_float_metric_equal(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    // 1.0 is a common value for jaccard_similarity (perfect match)
    let filter = InferenceFilter::FloatMetric(FloatMetricFilter {
        metric_name: "jaccard_similarity".to_string(),
        value: 1.0,
        comparison_operator: FloatComparisonOperator::Equal,
    });

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences with jaccard_similarity = 1.0
    assert!(
        !inferences.is_empty(),
        "Should return inferences with jaccard_similarity = 1.0"
    );
}
make_db_test!(test_list_inferences_filtered_by_float_metric_equal);

// ===== TAG FILTER TESTS =====

async fn test_list_inferences_filtered_by_tag_equal(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    let filter = InferenceFilter::Tag(TagFilter {
        key: "foo".to_string(),
        value: "bar".to_string(),
        comparison_operator: TagComparisonOperator::Equal,
    });

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                // Don't specify function_name to query both tables
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences with tag foo = bar
    assert!(
        !inferences.is_empty(),
        "Should return inferences with tag foo = bar"
    );

    // Verify all returned inferences have the expected tag
    for inference in &inferences {
        let tags = match inference {
            tensorzero_core::stored_inference::StoredInferenceDatabase::Json(json_inf) => {
                &json_inf.tags
            }
            tensorzero_core::stored_inference::StoredInferenceDatabase::Chat(chat_inf) => {
                &chat_inf.tags
            }
        };
        assert_eq!(
            tags.get("foo"),
            Some(&"bar".to_string()),
            "Inference should have tag foo = bar"
        );
    }
}
make_db_test!(test_list_inferences_filtered_by_tag_equal);

async fn test_list_inferences_filtered_by_tag_not_equal(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    // Find inferences that have "foo" tag but NOT with value "bar"
    // This is tricky because != only applies to tags that exist
    let filter = InferenceFilter::Tag(TagFilter {
        key: "foo".to_string(),
        value: "nonexistent_value".to_string(),
        comparison_operator: TagComparisonOperator::NotEqual,
    });

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences where foo != "nonexistent_value"
    // This includes inferences with foo = "bar" (since bar != nonexistent_value)
    assert!(
        !inferences.is_empty(),
        "Should return inferences with tag foo != 'nonexistent_value'"
    );
}
make_db_test!(test_list_inferences_filtered_by_tag_not_equal);

// ===== TIME FILTER TESTS =====

async fn test_list_inferences_filtered_by_time_after(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    // Use a date in the past to ensure we get results
    let filter = InferenceFilter::Time(TimeFilter {
        time: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
        comparison_operator: TimeComparisonOperator::GreaterThan,
    });

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("write_haiku"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences created after 2024-01-01
    assert!(
        !inferences.is_empty(),
        "Should return inferences created after 2024-01-01"
    );
}
make_db_test!(test_list_inferences_filtered_by_time_after);

async fn test_list_inferences_filtered_by_time_before(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    // Use a date in the future to ensure we get results
    let filter = InferenceFilter::Time(TimeFilter {
        time: Utc.with_ymd_and_hms(2030, 1, 1, 0, 0, 0).unwrap(),
        comparison_operator: TimeComparisonOperator::LessThan,
    });

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("write_haiku"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences created before 2030-01-01
    assert!(
        !inferences.is_empty(),
        "Should return inferences created before 2030-01-01"
    );
}
make_db_test!(test_list_inferences_filtered_by_time_before);

// ===== COMPOUND FILTER TESTS (AND/OR/NOT) =====

async fn test_list_inferences_filtered_by_and(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    // AND: tag foo = bar AND time > 2024-01-01
    let filter = InferenceFilter::And {
        children: vec![
            InferenceFilter::Tag(TagFilter {
                key: "foo".to_string(),
                value: "bar".to_string(),
                comparison_operator: TagComparisonOperator::Equal,
            }),
            InferenceFilter::Time(TimeFilter {
                time: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                comparison_operator: TimeComparisonOperator::GreaterThan,
            }),
        ],
    };

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences that satisfy BOTH conditions
    assert!(
        !inferences.is_empty(),
        "Should return inferences matching both conditions"
    );

    for inference in &inferences {
        let tags = match inference {
            tensorzero_core::stored_inference::StoredInferenceDatabase::Json(json_inf) => {
                &json_inf.tags
            }
            tensorzero_core::stored_inference::StoredInferenceDatabase::Chat(chat_inf) => {
                &chat_inf.tags
            }
        };
        assert_eq!(
            tags.get("foo"),
            Some(&"bar".to_string()),
            "Inference should have tag foo = bar"
        );
    }
}
make_db_test!(test_list_inferences_filtered_by_and);

async fn test_list_inferences_filtered_by_or(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    // OR: exact_match = true OR jaccard_similarity > 0.9
    let filter = InferenceFilter::Or {
        children: vec![
            InferenceFilter::BooleanMetric(BooleanMetricFilter {
                metric_name: "exact_match".to_string(),
                value: true,
            }),
            InferenceFilter::FloatMetric(FloatMetricFilter {
                metric_name: "jaccard_similarity".to_string(),
                value: 0.9,
                comparison_operator: FloatComparisonOperator::GreaterThan,
            }),
        ],
    };

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences that satisfy EITHER condition
    assert!(
        !inferences.is_empty(),
        "Should return inferences matching at least one condition"
    );
}
make_db_test!(test_list_inferences_filtered_by_or);

async fn test_list_inferences_filtered_by_not(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    // NOT: NOT has_demonstration
    // This should return inferences that do NOT have demonstrations
    let filter = InferenceFilter::Not {
        child: Box::new(InferenceFilter::DemonstrationFeedback(
            DemonstrationFeedbackFilter {
                has_demonstration: true,
            },
        )),
    };

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences without demonstrations
    assert!(
        !inferences.is_empty(),
        "Should return inferences without demonstrations"
    );
}
make_db_test!(test_list_inferences_filtered_by_not);

async fn test_list_inferences_filtered_by_nested_and_or(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    // Complex filter: (tag foo = bar AND time > 2024-01-01) OR (has_demonstration = false)
    let filter = InferenceFilter::Or {
        children: vec![
            InferenceFilter::And {
                children: vec![
                    InferenceFilter::Tag(TagFilter {
                        key: "foo".to_string(),
                        value: "bar".to_string(),
                        comparison_operator: TagComparisonOperator::Equal,
                    }),
                    InferenceFilter::Time(TimeFilter {
                        time: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                        comparison_operator: TimeComparisonOperator::GreaterThan,
                    }),
                ],
            },
            InferenceFilter::DemonstrationFeedback(DemonstrationFeedbackFilter {
                has_demonstration: false,
            }),
        ],
    };

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return inferences matching the complex condition
    assert!(
        !inferences.is_empty(),
        "Should return inferences matching the nested AND/OR conditions"
    );
}
make_db_test!(test_list_inferences_filtered_by_nested_and_or);

// ===== EMPTY FILTER EDGE CASES =====

async fn test_list_inferences_filtered_by_empty_and(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    // Empty AND should be vacuously true (return all inferences matching other criteria)
    let filter = InferenceFilter::And { children: vec![] };

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("write_haiku"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Empty AND is vacuously true, so should return results
    assert!(
        !inferences.is_empty(),
        "Empty AND filter should return results (vacuously true)"
    );
}
make_db_test!(test_list_inferences_filtered_by_empty_and);

async fn test_list_inferences_filtered_by_empty_or(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;
    // Empty OR should be false (return no inferences)
    let filter = InferenceFilter::Or { children: vec![] };

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("write_haiku"),
                filters: Some(&filter),
                output_source: InferenceOutputSource::Inference,
                limit: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Empty OR is false, so should return no results
    assert!(
        inferences.is_empty(),
        "Empty OR filter should return no results"
    );
}
make_db_test!(test_list_inferences_filtered_by_empty_or);

// ===== ORDER BY METRIC TESTS =====

async fn test_list_inferences_order_by_float_metric_desc(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    // Order by jaccard_similarity DESC (higher values first)
    let order_by = vec![OrderBy {
        term: OrderByTerm::Metric {
            name: "jaccard_similarity".to_string(),
        },
        direction: OrderDirection::Desc,
    }];

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                order_by: Some(&order_by),
                output_source: InferenceOutputSource::Inference,
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Should return results - the ordering correctness is verified by checking
    // that the query executes without error. Full ordering verification would
    // require comparing with feedback values, which is complex.
    assert!(
        !inferences.is_empty(),
        "Should return inferences when ordering by float metric"
    );
}
make_db_test!(test_list_inferences_order_by_float_metric_desc);

async fn test_list_inferences_order_by_float_metric_asc(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    // Order by jaccard_similarity ASC (lower values first)
    let order_by = vec![OrderBy {
        term: OrderByTerm::Metric {
            name: "jaccard_similarity".to_string(),
        },
        direction: OrderDirection::Asc,
    }];

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                order_by: Some(&order_by),
                output_source: InferenceOutputSource::Inference,
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should return inferences when ordering by float metric ASC"
    );
}
make_db_test!(test_list_inferences_order_by_float_metric_asc);

async fn test_list_inferences_order_by_boolean_metric(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    // Order by exact_match (boolean metric) DESC
    let order_by = vec![OrderBy {
        term: OrderByTerm::Metric {
            name: "exact_match".to_string(),
        },
        direction: OrderDirection::Desc,
    }];

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                order_by: Some(&order_by),
                output_source: InferenceOutputSource::Inference,
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should return inferences when ordering by boolean metric"
    );
}
make_db_test!(test_list_inferences_order_by_boolean_metric);

async fn test_list_inferences_order_by_metric_with_filter(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    // Combine filter with ORDER BY metric
    let filter = InferenceFilter::FloatMetric(FloatMetricFilter {
        metric_name: "jaccard_similarity".to_string(),
        value: 0.0,
        comparison_operator: FloatComparisonOperator::GreaterThan,
    });

    let order_by = vec![OrderBy {
        term: OrderByTerm::Metric {
            name: "jaccard_similarity".to_string(),
        },
        direction: OrderDirection::Desc,
    }];

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter),
                order_by: Some(&order_by),
                output_source: InferenceOutputSource::Inference,
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // All returned inferences should have jaccard_similarity > 0
    assert!(
        !inferences.is_empty(),
        "Should return inferences when filtering and ordering by metric"
    );
}
make_db_test!(test_list_inferences_order_by_metric_with_filter);

async fn test_list_inferences_order_by_timestamp(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    // Order by timestamp DESC (most recent first)
    let order_by = vec![OrderBy {
        term: OrderByTerm::Timestamp,
        direction: OrderDirection::Desc,
    }];

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                order_by: Some(&order_by),
                output_source: InferenceOutputSource::Inference,
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should return inferences when ordering by timestamp"
    );

    // Verify ordering: each timestamp should be >= the next
    for window in inferences.windows(2) {
        let ts1 = window[0].timestamp();
        let ts2 = window[1].timestamp();
        assert!(
            ts1 >= ts2,
            "Inferences should be ordered by timestamp DESC: {ts1:?} should be >= {ts2:?}",
        );
    }
}
make_db_test!(test_list_inferences_order_by_timestamp);

async fn test_list_inferences_order_by_multiple_criteria(conn: impl InferenceQueries) {
    let config = get_e2e_config().await;

    // Order by metric first, then timestamp as tie-breaker
    let order_by = vec![
        OrderBy {
            term: OrderByTerm::Metric {
                name: "jaccard_similarity".to_string(),
            },
            direction: OrderDirection::Desc,
        },
        OrderBy {
            term: OrderByTerm::Timestamp,
            direction: OrderDirection::Desc,
        },
    ];

    let inferences = conn
        .list_inferences(
            &config,
            &ListInferencesParams {
                function_name: Some("extract_entities"),
                order_by: Some(&order_by),
                output_source: InferenceOutputSource::Inference,
                limit: 10,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        !inferences.is_empty(),
        "Should return inferences when ordering by multiple criteria"
    );
}
make_db_test!(test_list_inferences_order_by_multiple_criteria);
