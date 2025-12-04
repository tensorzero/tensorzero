#![expect(clippy::print_stdout)]
use tensorzero_core::db::{clickhouse::test_helpers::get_clickhouse, feedback::FeedbackQueries};

/// Test get_aggregated_feedback_by_variant without any filters
#[tokio::test]
async fn test_get_aggregated_feedback_by_variant_no_filters() {
    let clickhouse = get_clickhouse().await;

    // Query for a function that should have some feedback data
    // The function "json_success" is used in the test fixtures
    let result = clickhouse
        .get_aggregated_feedback_by_variant("json_success", None, None)
        .await
        .unwrap();

    println!("Results without filters: {result:#?}");

    // We don't assert on specific data since the test database may vary,
    // but we verify the query executes successfully and returns a valid result
}

/// Test get_aggregated_feedback_by_variant with a metric filter
#[tokio::test]
async fn test_get_aggregated_feedback_by_variant_with_metric_filter() {
    let clickhouse = get_clickhouse().await;

    // Query with a specific metric filter
    let result = clickhouse
        .get_aggregated_feedback_by_variant("json_success", None, Some("task_success"))
        .await
        .unwrap();

    println!("Results with metric filter: {result:#?}");

    // Verify all results have the correct metric_name
    for feedback in &result {
        assert_eq!(feedback.metric_name, "task_success");
    }
}

/// Test get_aggregated_feedback_by_variant with a variant filter
#[tokio::test]
async fn test_get_aggregated_feedback_by_variant_with_variant_filter() {
    let clickhouse = get_clickhouse().await;

    // Query with a specific variant filter
    let result = clickhouse
        .get_aggregated_feedback_by_variant("json_success", Some("test"), None)
        .await
        .unwrap();

    println!("Results with variant filter: {result:#?}");

    // Verify all results have the correct variant_name
    for feedback in &result {
        assert_eq!(feedback.variant_name, "test");
    }
}

/// Test get_aggregated_feedback_by_variant with both filters
#[tokio::test]
async fn test_get_aggregated_feedback_by_variant_with_both_filters() {
    let clickhouse = get_clickhouse().await;

    // Query with both filters
    let result = clickhouse
        .get_aggregated_feedback_by_variant("json_success", Some("test"), Some("task_success"))
        .await
        .unwrap();

    println!("Results with both filters: {result:#?}");

    // Verify all results have the correct variant_name and metric_name
    for feedback in &result {
        assert_eq!(feedback.variant_name, "test");
        assert_eq!(feedback.metric_name, "task_success");
    }
}

/// Test get_aggregated_feedback_by_variant with a nonexistent function returns empty
#[tokio::test]
async fn test_get_aggregated_feedback_by_variant_nonexistent_function() {
    let clickhouse = get_clickhouse().await;

    // Query for a function that doesn't exist
    let result = clickhouse
        .get_aggregated_feedback_by_variant("nonexistent_function_12345", None, None)
        .await
        .unwrap();

    assert!(
        result.is_empty(),
        "Should return empty for nonexistent function"
    );
}

/// Test get_aggregated_feedback_by_variant returns proper statistical aggregates
#[tokio::test]
async fn test_get_aggregated_feedback_by_variant_statistics() {
    let clickhouse = get_clickhouse().await;

    // Query for feedback data
    let result = clickhouse
        .get_aggregated_feedback_by_variant("json_success", None, Some("task_success"))
        .await
        .unwrap();

    println!("Statistics results: {result:#?}");

    // Verify statistical properties of returned data
    for feedback in &result {
        // Mean should be between 0 and 1 for boolean metrics converted to float
        assert!(
            feedback.mean >= 0.0 && feedback.mean <= 1.0,
            "Mean should be between 0 and 1 for task_success metric"
        );

        // Variance should be non-negative if present
        if let Some(variance) = feedback.variance {
            assert!(variance >= 0.0, "Variance should be non-negative");
        }

        // Count should be positive
        assert!(feedback.count > 0, "Count should be positive");
    }
}
