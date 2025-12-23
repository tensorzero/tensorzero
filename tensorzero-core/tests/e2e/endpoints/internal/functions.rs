//! E2E tests for the functions endpoints.

use reqwest::Client;
use serde_json::json;
use tensorzero_core::endpoints::functions::internal::{
    MetricsWithFeedbackResponse, VariantPerformancesResponse,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Helper function to create an inference and return its inference_id
async fn create_inference(client: &Client, function_name: &str) -> Uuid {
    let inference_payload = json!({
        "function_name": function_name,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(
        response.status().is_success(),
        "Failed to create inference: {}",
        response.status()
    );

    let response_json: serde_json::Value = response.json().await.unwrap();
    Uuid::parse_str(response_json["inference_id"].as_str().unwrap()).unwrap()
}

/// Helper function to submit feedback for an inference
async fn submit_inference_feedback(
    client: &Client,
    inference_id: Uuid,
    metric_name: &str,
    value: serde_json::Value,
) {
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": metric_name,
        "value": value,
    });

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        response.status().is_success(),
        "Failed to submit feedback: {}",
        response.status()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_function_metrics_with_feedback() {
    let http_client = Client::new();

    // Create inferences with different types of feedback
    let inference_id = create_inference(&http_client, "basic_test").await;

    // Submit boolean metric feedback (task_success is a boolean metric at inference level)
    submit_inference_feedback(&http_client, inference_id, "task_success", json!(true)).await;

    // Wait a bit for ClickHouse to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Query metrics with feedback for the function
    let url = get_gateway_endpoint("/internal/functions/basic_test/metrics");
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "get_function_metrics request failed: status={:?}",
        resp.status()
    );

    let metrics: MetricsWithFeedbackResponse = resp.json().await.unwrap();

    // Should have at least the task_success metric
    assert!(
        metrics
            .metrics
            .iter()
            .any(|m| m.metric_name == "task_success"),
        "Expected to find task_success metric in response"
    );

    let task_success_metric = metrics
        .metrics
        .iter()
        .find(|m| m.metric_name == "task_success")
        .unwrap();

    assert_eq!(task_success_metric.function_name, "basic_test");
    assert!(task_success_metric.feedback_count > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_function_metrics_with_variant_filter() {
    let http_client = Client::new();

    // Create inference (which will use a specific variant based on the config)
    let inference_id = create_inference(&http_client, "basic_test").await;

    // Submit feedback
    submit_inference_feedback(&http_client, inference_id, "task_success", json!(false)).await;

    // Wait for ClickHouse
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Query with variant_name parameter
    // Note: We're using "variant_0" which is defined in the test config
    let url = get_gateway_endpoint("/internal/functions/basic_test/metrics?variant_name=variant_0");
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "get_function_metrics with variant filter failed: status={:?}",
        resp.status()
    );

    let metrics: MetricsWithFeedbackResponse = resp.json().await.unwrap();

    // All metrics should be for the same function
    for metric in &metrics.metrics {
        assert_eq!(metric.function_name, "basic_test");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_function_metrics_nonexistent_function() {
    let http_client = Client::new();

    // Try to query metrics for a function that doesn't exist
    let url = get_gateway_endpoint("/internal/functions/nonexistent_function/metrics");
    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::NOT_FOUND,
        "Expected 404 for nonexistent function"
    );
}

// =====================================================================
// Tests for get_variant_performances endpoint
// =====================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_get_variant_performances_cumulative() {
    let http_client = Client::new();

    // Create an inference and submit feedback
    let inference_id = create_inference(&http_client, "basic_test").await;
    submit_inference_feedback(&http_client, inference_id, "task_success", json!(true)).await;

    // Wait for ClickHouse to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Query variant performances
    let url = get_gateway_endpoint(
        "/internal/functions/basic_test/variant-performances?metric_name=task_success&time_window=cumulative",
    );
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "get_variant_performances request failed: status={:?}, body={:?}",
        resp.status(),
        resp.text().await
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_variant_performances_with_time_window() {
    let http_client = Client::new();

    // Create an inference and submit feedback
    let inference_id = create_inference(&http_client, "basic_test").await;
    submit_inference_feedback(&http_client, inference_id, "task_success", json!(false)).await;

    // Wait for ClickHouse to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Test with different time windows
    for time_window in ["minute", "hour", "day", "week", "month"] {
        let url = get_gateway_endpoint(&format!(
            "/internal/functions/basic_test/variant-performances?metric_name=task_success&time_window={time_window}"
        ));
        let resp = http_client.get(url).send().await.unwrap();

        assert!(
            resp.status().is_success(),
            "get_variant_performances failed for time_window={}: status={:?}",
            time_window,
            resp.status()
        );

        let response: VariantPerformancesResponse = resp.json().await.unwrap();
        // Response may be empty if no data, but should be a valid response
        for perf in &response.performances {
            assert!(!perf.variant_name.is_empty());
            assert!(perf.count > 0);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_variant_performances_with_variant_filter() {
    let http_client = Client::new();

    // Create an inference and submit feedback
    let inference_id = create_inference(&http_client, "basic_test").await;
    submit_inference_feedback(&http_client, inference_id, "task_success", json!(true)).await;

    // Wait for ClickHouse to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Query with variant_name filter - use "test" which is a valid variant for basic_test
    let url = get_gateway_endpoint(
        "/internal/functions/basic_test/variant-performances?metric_name=task_success&time_window=cumulative&variant_name=test",
    );
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "get_variant_performances with variant filter failed: status={:?}",
        resp.status()
    );

    let response: VariantPerformancesResponse = resp.json().await.unwrap();

    // All results should be for the filtered variant
    for perf in &response.performances {
        assert_eq!(perf.variant_name, "test");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_variant_performances_nonexistent_function() {
    let http_client = Client::new();

    let url = get_gateway_endpoint(
        "/internal/functions/nonexistent_function/variant-performances?metric_name=task_success&time_window=cumulative",
    );
    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::NOT_FOUND,
        "Expected 404 for nonexistent function"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_variant_performances_nonexistent_metric() {
    let http_client = Client::new();

    let url = get_gateway_endpoint(
        "/internal/functions/basic_test/variant-performances?metric_name=nonexistent_metric&time_window=cumulative",
    );
    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::NOT_FOUND,
        "Expected 404 for nonexistent metric"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_variant_performances_nonexistent_variant() {
    let http_client = Client::new();

    let url = get_gateway_endpoint(
        "/internal/functions/basic_test/variant-performances?metric_name=task_success&time_window=cumulative&variant_name=nonexistent_variant",
    );
    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::NOT_FOUND,
        "Expected 404 for nonexistent variant"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_variant_performances_missing_required_params() {
    let http_client = Client::new();

    // Missing metric_name
    let url = get_gateway_endpoint(
        "/internal/functions/basic_test/variant-performances?time_window=cumulative",
    );
    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_client_error(),
        "Expected error when metric_name is missing"
    );

    // Missing time_window
    let url = get_gateway_endpoint(
        "/internal/functions/basic_test/variant-performances?metric_name=task_success",
    );
    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_client_error(),
        "Expected error when time_window is missing"
    );
}
