//! E2E tests for the feedback endpoints.

use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use tensorzero_core::endpoints::feedback::internal::{
    CountFeedbackByTargetIdResponse, GetCumulativeFeedbackTimeseriesResponse,
    GetDemonstrationFeedbackResponse, GetFeedbackBoundsResponse, GetFeedbackByTargetIdResponse,
    LatestFeedbackIdByMetricResponse,
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
) -> Uuid {
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

    let response_json: serde_json::Value = response.json().await.unwrap();
    Uuid::parse_str(response_json["feedback_id"].as_str().unwrap()).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_latest_feedback_id_by_metric_with_data() {
    let http_client = Client::new();

    // Create an inference
    let inference_id = create_inference(&http_client, "basic_test").await;

    // Submit feedback for task_success metric
    let feedback_id =
        submit_inference_feedback(&http_client, inference_id, "task_success", json!(true)).await;

    // Wait for ClickHouse to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Query latest feedback by metric for the inference
    let url = get_gateway_endpoint(&format!(
        "/internal/feedback/{inference_id}/latest-id-by-metric"
    ));
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "get_latest_feedback_id_by_metric request failed: status={:?}",
        resp.status()
    );

    let response: LatestFeedbackIdByMetricResponse = resp.json().await.unwrap();

    // Should have task_success metric
    assert!(
        response.feedback_id_by_metric.contains_key("task_success"),
        "Expected to find task_success in response"
    );

    // Verify the feedback ID matches
    assert_eq!(
        response.feedback_id_by_metric.get("task_success"),
        Some(&feedback_id.to_string())
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_latest_feedback_id_by_metric_multiple_feedback_same_metric() {
    let http_client = Client::new();

    // Create an inference
    let inference_id = create_inference(&http_client, "basic_test").await;

    // Submit multiple feedback entries for the same metric
    let _feedback_id_1 =
        submit_inference_feedback(&http_client, inference_id, "task_success", json!(false)).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    let feedback_id_2 =
        submit_inference_feedback(&http_client, inference_id, "task_success", json!(true)).await;

    // Wait for ClickHouse to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Query latest feedback by metric
    let url = get_gateway_endpoint(&format!(
        "/internal/feedback/{inference_id}/latest-id-by-metric"
    ));
    let resp = http_client.get(url).send().await.unwrap();

    assert!(resp.status().is_success());
    let response: LatestFeedbackIdByMetricResponse = resp.json().await.unwrap();

    // Should only have the latest feedback ID (feedback_id_2)
    assert_eq!(
        response.feedback_id_by_metric.get("task_success"),
        Some(&feedback_id_2.to_string()),
        "Expected to get the latest feedback ID"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_latest_feedback_id_by_metric_nonexistent_target() {
    let http_client = Client::new();

    // Use a UUID that likely doesn't exist
    let nonexistent_id = Uuid::now_v7();
    let url = get_gateway_endpoint(&format!(
        "/internal/feedback/{nonexistent_id}/latest-id-by-metric"
    ));

    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "Should return success even for nonexistent target"
    );

    let response: LatestFeedbackIdByMetricResponse = resp.json().await.unwrap();

    // Should return empty map
    assert_eq!(
        response.feedback_id_by_metric,
        HashMap::new(),
        "Expected empty map for nonexistent target"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_latest_feedback_id_by_metric_invalid_uuid() {
    let http_client = Client::new();

    // Use an invalid UUID
    let url = get_gateway_endpoint("/internal/feedback/not-a-valid-uuid/latest-id-by-metric");

    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for invalid UUID"
    );
}

// ==================== Get Feedback Bounds By Target ID Tests ====================

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_bounds_by_target_id_with_feedback() {
    let http_client = Client::new();

    let inference_id = create_inference(&http_client, "basic_test").await;
    let feedback_id =
        submit_inference_feedback(&http_client, inference_id, "task_success", json!(true)).await;

    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    let url = get_gateway_endpoint(&format!("/internal/feedback/{inference_id}/bounds"));
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "Expected success when querying feedback bounds"
    );

    let response: GetFeedbackBoundsResponse = resp.json().await.unwrap();
    assert_eq!(
        response.first_id,
        Some(feedback_id),
        "Expected first_id to match the submitted feedback"
    );
    assert_eq!(
        response.last_id,
        Some(feedback_id),
        "Expected last_id to match the submitted feedback"
    );
    assert_eq!(
        response.by_type.boolean.first_id,
        Some(feedback_id),
        "Expected boolean bounds to include the feedback id"
    );
    assert!(
        response.by_type.float.first_id.is_none(),
        "Expected float bounds to be empty"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_bounds_by_target_id_nonexistent_target() {
    let http_client = Client::new();
    let nonexistent_id = Uuid::now_v7();

    let url = get_gateway_endpoint(&format!("/internal/feedback/{nonexistent_id}/bounds"));
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "Expected success for nonexistent target"
    );

    let response: GetFeedbackBoundsResponse = resp.json().await.unwrap();
    assert!(
        response.first_id.is_none() && response.last_id.is_none(),
        "Expected no bounds for nonexistent target"
    );
    assert!(
        response.by_type.boolean.first_id.is_none(),
        "Expected boolean bounds to be empty"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_bounds_by_target_id_invalid_uuid() {
    let http_client = Client::new();

    let url = get_gateway_endpoint("/internal/feedback/not-a-valid-uuid/bounds");
    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for invalid UUID"
    );
}

// ==================== Get Feedback By Target ID Tests ====================

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_by_target_id_with_data() {
    let http_client = Client::new();

    // Create an inference
    let inference_id = create_inference(&http_client, "basic_test").await;

    // Submit feedback for task_success metric
    let _feedback_id =
        submit_inference_feedback(&http_client, inference_id, "task_success", json!(true)).await;

    // Wait for ClickHouse to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Query feedback by target ID
    let url = get_gateway_endpoint(&format!("/internal/feedback/{inference_id}"));
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "get_feedback_by_target_id request failed: status={:?}",
        resp.status()
    );

    let response: GetFeedbackByTargetIdResponse = resp.json().await.unwrap();

    // Should have at least one feedback entry
    assert!(
        !response.feedback.is_empty(),
        "Expected to find feedback entries"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_by_target_id_with_pagination() {
    let http_client = Client::new();

    // Create an inference
    let inference_id = create_inference(&http_client, "basic_test").await;

    // Submit multiple feedback entries
    let _feedback_id_1 =
        submit_inference_feedback(&http_client, inference_id, "task_success", json!(true)).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    let _feedback_id_2 =
        submit_inference_feedback(&http_client, inference_id, "task_success", json!(false)).await;

    // Wait for ClickHouse to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Query feedback with limit=1
    let url = get_gateway_endpoint(&format!("/internal/feedback/{inference_id}?limit=1"));
    let resp = http_client.get(url).send().await.unwrap();

    assert!(resp.status().is_success());
    let response: GetFeedbackByTargetIdResponse = resp.json().await.unwrap();

    // Should have at most 1 feedback entry
    assert!(
        response.feedback.len() <= 1,
        "Expected at most 1 feedback entry with limit=1"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_by_target_id_nonexistent_target() {
    let http_client = Client::new();

    // Use a UUID that likely doesn't exist
    let nonexistent_id = Uuid::now_v7();
    let url = get_gateway_endpoint(&format!("/internal/feedback/{nonexistent_id}"));

    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "Should return success even for nonexistent target"
    );

    let response: GetFeedbackByTargetIdResponse = resp.json().await.unwrap();

    // Should return empty list
    assert!(
        response.feedback.is_empty(),
        "Expected empty list for nonexistent target"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_by_target_id_invalid_uuid() {
    let http_client = Client::new();

    // Use an invalid UUID
    let url = get_gateway_endpoint("/internal/feedback/not-a-valid-uuid");

    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for invalid UUID"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_by_target_id_rejects_both_before_and_after() {
    let http_client = Client::new();

    let target_id = Uuid::now_v7();
    let before_id = Uuid::now_v7();
    let after_id = Uuid::now_v7();

    // Try to use both before and after
    let url = get_gateway_endpoint(&format!(
        "/internal/feedback/{target_id}?before={before_id}&after={after_id}"
    ));

    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 when both before and after are specified"
    );
}

// ==================== Count Feedback By Target ID Tests ====================

#[tokio::test(flavor = "multi_thread")]
async fn test_count_feedback_by_target_id_with_feedback() {
    let http_client = Client::new();

    let inference_id = create_inference(&http_client, "basic_test").await;
    let _feedback_id =
        submit_inference_feedback(&http_client, inference_id, "task_success", json!(true)).await;

    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    let url = get_gateway_endpoint(&format!("/internal/feedback/{inference_id}/count"));
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "Expected success when counting feedback"
    );

    let response: CountFeedbackByTargetIdResponse = resp.json().await.unwrap();
    assert_eq!(
        response.count, 1,
        "Expected count to be 1 for single feedback entry"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_feedback_by_target_id_multiple_feedback() {
    let http_client = Client::new();

    let inference_id = create_inference(&http_client, "basic_test").await;

    // Submit multiple feedback entries
    submit_inference_feedback(&http_client, inference_id, "task_success", json!(true)).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    submit_inference_feedback(&http_client, inference_id, "task_success", json!(false)).await;

    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    let url = get_gateway_endpoint(&format!("/internal/feedback/{inference_id}/count"));
    let resp = http_client.get(url).send().await.unwrap();

    assert!(resp.status().is_success());
    let response: CountFeedbackByTargetIdResponse = resp.json().await.unwrap();

    assert_eq!(
        response.count, 2,
        "Expected count to be 2 for multiple feedback entries"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_feedback_by_target_id_nonexistent_target() {
    let http_client = Client::new();
    let nonexistent_id = Uuid::now_v7();

    let url = get_gateway_endpoint(&format!("/internal/feedback/{nonexistent_id}/count"));
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "Expected success for nonexistent target"
    );

    let response: CountFeedbackByTargetIdResponse = resp.json().await.unwrap();
    assert_eq!(
        response.count, 0,
        "Expected count to be 0 for nonexistent target"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_count_feedback_by_target_id_invalid_uuid() {
    let http_client = Client::new();

    let url = get_gateway_endpoint("/internal/feedback/not-a-valid-uuid/count");
    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for invalid UUID"
    );
}

// ==================== Get Cumulative Feedback Timeseries Tests ====================

#[tokio::test(flavor = "multi_thread")]
async fn test_get_cumulative_feedback_timeseries_with_data() {
    let http_client = Client::new();

    // Create an inference to generate feedback data
    let inference_id = create_inference(&http_client, "basic_test").await;

    // Submit feedback for task_success metric
    submit_inference_feedback(&http_client, inference_id, "task_success", json!(true)).await;

    // Wait for ClickHouse to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Query cumulative feedback timeseries
    let url = get_gateway_endpoint(
        "/internal/feedback/timeseries?function_name=basic_test&metric_name=task_success&time_window=hour&max_periods=24",
    );
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "get_cumulative_feedback_timeseries request failed: status={:?}",
        resp.status()
    );

    let response: GetCumulativeFeedbackTimeseriesResponse = resp.json().await.unwrap();

    // The response should be valid (could be empty if no data or have entries)
    // We're mainly verifying the endpoint works correctly
    let _ = response.timeseries;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_cumulative_feedback_timeseries_with_variant_filter() {
    let http_client = Client::new();

    // Query cumulative feedback timeseries with variant_names filter
    let url = get_gateway_endpoint(
        "/internal/feedback/timeseries?function_name=basic_test&metric_name=task_success&time_window=day&max_periods=7&variant_names=test",
    );
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "get_cumulative_feedback_timeseries request failed with variant filter: status={:?}",
        resp.status()
    );

    let _response: GetCumulativeFeedbackTimeseriesResponse = resp.json().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_cumulative_feedback_timeseries_different_time_windows() {
    let http_client = Client::new();

    // Test different time windows (excluding cumulative which is not supported)
    for time_window in &["minute", "hour", "day", "week", "month"] {
        let url = get_gateway_endpoint(&format!(
            "/internal/feedback/timeseries?function_name=basic_test&metric_name=task_success&time_window={time_window}&max_periods=5",
        ));
        let resp = http_client.get(url).send().await.unwrap();

        assert!(
            resp.status().is_success(),
            "get_cumulative_feedback_timeseries request failed for time_window={}: status={:?}",
            time_window,
            resp.status()
        );

        let _response: GetCumulativeFeedbackTimeseriesResponse = resp.json().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_cumulative_feedback_timeseries_cumulative_window_returns_error() {
    let http_client = Client::new();

    // Cumulative time window is not supported for feedback timeseries
    let url = get_gateway_endpoint(
        "/internal/feedback/timeseries?function_name=basic_test&metric_name=task_success&time_window=cumulative&max_periods=5",
    );
    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for cumulative time_window"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_cumulative_feedback_timeseries_missing_params() {
    let http_client = Client::new();

    // Missing function_name
    let url = get_gateway_endpoint(
        "/internal/feedback/timeseries?metric_name=task_success&time_window=hour&max_periods=24",
    );
    let resp = http_client.get(url).send().await.unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 when function_name is missing"
    );

    // Missing metric_name
    let url = get_gateway_endpoint(
        "/internal/feedback/timeseries?function_name=basic_test&time_window=hour&max_periods=24",
    );
    let resp = http_client.get(url).send().await.unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 when metric_name is missing"
    );

    // Missing time_window
    let url = get_gateway_endpoint(
        "/internal/feedback/timeseries?function_name=basic_test&metric_name=task_success&max_periods=24",
    );
    let resp = http_client.get(url).send().await.unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 when time_window is missing"
    );

    // Missing max_periods
    let url = get_gateway_endpoint(
        "/internal/feedback/timeseries?function_name=basic_test&metric_name=task_success&time_window=hour",
    );
    let resp = http_client.get(url).send().await.unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 when max_periods is missing"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_cumulative_feedback_timeseries_invalid_time_window() {
    let http_client = Client::new();

    let url = get_gateway_endpoint(
        "/internal/feedback/timeseries?function_name=basic_test&metric_name=task_success&time_window=invalid&max_periods=24",
    );
    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for invalid time_window"
    );
}

// ==================== Get Demonstration Feedback By Inference ID Tests ====================

/// Helper function to submit demonstration feedback for an inference
async fn submit_demonstration_feedback(
    client: &Client,
    inference_id: Uuid,
    value: serde_json::Value,
) -> Uuid {
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
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
        "Failed to submit demonstration feedback: {}",
        response.status()
    );

    let response_json: serde_json::Value = response.json().await.unwrap();
    Uuid::parse_str(response_json["feedback_id"].as_str().unwrap()).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_demonstration_feedback_with_data() {
    let http_client = Client::new();

    // Create an inference
    let inference_id = create_inference(&http_client, "basic_test").await;

    // Submit demonstration feedback
    let feedback_id =
        submit_demonstration_feedback(&http_client, inference_id, json!("This is a demonstration"))
            .await;

    // Wait for ClickHouse to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Query demonstration feedback by inference ID
    let url = get_gateway_endpoint(&format!("/internal/feedback/{inference_id}/demonstrations"));
    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "get_demonstration_feedback request failed: status={:?}",
        resp.status()
    );

    let response: GetDemonstrationFeedbackResponse = resp.json().await.unwrap();

    // Should have at least one feedback entry
    assert!(
        !response.feedback.is_empty(),
        "Expected to find demonstration feedback entries"
    );

    // Verify the feedback ID matches
    assert_eq!(
        response.feedback[0].id, feedback_id,
        "Expected feedback ID to match"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_demonstration_feedback_with_pagination() {
    let http_client = Client::new();

    // Create an inference
    let inference_id = create_inference(&http_client, "basic_test").await;

    // Submit multiple demonstration feedback entries
    submit_demonstration_feedback(&http_client, inference_id, json!("Demo 1")).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    submit_demonstration_feedback(&http_client, inference_id, json!("Demo 2")).await;

    // Wait for ClickHouse to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

    // Query demonstration feedback with limit=1
    let url = get_gateway_endpoint(&format!(
        "/internal/feedback/{inference_id}/demonstrations?limit=1"
    ));
    let resp = http_client.get(url).send().await.unwrap();

    assert!(resp.status().is_success());
    let response: GetDemonstrationFeedbackResponse = resp.json().await.unwrap();

    // Should have at most 1 feedback entry
    assert!(
        response.feedback.len() <= 1,
        "Expected at most 1 feedback entry with limit=1"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_demonstration_feedback_nonexistent_target() {
    let http_client = Client::new();

    // Use a UUID that likely doesn't exist
    let nonexistent_id = Uuid::now_v7();
    let url = get_gateway_endpoint(&format!(
        "/internal/feedback/{nonexistent_id}/demonstrations"
    ));

    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        resp.status().is_success(),
        "Should return success even for nonexistent target"
    );

    let response: GetDemonstrationFeedbackResponse = resp.json().await.unwrap();

    // Should return empty list
    assert!(
        response.feedback.is_empty(),
        "Expected empty list for nonexistent target"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_demonstration_feedback_invalid_uuid() {
    let http_client = Client::new();

    // Use an invalid UUID
    let url = get_gateway_endpoint("/internal/feedback/not-a-valid-uuid/demonstrations");

    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 for invalid UUID"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_demonstration_feedback_rejects_both_before_and_after() {
    let http_client = Client::new();

    let inference_id = Uuid::now_v7();
    let before_id = Uuid::now_v7();
    let after_id = Uuid::now_v7();

    // Try to use both before and after
    let url = get_gateway_endpoint(&format!(
        "/internal/feedback/{inference_id}/demonstrations?before={before_id}&after={after_id}"
    ));

    let resp = http_client.get(url).send().await.unwrap();

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "Expected 400 when both before and after are specified"
    );
}
