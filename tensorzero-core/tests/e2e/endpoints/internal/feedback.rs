//! E2E tests for the feedback endpoints.

use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use tensorzero_core::endpoints::feedback::internal::LatestFeedbackIdByMetricResponse;
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
