/// Tests for the GET /internal/feedback/{metric_name} endpoint.
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::time::Duration;
use uuid::Uuid;

use tensorzero_core::endpoints::feedback::GetFeedbackResponse;

use crate::common::get_gateway_endpoint;

/// Helper to run an inference and get an inference_id
async fn run_inference(client: &Client) -> Uuid {
    let inference_payload = json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    Uuid::parse_str(inference_id).unwrap()
}

/// Helper to submit boolean feedback
async fn submit_boolean_feedback(
    client: &Client,
    inference_id: Uuid,
    metric_name: &str,
    value: bool,
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

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap().as_str().unwrap();
    Uuid::parse_str(feedback_id).unwrap()
}

/// Helper to submit float feedback
async fn submit_float_feedback(
    client: &Client,
    inference_id: Uuid,
    metric_name: &str,
    value: f64,
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

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap().as_str().unwrap();
    Uuid::parse_str(feedback_id).unwrap()
}

#[tokio::test]
async fn test_get_feedback_boolean_metric() {
    let client = Client::new();

    // Run inference to get a valid inference_id
    let inference_id = run_inference(&client).await;

    // Submit boolean feedback
    let feedback_id = submit_boolean_feedback(&client, inference_id, "task_success", true).await;

    // Wait for async write to complete
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Query the get_feedback endpoint with retry
    let mut found = false;
    for _ in 0..10 {
        let resp = client
            .get(get_gateway_endpoint("/internal/feedback/task_success"))
            .send()
            .await
            .unwrap();

        if resp.status() == StatusCode::OK {
            let body: GetFeedbackResponse = resp.json().await.unwrap();
            if body
                .feedback
                .iter()
                .any(|f| f.target_id == inference_id && f.feedback_id == feedback_id)
            {
                // Verify the value is correct (boolean true)
                let our_feedback = body
                    .feedback
                    .iter()
                    .find(|f| f.target_id == inference_id)
                    .unwrap();
                assert_eq!(our_feedback.value, json!(true));
                found = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    assert!(
        found,
        "Feedback for inference_id {inference_id} should appear in get_feedback response"
    );
}

#[tokio::test]
async fn test_get_feedback_float_metric() {
    let client = Client::new();

    // Run inference to get a valid inference_id
    let inference_id = run_inference(&client).await;

    // Submit float feedback (brevity_score is inference-level float)
    let feedback_id = submit_float_feedback(&client, inference_id, "brevity_score", 0.85).await;

    // Wait for async write to complete
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Query the get_feedback endpoint with retry
    let mut found = false;
    for _ in 0..10 {
        let resp = client
            .get(get_gateway_endpoint("/internal/feedback/brevity_score"))
            .send()
            .await
            .unwrap();

        if resp.status() == StatusCode::OK {
            let body: GetFeedbackResponse = resp.json().await.unwrap();
            if body
                .feedback
                .iter()
                .any(|f| f.target_id == inference_id && f.feedback_id == feedback_id)
            {
                // Verify the value is correct
                let our_feedback = body
                    .feedback
                    .iter()
                    .find(|f| f.target_id == inference_id)
                    .unwrap();
                assert_eq!(our_feedback.value, json!(0.85));
                found = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    assert!(
        found,
        "Feedback for inference_id {inference_id} should appear in get_feedback response"
    );
}

#[tokio::test]
async fn test_get_feedback_with_pagination() {
    let client = Client::new();

    // Create multiple inferences and feedback entries
    let mut inference_ids = Vec::new();
    for _ in 0..3 {
        let inference_id = run_inference(&client).await;
        submit_boolean_feedback(&client, inference_id, "exact_match", true).await;
        inference_ids.push(inference_id);
    }

    // Wait for async writes to complete
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Wait for all feedback to be visible
    let mut all_found = false;
    for _ in 0..10 {
        let resp = client
            .get(get_gateway_endpoint("/internal/feedback/exact_match"))
            .send()
            .await
            .unwrap();

        if resp.status() == StatusCode::OK {
            let body: GetFeedbackResponse = resp.json().await.unwrap();
            let found_count = inference_ids
                .iter()
                .filter(|id| body.feedback.iter().any(|f| &f.target_id == *id))
                .count();

            if found_count == inference_ids.len() {
                all_found = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    assert!(
        all_found,
        "All 3 feedback entries should appear in the list"
    );

    // Test limit parameter
    let resp = client
        .get(get_gateway_endpoint(
            "/internal/feedback/exact_match?limit=1",
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body: GetFeedbackResponse = resp.json().await.unwrap();
    assert_eq!(
        body.feedback.len(),
        1,
        "Should return exactly 1 feedback with limit=1"
    );

    // Test pagination with offset
    let resp_page_1 = client
        .get(get_gateway_endpoint(
            "/internal/feedback/exact_match?limit=1&offset=0",
        ))
        .send()
        .await
        .unwrap();

    let resp_page_2 = client
        .get(get_gateway_endpoint(
            "/internal/feedback/exact_match?limit=1&offset=1",
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(resp_page_1.status(), StatusCode::OK);
    assert_eq!(resp_page_2.status(), StatusCode::OK);

    let body_page_1: GetFeedbackResponse = resp_page_1.json().await.unwrap();
    let body_page_2: GetFeedbackResponse = resp_page_2.json().await.unwrap();

    assert_eq!(body_page_1.feedback.len(), 1);
    assert_eq!(body_page_2.feedback.len(), 1);

    // Pages should have different target_ids (they're ordered by feedback_id DESC)
    assert_ne!(
        body_page_1.feedback[0].target_id, body_page_2.feedback[0].target_id,
        "Different pages should return different feedback entries"
    );
}

#[tokio::test]
async fn test_get_feedback_nonexistent_metric() {
    let client = Client::new();

    // Query a metric that doesn't exist in the config
    let nonexistent_metric = format!("nonexistent_metric_{}", Uuid::now_v7());
    let resp = client
        .get(get_gateway_endpoint(&format!(
            "/internal/feedback/{nonexistent_metric}"
        )))
        .send()
        .await
        .unwrap();

    // Should return 404 Not Found
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_feedback_empty_result() {
    let client = Client::new();

    // Query a valid metric that has no feedback data
    // prometheus_test_boolean1 is a valid metric but unlikely to have feedback in tests
    let resp = client
        .get(get_gateway_endpoint(
            "/internal/feedback/prometheus_test_boolean1",
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let _body: GetFeedbackResponse = resp.json().await.unwrap();

    // Result should be valid (empty list or with some entries)
    // We can't guarantee empty since other tests may have written to it
    // Just verify the response parses correctly
}

#[tokio::test]
async fn test_get_feedback_returns_latest_per_target() {
    let client = Client::new();

    // Run inference to get a valid inference_id
    let inference_id = run_inference(&client).await;

    // Submit multiple feedback values for the same target_id
    // First feedback: false
    submit_boolean_feedback(&client, inference_id, "test_metric", false).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Second feedback: true (should be the latest)
    let latest_feedback_id =
        submit_boolean_feedback(&client, inference_id, "test_metric", true).await;

    // Wait for async writes to complete
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Query the get_feedback endpoint with retry
    let mut found = false;
    for _ in 0..10 {
        let resp = client
            .get(get_gateway_endpoint("/internal/feedback/test_metric"))
            .send()
            .await
            .unwrap();

        if resp.status() == StatusCode::OK {
            let body: GetFeedbackResponse = resp.json().await.unwrap();

            // Find feedback for our inference_id
            let our_feedback: Vec<_> = body
                .feedback
                .iter()
                .filter(|f| f.target_id == inference_id)
                .collect();

            // Should only have ONE entry per target_id (the latest)
            if our_feedback.len() == 1 {
                let feedback = our_feedback[0];
                // Should be the latest value (true) with the latest feedback_id
                assert_eq!(
                    feedback.value,
                    json!(true),
                    "Should return the latest value"
                );
                assert_eq!(
                    feedback.feedback_id, latest_feedback_id,
                    "Should return the latest feedback_id"
                );
                found = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    assert!(
        found,
        "Should find exactly one feedback entry for inference_id {inference_id} with the latest value"
    );
}
