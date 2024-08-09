use reqwest::Client;
use std::collections::HashMap;

use crate::e2e::common::get_gateway_endpoint;

/// This file is used to test the Prometheus metrics endpoint of the gateway.
///
/// Namely, it tests that `request_count` is incremented correctly for inference and feedback requests.

#[tokio::test]
async fn test_prometheus_metrics_inference_nonstreaming() {
    let prometheus_metric_name =
        "request_count{endpoint=\"inference\",function_name=\"prometheus_test1\"}";
    let client = Client::new();

    let request_count_before = get_metric_u32(&client, prometheus_metric_name).await;

    // Run inference (standard)
    let inference_payload = serde_json::json!({
        "function_name": "prometheus_test1",
        "input": [
            {"role": "user", "content": "Hello, world!"}
        ],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after,
        request_count_before + 1,
        "Inference request count should have increased by 1"
    );
}

#[tokio::test]
async fn test_prometheus_metrics_inference_nonstreaming_dryrun() {
    let prometheus_metric_name =
        "request_count{endpoint=\"inference\",function_name=\"prometheus_test2\"}";
    let client = Client::new();

    let request_count_before = get_metric_u32(&client, prometheus_metric_name).await;

    // Run inference (standard, dryrun)
    let inference_payload = serde_json::json!({
        "function_name": "prometheus_test2",
        "input": [
            {"role": "user", "content": "Hello, world!"}
        ],
        "stream": false,
        "dryrun": true,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after, request_count_before,
        "Inference request count should not have changed when dryrun is true"
    );
}

#[tokio::test]
async fn test_prometheus_metrics_inference_streaming() {
    let prometheus_metric_name =
        "request_count{endpoint=\"inference\",function_name=\"prometheus_test3\"}";
    let client = Client::new();

    let request_count_before = get_metric_u32(&client, prometheus_metric_name).await;

    // Run inference (streaming)
    let inference_payload = serde_json::json!({
        "function_name": "prometheus_test3",
        "input": [
            {"role": "user", "content": "Hello, world!"}
        ],
        "stream": true,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after,
        request_count_before + 1,
        "Inference request count should have increased by 1"
    );
}

#[tokio::test]
async fn test_prometheus_metrics_inference_streaming_dryrun() {
    let prometheus_metric_name =
        "request_count{endpoint=\"inference\",function_name=\"prometheus_test4\"}";
    let client = Client::new();

    let request_count_before = get_metric_u32(&client, prometheus_metric_name).await;

    // Run inference (streaming, dryrun)
    let inference_payload = serde_json::json!({
        "function_name": "prometheus_test4",
        "input": [
            {"role": "user", "content": "Hello, world!"}
        ],
        "stream": true,
        "dryrun": true,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after, request_count_before,
        "Inference request count should not have changed when dryrun is true"
    );
}

#[tokio::test]
async fn test_prometheus_metrics_feedback_boolean() {
    let prometheus_metric_name =
        "request_count{endpoint=\"feedback\",metric_name=\"prometheus_test_boolean1\"}";
    let client = Client::new();

    let request_count_before = get_metric_u32(&client, prometheus_metric_name).await;

    // Send feedback for task_success
    let feedback_payload = serde_json::json!({
        "inference_id": uuid::Uuid::now_v7(),
        "metric_name": "prometheus_test_boolean1",
        "value": true,
    });

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after,
        request_count_before + 1,
        "Feedback request count should have increased by 1"
    );
}

#[tokio::test]
async fn test_prometheus_metrics_feedback_boolean_dryrun() {
    let prometheus_metric_name =
        "request_count{endpoint=\"feedback\",metric_name=\"prometheus_test_boolean2\"}";
    let client = Client::new();

    let request_count_before = get_metric_u32(&client, prometheus_metric_name).await;

    // Send feedback for task_success
    let feedback_payload = serde_json::json!({
        "inference_id": uuid::Uuid::now_v7(),
        "metric_name": "prometheus_test_boolean2",
        "value": true,
        "dryrun": true,
    });

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after, request_count_before,
        "Feedback request count should not have changed when dryrun is true"
    );
}

#[tokio::test]
async fn test_prometheus_metrics_feedback_float() {
    let prometheus_metric_name =
        "request_count{endpoint=\"feedback\",metric_name=\"prometheus_test_float1\"}";
    let client = Client::new();

    let request_count_before = get_metric_u32(&client, prometheus_metric_name).await;

    // Send feedback for task_success
    let feedback_payload = serde_json::json!({
        "inference_id": uuid::Uuid::now_v7(),
        "metric_name": "prometheus_test_float1",
        "value": 5.0,
    });

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after,
        request_count_before + 1,
        "Feedback request count should have increased by 1"
    );
}

#[tokio::test]
async fn test_prometheus_metrics_feedback_float_dryrun() {
    let prometheus_metric_name =
        "request_count{endpoint=\"feedback\",metric_name=\"prometheus_test_float2\"}";
    let client = Client::new();

    let request_count_before = get_metric_u32(&client, prometheus_metric_name).await;

    // Send feedback for task_success
    let feedback_payload = serde_json::json!({
        "inference_id": uuid::Uuid::now_v7(),
        "metric_name": "prometheus_test_float2",
        "value": 5.0,
        "dryrun": true,
    });

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after, request_count_before,
        "Feedback request count should not have changed when dryrun is true"
    );
}

#[tokio::test]
async fn test_prometheus_metrics_feedback_comment() {
    // POSSIBLE FLAKINESS
    //
    // This test could be flaky if another test sends a comment feedback at the same time.
    // Unlike metrics, comments are global, so there is no way to isolate this test at the moment.
    // If this becomes flaky, we should make sure this test runs sequentially with other tests.

    let prometheus_metric_name = "request_count{endpoint=\"feedback\",metric_name=\"comment\"}";
    let client = Client::new();

    let request_count_before = get_metric_u32(&client, prometheus_metric_name).await;

    // Send feedback for comment
    let feedback_payload = serde_json::json!({
        "inference_id": uuid::Uuid::now_v7(),
        "metric_name": "comment",
        "value": "Splendid!",
    });

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after,
        request_count_before + 1,
        "Feedback request count should have increased by 1"
    );

    // Send feedback for comment (dryrun)
    let feedback_payload = serde_json::json!({
        "inference_id": uuid::Uuid::now_v7(),
        "metric_name": "comment",
        "value": "Splendid!",
        "dryrun": true,
    });

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after_dryrun = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after, request_count_after_dryrun,
        "Feedback request count should not have changed when dryrun is true"
    );
}

#[tokio::test]
async fn test_prometheus_metrics_feedback_demonstration() {
    // POSSIBLE FLAKINESS
    //
    // This test could be flaky if another test sends a demonstration feedback at the same time.
    // Unlike metrics, demonstrations are global, so there is no way to isolate this test at the moment.
    // If this becomes flaky, we should make sure this test runs sequentially with other tests.

    let prometheus_metric_name =
        "request_count{endpoint=\"feedback\",metric_name=\"demonstration\"}";
    let client = Client::new();

    let request_count_before = get_metric_u32(&client, prometheus_metric_name).await;

    // Send feedback for demonstration
    let feedback_payload = serde_json::json!({
        "inference_id": uuid::Uuid::now_v7(),
        "metric_name": "demonstration",
        "value": "Megumin is a powerful arch-wizard with a dramatic flare and extremely vicious competitive side.",
    });

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after,
        request_count_before + 1,
        "Feedback request count should have increased by 1"
    );

    // Send feedback for demonstration (dryrun)
    let feedback_payload = serde_json::json!({
        "inference_id": uuid::Uuid::now_v7(),
        "metric_name": "comment",
        "value": "Splendid!",
        "dryrun": true,
    });

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Sleep for 1 second to allow metrics to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Get metrics after inference
    let request_count_after_dryrun = get_metric_u32(&client, prometheus_metric_name).await;

    assert_eq!(
        request_count_after, request_count_after_dryrun,
        "Feedback request count should not have changed when dryrun is true"
    );
}

async fn get_metric_u32(client: &Client, metric_name: &str) -> u32 {
    let metrics = get_metrics(client).await;
    metrics
        .get(metric_name)
        .unwrap_or(&"0".to_string())
        .to_string()
        .parse::<u32>()
        .unwrap()
}

async fn get_metrics(client: &Client) -> HashMap<String, String> {
    let response = client
        .get(get_gateway_endpoint("/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let metrics: HashMap<String, String> = response
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let mut parts = line.splitn(2, ' ');
            match (parts.next(), parts.next()) {
                (Some(key), Some(value)) => Some((key.to_string(), value.to_string())),
                _ => None,
            }
        })
        .collect();

    metrics
}
