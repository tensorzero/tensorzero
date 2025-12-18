//! E2E tests for the inference stats endpoints.
//!
//! These tests use the functions and metrics defined in tensorzero-core/tests/e2e/config/
//! - Functions: basic_test, weather_helper, json_success
//! - Metrics: task_success (bool/inference), brevity_score (float/inference),
//!   goal_achieved (bool/episode), user_rating (float/episode)

use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use tensorzero_core::endpoints::internal::inference_stats::{
    InferenceStatsResponse, InferenceWithFeedbackStatsResponse,
};
use tokio::time::{Duration, sleep};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Helper function to create an inference and return its inference_id and episode_id
async fn create_inference(client: &Client, function_name: &str) -> (Uuid, Uuid) {
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

    let response_json: Value = response.json().await.unwrap();
    let inference_id = Uuid::parse_str(response_json["inference_id"].as_str().unwrap()).unwrap();
    let episode_id = Uuid::parse_str(response_json["episode_id"].as_str().unwrap()).unwrap();

    (inference_id, episode_id)
}

/// Helper function to submit feedback for an inference
async fn submit_inference_feedback(
    client: &Client,
    inference_id: Uuid,
    metric_name: &str,
    value: Value,
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

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "Failed to submit feedback"
    );

    let response_json: Value = response.json().await.unwrap();
    Uuid::parse_str(response_json["feedback_id"].as_str().unwrap()).unwrap()
}

/// Helper function to submit feedback for an episode
async fn submit_episode_feedback(
    client: &Client,
    episode_id: Uuid,
    metric_name: &str,
    value: Value,
) -> Uuid {
    let payload = json!({
        "episode_id": episode_id,
        "metric_name": metric_name,
        "value": value,
    });

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "Failed to submit episode feedback"
    );

    let response_json: Value = response.json().await.unwrap();
    Uuid::parse_str(response_json["feedback_id"].as_str().unwrap()).unwrap()
}

// Tests for inference stats endpoint

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_chat_function() {
    let client = Client::new();

    // First get the current count
    let url = get_gateway_endpoint("/internal/functions/basic_test/inference-stats");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceStatsResponse = serde_json::from_str(&body).unwrap();
    let initial_count = initial_response.inference_count;

    // Create a new inference
    let (_inference_id, _episode_id) = create_inference(&client, "basic_test").await;

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Verify the count increased
    let resp = client.get(url).send().await.unwrap();
    let response: InferenceStatsResponse = resp.json().await.unwrap();
    assert!(
        response.inference_count > initial_count,
        "Expected inference_count to increase from {} after creating inference, got {}",
        initial_count,
        response.inference_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_json_function() {
    let client = Client::new();

    // First get the current count
    let url = get_gateway_endpoint("/internal/functions/json_success/inference-stats");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceStatsResponse = serde_json::from_str(&body).unwrap();
    let initial_count = initial_response.inference_count;

    // Create a new inference for json_success function
    let inference_payload = json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "TestBot"},
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

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Verify the count increased
    let resp = client.get(url).send().await.unwrap();
    let response: InferenceStatsResponse = resp.json().await.unwrap();
    assert!(
        response.inference_count > initial_count,
        "Expected inference_count to increase from {} after creating inference, got {}",
        initial_count,
        response.inference_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_chat_function_with_variant() {
    let client = Client::new();
    // Use the "test" variant which exists in basic_test function
    let url =
        get_gateway_endpoint("/internal/functions/basic_test/inference-stats?variant_name=test");

    let resp = client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let _response: InferenceStatsResponse = serde_json::from_str(&body).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_json_function_with_variant() {
    let client = Client::new();
    // Use the "test" variant which exists in json_success function
    let url =
        get_gateway_endpoint("/internal/functions/json_success/inference-stats?variant_name=test");

    let resp = client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let _response: InferenceStatsResponse = serde_json::from_str(&body).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_unknown_function() {
    let client = Client::new();
    let url = get_gateway_endpoint("/internal/functions/nonexistent_function/inference-stats");

    let resp = client.get(url).send().await.unwrap();

    assert!(
        !resp.status().is_success(),
        "Request for unknown function should fail"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_unknown_variant() {
    let client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/basic_test/inference-stats?variant_name=nonexistent_variant",
    );

    let resp = client.get(url).send().await.unwrap();

    assert!(
        !resp.status().is_success(),
        "Request for unknown variant should fail"
    );
}

// Tests for feedback stats endpoint

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_float_metric() {
    let client = Client::new();

    // Create an inference
    let (inference_id, _episode_id) = create_inference(&client, "basic_test").await;

    // Get initial feedback stats
    let url = get_gateway_endpoint("/internal/functions/basic_test/inference-stats/brevity_score");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceWithFeedbackStatsResponse = serde_json::from_str(&body).unwrap();
    let initial_feedback_count = initial_response.feedback_count;

    // Submit feedback for the inference
    let _feedback_id =
        submit_inference_feedback(&client, inference_id, "brevity_score", json!(0.85)).await;

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Verify feedback count increased
    let resp = client.get(url).send().await.unwrap();
    let response: InferenceWithFeedbackStatsResponse = resp.json().await.unwrap();
    assert!(
        response.feedback_count > initial_feedback_count,
        "Expected feedback_count to increase from {} after submitting feedback, got {}",
        initial_feedback_count,
        response.feedback_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_boolean_metric() {
    let client = Client::new();

    // Create an inference
    let (inference_id, _episode_id) = create_inference(&client, "basic_test").await;

    // Get initial feedback stats
    let url = get_gateway_endpoint("/internal/functions/basic_test/inference-stats/task_success");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceWithFeedbackStatsResponse = serde_json::from_str(&body).unwrap();
    let initial_feedback_count = initial_response.feedback_count;

    // Submit boolean feedback for the inference
    let _feedback_id =
        submit_inference_feedback(&client, inference_id, "task_success", json!(true)).await;

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Verify feedback count increased
    let resp = client.get(url).send().await.unwrap();
    let response: InferenceWithFeedbackStatsResponse = resp.json().await.unwrap();
    assert!(
        response.feedback_count > initial_feedback_count,
        "Expected feedback_count to increase from {} after submitting feedback, got {}",
        initial_feedback_count,
        response.feedback_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_with_threshold() {
    let client = Client::new();

    // Create an inference and submit feedback with a specific value
    let (inference_id, _episode_id) = create_inference(&client, "basic_test").await;
    let _feedback_id =
        submit_inference_feedback(&client, inference_id, "brevity_score", json!(0.75)).await;

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Get stats without threshold
    let url_total =
        get_gateway_endpoint("/internal/functions/basic_test/inference-stats/brevity_score");
    let resp_total = client.get(url_total).send().await.unwrap();
    let status_total = resp_total.status();
    let body_total = resp_total.text().await.unwrap();
    assert!(
        status_total.is_success(),
        "Expected success status for total, got {status_total}: {body_total}"
    );
    let total_response: InferenceWithFeedbackStatsResponse =
        serde_json::from_str(&body_total).unwrap();

    // Get stats with threshold > our feedback value (should have fewer results)
    let url_high_threshold = get_gateway_endpoint(
        "/internal/functions/basic_test/inference-stats/brevity_score?threshold=0.9",
    );
    let resp_high = client.get(url_high_threshold).send().await.unwrap();
    let status_high = resp_high.status();
    let body_high = resp_high.text().await.unwrap();
    assert!(
        status_high.is_success(),
        "Expected success status for high threshold, got {status_high}: {body_high}"
    );
    let high_threshold_response: InferenceWithFeedbackStatsResponse =
        serde_json::from_str(&body_high).unwrap();

    // Get stats with threshold < our feedback value (should include our feedback)
    let url_low_threshold = get_gateway_endpoint(
        "/internal/functions/basic_test/inference-stats/brevity_score?threshold=0.5",
    );
    let resp_low = client.get(url_low_threshold).send().await.unwrap();
    let status_low = resp_low.status();
    let body_low = resp_low.text().await.unwrap();
    assert!(
        status_low.is_success(),
        "Expected success status for low threshold, got {status_low}: {body_low}"
    );
    let low_threshold_response: InferenceWithFeedbackStatsResponse =
        serde_json::from_str(&body_low).unwrap();

    // Verify that high threshold has fewer or equal inferences than low threshold
    assert!(
        high_threshold_response.inference_count <= low_threshold_response.inference_count,
        "High threshold ({}) inference_count ({}) should be <= low threshold inference_count ({})",
        0.9,
        high_threshold_response.inference_count,
        low_threshold_response.inference_count
    );

    // Total feedback count should be >= the inference counts with thresholds
    assert!(
        total_response.feedback_count >= high_threshold_response.inference_count,
        "Total feedback_count ({}) should be >= high threshold inference_count ({})",
        total_response.feedback_count,
        high_threshold_response.inference_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_demonstration() {
    let client = Client::new();
    // Use json_success which should be able to have demonstrations
    let url =
        get_gateway_endpoint("/internal/functions/json_success/inference-stats/demonstration");

    let resp = client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let _response: InferenceWithFeedbackStatsResponse = serde_json::from_str(&body).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_unknown_function() {
    let client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/nonexistent_function/inference-stats/some_metric",
    );

    let resp = client.get(url).send().await.unwrap();

    assert!(
        !resp.status().is_success(),
        "Request for unknown function should fail"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_unknown_metric() {
    let client = Client::new();
    let url =
        get_gateway_endpoint("/internal/functions/basic_test/inference-stats/nonexistent_metric");

    let resp = client.get(url).send().await.unwrap();

    assert!(
        !resp.status().is_success(),
        "Request for unknown metric should fail"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_episode_level_boolean_metric() {
    let client = Client::new();

    // Create an inference to get an episode_id
    let (_inference_id, episode_id) = create_inference(&client, "weather_helper").await;

    // Get initial feedback stats
    let url =
        get_gateway_endpoint("/internal/functions/weather_helper/inference-stats/goal_achieved");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceWithFeedbackStatsResponse = serde_json::from_str(&body).unwrap();
    let initial_feedback_count = initial_response.feedback_count;

    // Submit episode-level boolean feedback
    let _feedback_id =
        submit_episode_feedback(&client, episode_id, "goal_achieved", json!(true)).await;

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Verify feedback count increased
    let resp = client.get(url).send().await.unwrap();
    let response: InferenceWithFeedbackStatsResponse = resp.json().await.unwrap();
    assert!(
        response.feedback_count > initial_feedback_count,
        "Expected feedback_count to increase from {} after submitting episode feedback, got {}",
        initial_feedback_count,
        response.feedback_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_episode_level_float_metric() {
    let client = Client::new();

    // Create an inference to get an episode_id
    let (_inference_id, episode_id) = create_inference(&client, "weather_helper").await;

    // Get initial feedback stats
    let url =
        get_gateway_endpoint("/internal/functions/weather_helper/inference-stats/user_rating");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceWithFeedbackStatsResponse = serde_json::from_str(&body).unwrap();
    let initial_feedback_count = initial_response.feedback_count;

    // Submit episode-level float feedback
    let _feedback_id =
        submit_episode_feedback(&client, episode_id, "user_rating", json!(4.5)).await;

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Verify feedback count increased
    let resp = client.get(url).send().await.unwrap();
    let response: InferenceWithFeedbackStatsResponse = resp.json().await.unwrap();
    assert!(
        response.feedback_count > initial_feedback_count,
        "Expected feedback_count to increase from {} after submitting episode feedback, got {}",
        initial_feedback_count,
        response.feedback_count
    );
}

// Tests from fixtures for write_haiku and extract_entities functions

/// Helper function to call the inference stats endpoint (for fixture-based tests)
async fn get_inference_stats_fixture(
    function_name: &str,
    query_params: &str,
) -> Result<InferenceStatsResponse, Box<dyn std::error::Error>> {
    let http_client = Client::new();
    let url = format!(
        "/internal/functions/{function_name}/inference-stats{}",
        if query_params.is_empty() {
            String::new()
        } else {
            format!("?{query_params}")
        }
    );

    let resp = http_client.get(get_gateway_endpoint(&url)).send().await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await?;
        return Err(format!("Request failed: status={status}, body={body}").into());
    }

    let response: InferenceStatsResponse = resp.json().await?;
    Ok(response)
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inference_stats_basic() {
    let res = get_inference_stats_fixture("write_haiku", "")
        .await
        .unwrap();

    assert!(
        res.inference_count >= 804,
        "Expected at least 804 inferences for write_haiku, got {}",
        res.inference_count
    );

    // Should not have stats_by_variant when group_by is not specified
    assert!(
        res.stats_by_variant.is_none(),
        "stats_by_variant should not be present without group_by"
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inference_stats_with_variant_filter() {
    let res = get_inference_stats_fixture("basic_test", "variant_name=test")
        .await
        .unwrap();

    assert!(
        res.inference_count >= 1,
        "Expected at least 1 inference for basic_test/test, got {}",
        res.inference_count
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inference_stats_group_by_variant() {
    let res = get_inference_stats_fixture("basic_test", "group_by=variant")
        .await
        .unwrap();

    let total_count = res.inference_count;
    assert!(
        total_count >= 1,
        "Expected at least 1 inference for basic_test, got {total_count}"
    );

    let stats_by_variant = res
        .stats_by_variant
        .expect("Expected stats_by_variant to be present");

    // Should have at least 2 variants
    assert!(
        stats_by_variant.len() >= 2,
        "Expected at least 2 variants, got {}",
        stats_by_variant.len()
    );

    // Sum of variant counts should equal total
    let sum_of_variants: u64 = stats_by_variant.iter().map(|v| v.inference_count).sum();
    assert_eq!(
        sum_of_variants, total_count,
        "Sum of variant counts ({sum_of_variants}) should equal total count ({total_count})"
    );

    // Variants should be ordered by inference_count DESC
    let counts: Vec<u64> = stats_by_variant.iter().map(|v| v.inference_count).collect();
    for i in 1..counts.len() {
        assert!(
            counts[i - 1] >= counts[i],
            "Counts should be in descending order"
        );
    }

    // Each variant should have last_used_at in ISO 8601 format
    for variant in &stats_by_variant {
        assert!(
            variant.last_used_at.contains('T') && variant.last_used_at.ends_with('Z'),
            "last_used_at should be in ISO 8601 format, got: {}",
            variant.last_used_at
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inference_stats_group_by_variant_json_function() {
    let res = get_inference_stats_fixture("extract_entities", "group_by=variant")
        .await
        .unwrap();

    assert!(
        res.inference_count >= 1,
        "Expected at least 1 inferences for extract_entities, got {}",
        res.inference_count
    );

    let stats_by_variant = res
        .stats_by_variant
        .expect("Expected stats_by_variant to be present");

    // Verify expected variant is present
    let variant_names: Vec<&str> = stats_by_variant
        .iter()
        .map(|v| v.variant_name.as_str())
        .collect();
    assert!(
        variant_names.contains(&"gpt4o_initial_prompt"),
        "Expected gpt4o_initial_prompt variant to be present"
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inference_stats_nonexistent_function() {
    let http_client = Client::new();
    let url = "/internal/functions/nonexistent_function/inference-stats";

    let resp = http_client
        .get(get_gateway_endpoint(url))
        .send()
        .await
        .unwrap();

    assert!(
        !resp.status().is_success(),
        "Request for nonexistent function should fail"
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inference_stats_nonexistent_variant() {
    let http_client = Client::new();
    let url = "/internal/functions/write_haiku/inference-stats?variant_name=nonexistent_variant";

    let resp = http_client
        .get(get_gateway_endpoint(url))
        .send()
        .await
        .unwrap();

    assert!(
        !resp.status().is_success(),
        "Request for nonexistent variant should fail"
    );
}
