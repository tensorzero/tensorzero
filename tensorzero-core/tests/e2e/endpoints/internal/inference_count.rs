//! E2E tests for the inference count endpoints.
//!
//! These tests use the functions and metrics defined in tensorzero-core/tests/e2e/config/
//! - Functions: basic_test, weather_helper, json_success
//! - Metrics: task_success (bool/inference), brevity_score (float/inference),
//!   goal_achieved (bool/episode), user_rating (float/episode)

use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use tensorzero_core::endpoints::internal::inference_count::{
    InferenceCountResponse, InferenceWithFeedbackCountResponse,
    ListFunctionsWithInferenceCountResponse,
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

// Tests for inference count endpoint

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_count_chat_function() {
    // create_inference doesn't write to Postgres yet
    let client = Client::new();

    // First get the current count
    let url = get_gateway_endpoint("/internal/functions/basic_test/inference_count");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceCountResponse = serde_json::from_str(&body).unwrap();
    let initial_count = initial_response.inference_count;

    // Create a new inference
    let (_inference_id, _episode_id) = create_inference(&client, "basic_test").await;

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Verify the count increased
    let resp = client.get(url).send().await.unwrap();
    let response: InferenceCountResponse = resp.json().await.unwrap();
    assert!(
        response.inference_count > initial_count,
        "Expected inference_count to increase from {} after creating inference, got {}",
        initial_count,
        response.inference_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_count_json_function() {
    // create_inference doesn't write to Postgres yet

    let client = Client::new();

    // First get the current count
    let url = get_gateway_endpoint("/internal/functions/json_success/inference_count");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceCountResponse = serde_json::from_str(&body).unwrap();
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
    let response: InferenceCountResponse = resp.json().await.unwrap();
    assert!(
        response.inference_count > initial_count,
        "Expected inference_count to increase from {} after creating inference, got {}",
        initial_count,
        response.inference_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_count_chat_function_with_variant() {
    let client = Client::new();
    // Use the "test" variant which exists in basic_test function
    let url =
        get_gateway_endpoint("/internal/functions/basic_test/inference_count?variant_name=test");

    let resp = client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let _response: InferenceCountResponse = serde_json::from_str(&body).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_count_json_function_with_variant() {
    let client = Client::new();
    // Use the "test" variant which exists in json_success function
    let url =
        get_gateway_endpoint("/internal/functions/json_success/inference_count?variant_name=test");

    let resp = client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let _response: InferenceCountResponse = serde_json::from_str(&body).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_count_unknown_function() {
    let client = Client::new();
    let url = get_gateway_endpoint("/internal/functions/nonexistent_function/inference_count");

    let resp = client.get(url).send().await.unwrap();

    assert!(
        !resp.status().is_success(),
        "Request for unknown function should fail"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_count_unknown_variant() {
    let client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/basic_test/inference_count?variant_name=nonexistent_variant",
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
    let url = get_gateway_endpoint("/internal/functions/basic_test/inference_count/brevity_score");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceWithFeedbackCountResponse = serde_json::from_str(&body).unwrap();
    let initial_feedback_count = initial_response.feedback_count;

    // Submit feedback for the inference
    let _feedback_id =
        submit_inference_feedback(&client, inference_id, "brevity_score", json!(0.85)).await;

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Verify feedback count increased
    let resp = client.get(url).send().await.unwrap();
    let response: InferenceWithFeedbackCountResponse = resp.json().await.unwrap();
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
    let url = get_gateway_endpoint("/internal/functions/basic_test/inference_count/task_success");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceWithFeedbackCountResponse = serde_json::from_str(&body).unwrap();
    let initial_feedback_count = initial_response.feedback_count;

    // Submit boolean feedback for the inference
    let _feedback_id =
        submit_inference_feedback(&client, inference_id, "task_success", json!(true)).await;

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Verify feedback count increased
    let resp = client.get(url).send().await.unwrap();
    let response: InferenceWithFeedbackCountResponse = resp.json().await.unwrap();
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
        get_gateway_endpoint("/internal/functions/basic_test/inference_count/brevity_score");
    let resp_total = client.get(url_total).send().await.unwrap();
    let status_total = resp_total.status();
    let body_total = resp_total.text().await.unwrap();
    assert!(
        status_total.is_success(),
        "Expected success status for total, got {status_total}: {body_total}"
    );
    let total_response: InferenceWithFeedbackCountResponse =
        serde_json::from_str(&body_total).unwrap();

    // Get stats with threshold > our feedback value (should have fewer results)
    let url_high_threshold = get_gateway_endpoint(
        "/internal/functions/basic_test/inference_count/brevity_score?threshold=0.9",
    );
    let resp_high = client.get(url_high_threshold).send().await.unwrap();
    let status_high = resp_high.status();
    let body_high = resp_high.text().await.unwrap();
    assert!(
        status_high.is_success(),
        "Expected success status for high threshold, got {status_high}: {body_high}"
    );
    let high_threshold_response: InferenceWithFeedbackCountResponse =
        serde_json::from_str(&body_high).unwrap();

    // Get stats with threshold < our feedback value (should include our feedback)
    let url_low_threshold = get_gateway_endpoint(
        "/internal/functions/basic_test/inference_count/brevity_score?threshold=0.5",
    );
    let resp_low = client.get(url_low_threshold).send().await.unwrap();
    let status_low = resp_low.status();
    let body_low = resp_low.text().await.unwrap();
    assert!(
        status_low.is_success(),
        "Expected success status for low threshold, got {status_low}: {body_low}"
    );
    let low_threshold_response: InferenceWithFeedbackCountResponse =
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
        get_gateway_endpoint("/internal/functions/json_success/inference_count/demonstration");

    let resp = client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let _response: InferenceWithFeedbackCountResponse = serde_json::from_str(&body).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_unknown_function() {
    let client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/nonexistent_function/inference_count/some_metric",
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
        get_gateway_endpoint("/internal/functions/basic_test/inference_count/nonexistent_metric");

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
        get_gateway_endpoint("/internal/functions/weather_helper/inference_count/goal_achieved");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceWithFeedbackCountResponse = serde_json::from_str(&body).unwrap();
    let initial_feedback_count = initial_response.feedback_count;

    // Submit episode-level boolean feedback
    let _feedback_id =
        submit_episode_feedback(&client, episode_id, "goal_achieved", json!(true)).await;

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Verify feedback count increased
    let resp = client.get(url).send().await.unwrap();
    let response: InferenceWithFeedbackCountResponse = resp.json().await.unwrap();
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
        get_gateway_endpoint("/internal/functions/weather_helper/inference_count/user_rating");
    let resp = client.get(url.clone()).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );
    let initial_response: InferenceWithFeedbackCountResponse = serde_json::from_str(&body).unwrap();
    let initial_feedback_count = initial_response.feedback_count;

    // Submit episode-level float feedback
    let _feedback_id =
        submit_episode_feedback(&client, episode_id, "user_rating", json!(4.5)).await;

    // Wait for ClickHouse to process
    sleep(Duration::from_millis(1000)).await;

    // Verify feedback count increased
    let resp = client.get(url).send().await.unwrap();
    let response: InferenceWithFeedbackCountResponse = resp.json().await.unwrap();
    assert!(
        response.feedback_count > initial_feedback_count,
        "Expected feedback_count to increase from {} after submitting episode feedback, got {}",
        initial_feedback_count,
        response.feedback_count
    );
}

// Tests from fixtures for write_haiku and extract_entities functions

/// Helper function to call the inference count endpoint (for fixture-based tests)
async fn get_inference_count_fixture(
    function_name: &str,
    query_params: &str,
) -> Result<InferenceCountResponse, Box<dyn std::error::Error>> {
    let http_client = Client::new();
    let url = format!(
        "/internal/functions/{function_name}/inference_count{}",
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

    let response: InferenceCountResponse = resp.json().await?;
    Ok(response)
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inference_count_basic() {
    let res = get_inference_count_fixture("write_haiku", "")
        .await
        .unwrap();

    assert!(
        res.inference_count >= 804,
        "Expected at least 804 inferences for write_haiku, got {}",
        res.inference_count
    );

    // Should not have count_by_variant when group_by is not specified
    assert!(
        res.count_by_variant.is_none(),
        "count_by_variant should not be present without group_by"
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inference_count_with_variant_filter() {
    let res = get_inference_count_fixture("write_haiku", "variant_name=initial_prompt_gpt4o_mini")
        .await
        .unwrap();

    assert!(
        res.inference_count >= 1,
        "Expected at least 1 inference for basic_test/test, got {}",
        res.inference_count
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inference_count_group_by_variant() {
    let res = get_inference_count_fixture("write_haiku", "group_by=variant")
        .await
        .unwrap();

    let total_count = res.inference_count;
    assert!(
        total_count >= 1,
        "Expected at least 1 inference for write_haiku, got {total_count}"
    );

    let count_by_variant = res
        .count_by_variant
        .expect("Expected count_by_variant to be present");

    // Should have at least 1 variant
    assert!(
        !count_by_variant.is_empty(),
        "Expected at least 1 variant, got {}",
        count_by_variant.len()
    );

    // Sum of variant counts should equal total
    let sum_of_variants: u64 = count_by_variant.iter().map(|v| v.inference_count).sum();
    assert_eq!(
        sum_of_variants, total_count,
        "Sum of variant counts ({sum_of_variants}) should equal total count ({total_count})"
    );

    // Variants should be ordered by inference_count DESC
    let counts: Vec<u64> = count_by_variant.iter().map(|v| v.inference_count).collect();
    for i in 1..counts.len() {
        assert!(
            counts[i - 1] >= counts[i],
            "Counts should be in descending order"
        );
    }

    // Each variant should have last_used_at in RFC 3339 format
    for variant in &count_by_variant {
        assert!(
            variant.last_used_at.contains('T') && variant.last_used_at.ends_with('Z'),
            "last_used_at should be in RFC 3339 format, got: {}",
            variant.last_used_at
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inference_count_group_by_variant_json_function() {
    let res = get_inference_count_fixture("extract_entities", "group_by=variant")
        .await
        .unwrap();

    assert!(
        res.inference_count >= 1,
        "Expected at least 1 inferences for extract_entities, got {}",
        res.inference_count
    );

    let count_by_variant = res
        .count_by_variant
        .expect("Expected count_by_variant to be present");

    // Verify expected variant is present
    let variant_names: Vec<&str> = count_by_variant
        .iter()
        .map(|v| v.variant_name.as_str())
        .collect();
    assert!(
        variant_names.contains(&"gpt4o_initial_prompt"),
        "Expected gpt4o_initial_prompt variant to be present"
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_inference_count_nonexistent_function() {
    let http_client = Client::new();
    let url = "/internal/functions/nonexistent_function/inference_count";

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
pub async fn test_get_inference_count_nonexistent_variant() {
    let http_client = Client::new();
    let url = "/internal/functions/write_haiku/inference_count?variant_name=nonexistent_variant";

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

// Tests for function throughput by variant endpoint

use tensorzero_core::endpoints::internal::inference_count::GetFunctionThroughputByVariantResponse;

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_function_throughput_by_variant_cumulative() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/write_haiku/throughput_by_variant?time_window=cumulative",
    );

    let resp = http_client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let response: GetFunctionThroughputByVariantResponse = serde_json::from_str(&body).unwrap();
    // write_haiku function has fixture data, so we should have throughput data
    assert!(
        !response.throughput.is_empty(),
        "Expected non-empty throughput data for write_haiku"
    );

    // For cumulative, all entries should have the same fixed period_start (epoch)
    for entry in &response.throughput {
        assert_eq!(
            entry.period_start.to_rfc3339(),
            "1970-01-01T00:00:00+00:00",
            "Cumulative should have epoch period_start"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_function_throughput_by_variant_week() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/write_haiku/throughput_by_variant?time_window=week&max_periods=5",
    );

    let resp = http_client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let response: GetFunctionThroughputByVariantResponse = serde_json::from_str(&body).unwrap();
    // May or may not have data depending on fixture data age
    // Just verify the response parses correctly
    for entry in &response.throughput {
        assert!(
            !entry.variant_name.is_empty(),
            "Variant name should not be empty"
        );
        assert!(entry.count > 0, "Count should be positive");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_function_throughput_by_variant_day() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/basic_test/throughput_by_variant?time_window=day&max_periods=30",
    );

    let resp = http_client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let _response: GetFunctionThroughputByVariantResponse = serde_json::from_str(&body).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_function_throughput_by_variant_nonexistent_function() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/nonexistent_function/throughput_by_variant?time_window=week",
    );

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        !resp.status().is_success(),
        "Request for nonexistent function should fail"
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_function_throughput_by_variant_missing_time_window() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/functions/write_haiku/throughput_by_variant");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        !resp.status().is_success(),
        "Request without time_window should fail"
    );
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_function_throughput_by_variant_default_max_periods() {
    let http_client = Client::new();
    // Don't specify max_periods, should use default of 10
    let url = get_gateway_endpoint(
        "/internal/functions/write_haiku/throughput_by_variant?time_window=month",
    );

    let resp = http_client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let _response: GetFunctionThroughputByVariantResponse = serde_json::from_str(&body).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_function_throughput_by_variant_extract_entities_week() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/extract_entities/throughput_by_variant?time_window=week&max_periods=10",
    );

    let resp = http_client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    // Check that the raw response has period_start in RFC 3339 format with milliseconds
    let rfc3339_regex =
        regex::Regex::new(r#""period_start"\s*:\s*"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z""#)
            .unwrap();
    assert!(
        rfc3339_regex.is_match(&body),
        "Response should contain period_start in RFC 3339 format with milliseconds, got: {body}"
    );

    let response: GetFunctionThroughputByVariantResponse = serde_json::from_str(&body).unwrap();
    // extract_entities has fixture data
    assert!(
        !response.throughput.is_empty(),
        "Expected non-empty throughput data for extract_entities"
    );

    // Check that all results have valid structure
    for entry in &response.throughput {
        assert!(
            !entry.variant_name.is_empty(),
            "Variant name should not be empty"
        );
        assert!(entry.count > 0, "Count should be positive");
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_function_throughput_by_variant_sorting_order() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/write_haiku/throughput_by_variant?time_window=day&max_periods=30",
    );

    let resp = http_client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let response: GetFunctionThroughputByVariantResponse = serde_json::from_str(&body).unwrap();

    // Check that results are sorted by period_start DESC, then variant_name DESC
    for i in 1..response.throughput.len() {
        let current = &response.throughput[i];
        let previous = &response.throughput[i - 1];

        if current.period_start == previous.period_start {
            // Same period, check variant_name DESC ordering
            assert!(
                current.variant_name <= previous.variant_name,
                "Within same period, variants should be sorted DESC. {} should come after {}",
                current.variant_name,
                previous.variant_name
            );
        } else {
            // Different periods, check period_start DESC ordering
            assert!(
                current.period_start <= previous.period_start,
                "Periods should be sorted DESC. {} should come after {}",
                current.period_start,
                previous.period_start
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
pub async fn test_get_function_throughput_by_variant_month() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/extract_entities/throughput_by_variant?time_window=month&max_periods=3",
    );

    let resp = http_client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    // Check that the raw response has period_start in RFC 3339 format with milliseconds
    let rfc3339_regex =
        regex::Regex::new(r#""period_start"\s*:\s*"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z""#)
            .unwrap();
    assert!(
        rfc3339_regex.is_match(&body),
        "Response should contain period_start in RFC 3339 format with milliseconds, got: {body}"
    );

    let response: GetFunctionThroughputByVariantResponse = serde_json::from_str(&body).unwrap();

    // Check that all results have valid structure
    for entry in &response.throughput {
        assert!(
            !entry.variant_name.is_empty(),
            "Variant name should not be empty"
        );
    }
}

// ============================================================================
// List Functions With Inference Count Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
pub async fn test_list_functions_with_inference_count() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/functions/inference_counts");

    let resp = http_client.get(url).send().await.unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    let response: ListFunctionsWithInferenceCountResponse = serde_json::from_str(&body).unwrap();

    // Should have at least 2 functions from fixture data
    assert!(
        response.functions.len() >= 2,
        "Expected at least 2 functions, got {}",
        response.functions.len()
    );

    // Check that we have write_haiku and extract_entities
    let function_names: Vec<&str> = response
        .functions
        .iter()
        .map(|f| f.function_name.as_str())
        .collect();
    assert!(
        function_names.contains(&"write_haiku"),
        "Expected write_haiku in results"
    );
    assert!(
        function_names.contains(&"extract_entities"),
        "Expected extract_entities in results"
    );

    // Each function should have a positive inference count
    for func in &response.functions {
        assert!(
            func.inference_count > 0,
            "Expected positive inference_count for {}, got {}",
            func.function_name,
            func.inference_count
        );
    }

    // Check that last_inference_timestamp is in RFC 3339 format with milliseconds
    let rfc3339_regex = regex::Regex::new(
        r#""last_inference_timestamp"\s*:\s*"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z""#,
    )
    .unwrap();
    assert!(
        rfc3339_regex.is_match(&body),
        "Response should contain last_inference_timestamp in RFC 3339 format with milliseconds"
    );
}
