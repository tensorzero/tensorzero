//! E2E tests for the inference stats endpoints.

use reqwest::Client;
use tensorzero_core::endpoints::internal::inference_stats::{
    InferenceStatsResponse, InferenceWithFeedbackStatsResponse,
};

use crate::common::get_gateway_endpoint;

// Tests for inference stats endpoint

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_chat_function() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/functions/write_haiku/inference-stats");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let response: InferenceStatsResponse = resp.json().await.unwrap();

    // The test data has at least 804 inferences for write_haiku (base fixture data)
    assert!(
        response.inference_count >= 804,
        "Expected at least 804 inferences for write_haiku, got {}",
        response.inference_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_json_function() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/functions/extract_entities/inference-stats");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let response: InferenceStatsResponse = resp.json().await.unwrap();

    // The test data has at least 604 inferences for extract_entities (base fixture data)
    assert!(
        response.inference_count >= 604,
        "Expected at least 604 inferences for extract_entities, got {}",
        response.inference_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_chat_function_with_variant() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/write_haiku/inference-stats?variant_name=initial_prompt_gpt4o_mini",
    );

    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let response: InferenceStatsResponse = resp.json().await.unwrap();

    // The test data has at least 649 inferences for write_haiku with variant initial_prompt_gpt4o_mini
    assert!(
        response.inference_count >= 649,
        "Expected at least 649 inferences for write_haiku/initial_prompt_gpt4o_mini, got {}",
        response.inference_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_json_function_with_variant() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/extract_entities/inference-stats?variant_name=gpt4o_initial_prompt",
    );

    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let response: InferenceStatsResponse = resp.json().await.unwrap();

    // The test data has at least 132 inferences for extract_entities with variant gpt4o_initial_prompt
    assert!(
        response.inference_count >= 132,
        "Expected at least 132 inferences for extract_entities/gpt4o_initial_prompt, got {}",
        response.inference_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_unknown_function() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/functions/nonexistent_function/inference-stats");

    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        !resp.status().is_success(),
        "Request for unknown function should fail"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_inference_stats_unknown_variant() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/write_haiku/inference-stats?variant_name=nonexistent_variant",
    );

    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        !resp.status().is_success(),
        "Request for unknown variant should fail"
    );
}

// Tests for feedback stats endpoint

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_float_metric() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/functions/write_haiku/inference-stats/haiku_rating");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let response: InferenceWithFeedbackStatsResponse = resp.json().await.unwrap();

    // The test database should have some haiku_rating feedbacks
    assert!(
        response.feedback_count > 0,
        "Should have feedbacks for haiku_rating metric on write_haiku"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_boolean_metric() {
    let http_client = Client::new();
    let url =
        get_gateway_endpoint("/internal/functions/extract_entities/inference-stats/exact_match");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let response: InferenceWithFeedbackStatsResponse = resp.json().await.unwrap();

    // The test database should have some exact_match feedbacks
    assert!(
        response.feedback_count > 0,
        "Should have feedbacks for exact_match metric on extract_entities"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_with_threshold() {
    let http_client = Client::new();

    // First get total feedbacks
    let url_total =
        get_gateway_endpoint("/internal/functions/write_haiku/inference-stats/haiku_rating");
    let resp_total = http_client.get(url_total).send().await.unwrap();
    assert!(resp_total.status().is_success());
    let total_response: InferenceWithFeedbackStatsResponse = resp_total.json().await.unwrap();

    // Then get with threshold
    let url_threshold = get_gateway_endpoint(
        "/internal/functions/write_haiku/inference-stats/haiku_rating?threshold=0.5",
    );
    let resp_threshold = http_client.get(url_threshold).send().await.unwrap();
    assert!(resp_threshold.status().is_success());
    let threshold_response: InferenceWithFeedbackStatsResponse =
        resp_threshold.json().await.unwrap();

    // Threshold inference_count should be < total feedback_count
    assert!(
        threshold_response.inference_count < total_response.feedback_count,
        "Threshold inference count ({}) should be < total feedback count ({})",
        threshold_response.inference_count,
        total_response.feedback_count
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_demonstration() {
    let http_client = Client::new();
    let url =
        get_gateway_endpoint("/internal/functions/extract_entities/inference-stats/demonstration");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let response: InferenceWithFeedbackStatsResponse = resp.json().await.unwrap();

    // The test database should have some demonstration feedbacks
    assert!(
        response.feedback_count > 0,
        "Should have demonstrations for extract_entities"
    );

    // For demonstrations, feedback_count == inference_count
    assert_eq!(
        response.feedback_count, response.inference_count,
        "For demonstrations, feedback_count should equal inference_count"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_unknown_function() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/nonexistent_function/inference-stats/some_metric",
    );

    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        !resp.status().is_success(),
        "Request for unknown function should fail"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_unknown_metric() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/extract_entities/inference-stats/nonexistent_metric",
    );

    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        !resp.status().is_success(),
        "Request for unknown metric should fail"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_episode_level_boolean_metric() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/extract_entities/inference-stats/exact_match_episode",
    );

    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let response: InferenceWithFeedbackStatsResponse = resp.json().await.unwrap();

    // We're verifying the endpoint works with episode-level metrics
    assert!(
        response.feedback_count > 0,
        "Should have feedbacks for boolean metric exact_match_episode on extract_entities"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_feedback_stats_episode_level_float_metric() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/functions/extract_entities/inference-stats/jaccard_similarity_episode",
    );

    let resp = http_client.get(url).send().await.unwrap();
    assert!(resp.status().is_success());

    let response: InferenceWithFeedbackStatsResponse = resp.json().await.unwrap();

    // We're verifying the endpoint works with episode-level metrics
    assert!(
        response.feedback_count > 0,
        "Should have feedbacks for float metric jaccard_similarity_episode on extract_entities"
    );
}
