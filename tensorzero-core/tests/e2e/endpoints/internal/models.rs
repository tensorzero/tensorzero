//! E2E tests for the model statistics endpoints.

use reqwest::Client;
use tensorzero_core::endpoints::internal::models::types::{
    CountModelsResponse, GetModelLatencyResponse, GetModelUsageResponse,
};

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_count_models_endpoint() {
    skip_for_postgres!();
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/models/count");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "count_models request failed: status={:?}",
        resp.status()
    );

    let response: CountModelsResponse = resp.json().await.unwrap();

    // The test database should have at least one model used
    assert!(
        response.model_count > 0,
        "Expected at least one model in the database"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_model_usage_endpoint() {
    skip_for_postgres!();
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/models/usage?time_window=week&max_periods=10");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "model_usage request failed: status={:?}",
        resp.status()
    );

    let response: GetModelUsageResponse = resp.json().await.unwrap();

    // The response should have data (may be empty if no recent usage)
    // Just verify we can deserialize the response successfully
    let _ = response.data;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_model_latency_endpoint() {
    skip_for_postgres!();
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/models/latency?time_window=week");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "model_latency request failed: status={:?}",
        resp.status()
    );

    let response: GetModelLatencyResponse = resp.json().await.unwrap();

    // The response should have data (may be empty if no recent usage)
    // Just verify we can deserialize the response successfully
    let _ = response.data;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_model_usage_endpoint_missing_params() {
    skip_for_postgres!();
    let http_client = Client::new();
    // Missing required parameters
    let url = get_gateway_endpoint("/internal/models/usage");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_client_error(),
        "Expected client error for missing params, got: {:?}",
        resp.status()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_model_latency_endpoint_missing_params() {
    skip_for_postgres!();
    let http_client = Client::new();
    // Missing required parameters
    let url = get_gateway_endpoint("/internal/models/latency");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_client_error(),
        "Expected client error for missing params, got: {:?}",
        resp.status()
    );
}
