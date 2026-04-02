//! E2E tests for the model statistics endpoints.

use reqwest::Client;
use tensorzero_core::endpoints::internal::models::types::{
    CountModelsResponse, GetCacheStatisticsResponse, GetModelLatencyResponse, GetModelUsageResponse,
};

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_count_models_endpoint() {
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
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/models/latency?time_window=week");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "model_latency request failed: status={:?}",
        resp.status()
    );

    let response: GetModelLatencyResponse = resp.json().await.unwrap();

    assert!(
        !response.quantiles.is_empty(),
        "Expected non-empty quantiles in latency response"
    );
    for datapoint in &response.data {
        assert_eq!(
            datapoint.response_time_ms_quantiles.len(),
            response.quantiles.len(),
            "response_time_ms_quantiles length should match quantiles length for model `{}`",
            datapoint.model_name
        );
        assert_eq!(
            datapoint.ttft_ms_quantiles.len(),
            response.quantiles.len(),
            "ttft_ms_quantiles length should match quantiles length for model `{}`",
            datapoint.model_name
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_model_usage_endpoint_missing_params() {
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
async fn test_cache_statistics_endpoint() {
    let http_client = Client::new();
    let url =
        get_gateway_endpoint("/internal/models/cache_statistics?time_window=week&max_periods=10");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "cache_statistics request failed: status={:?}",
        resp.status()
    );

    let response: GetCacheStatisticsResponse = resp.json().await.unwrap();

    // Verify we can deserialize the response and data points have expected structure
    for point in &response.data {
        assert!(
            !point.model_name.is_empty(),
            "model_name should not be empty"
        );
        assert!(
            !point.model_provider_name.is_empty(),
            "model_provider_name should not be empty"
        );
        // cache_read_ratio should be None or between 0.0 and 1.0
        if let Some(ratio) = point.cache_read_ratio {
            assert!(
                (0.0..=1.0).contains(&ratio),
                "cache_read_ratio should be between 0.0 and 1.0, got {ratio}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cache_statistics_with_model_filter() {
    let http_client = Client::new();
    let url = get_gateway_endpoint(
        "/internal/models/cache_statistics?time_window=week&max_periods=10&model_name=dummy%3A%3Agood",
    );

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "cache_statistics with model filter request failed: status={:?}",
        resp.status()
    );

    let response: GetCacheStatisticsResponse = resp.json().await.unwrap();

    // All returned data points should match the filter
    for point in &response.data {
        assert_eq!(
            point.model_name, "dummy::good",
            "Expected model_name to be `dummy::good`, got `{}`",
            point.model_name
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cache_statistics_missing_params() {
    let http_client = Client::new();
    // Missing required parameters
    let url = get_gateway_endpoint("/internal/models/cache_statistics");

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_client_error(),
        "Expected client error for missing params, got: {:?}",
        resp.status()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_model_latency_endpoint_missing_params() {
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
