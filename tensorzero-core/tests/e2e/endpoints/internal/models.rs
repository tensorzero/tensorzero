//! E2E tests for the model statistics endpoints.

use reqwest::Client;
use tensorzero_core::endpoints::internal::models::types::CountModelsResponse;

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
