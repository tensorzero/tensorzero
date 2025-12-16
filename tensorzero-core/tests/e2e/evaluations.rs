//! E2E tests for evaluation endpoints.

use reqwest::{Client, StatusCode};
use serde_json::Value;

use crate::common::get_gateway_endpoint;

/// Test the count evaluation runs endpoint.
/// This tests GET /internal/evaluations/run-stats
#[tokio::test]
async fn test_count_evaluation_runs() {
    let client = Client::new();

    let response = client
        .get(get_gateway_endpoint("/internal/evaluations/run-stats"))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body: Value = response.json().await.unwrap();

    // Verify the response structure
    assert!(
        body.get("count").is_some(),
        "Response should have `count` field"
    );

    let count = body.get("count").unwrap().as_u64().unwrap();

    // The e2e test database should have some evaluation runs
    assert!(
        count > 0,
        "Expected at least one evaluation run in the test database"
    );
}
