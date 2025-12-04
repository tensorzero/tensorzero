#![expect(clippy::print_stdout)]
use reqwest::Client;
use serde_json::Value;

use crate::common::get_gateway_endpoint;

/// Test the GET /internal/aggregated_feedback/{function_name} endpoint without filters
#[tokio::test]
async fn e2e_test_get_aggregated_feedback_no_filters() {
    let client = Client::new();

    let response = client
        .get(get_gateway_endpoint(
            "/internal/aggregated_feedback/json_success",
        ))
        .send()
        .await
        .unwrap();

    println!("Response status: {}", response.status());

    assert!(
        response.status().is_success(),
        "Expected success status code"
    );

    let response_json: Value = response.json().await.unwrap();
    println!("Response: {response_json:#?}");

    // Verify the response structure
    assert!(
        response_json.get("feedback").is_some(),
        "Response should have 'feedback' field"
    );

    let feedback = response_json.get("feedback").unwrap().as_array().unwrap();

    // Verify each feedback item has the expected fields
    for item in feedback {
        assert!(
            item.get("variant_name").is_some(),
            "Each item should have 'variant_name'"
        );
        assert!(
            item.get("metric_name").is_some(),
            "Each item should have 'metric_name'"
        );
        assert!(item.get("mean").is_some(), "Each item should have 'mean'");
        assert!(item.get("count").is_some(), "Each item should have 'count'");
        // variance can be null for sample size 1
    }
}

/// Test the GET /internal/aggregated_feedback/{function_name} endpoint with metric filter
#[tokio::test]
async fn e2e_test_get_aggregated_feedback_with_metric_filter() {
    let client = Client::new();

    let response = client
        .get(get_gateway_endpoint(
            "/internal/aggregated_feedback/json_success?metric_name=task_success",
        ))
        .send()
        .await
        .unwrap();

    println!("Response status: {}", response.status());

    assert!(
        response.status().is_success(),
        "Expected success status code"
    );

    let response_json: Value = response.json().await.unwrap();
    println!("Response with metric filter: {response_json:#?}");

    let feedback = response_json.get("feedback").unwrap().as_array().unwrap();

    // Verify all items have the filtered metric_name
    for item in feedback {
        let metric_name = item.get("metric_name").unwrap().as_str().unwrap();
        assert_eq!(
            metric_name, "task_success",
            "All items should have the filtered metric_name"
        );
    }
}

/// Test the GET /internal/aggregated_feedback/{function_name} endpoint with variant filter
#[tokio::test]
async fn e2e_test_get_aggregated_feedback_with_variant_filter() {
    let client = Client::new();

    let response = client
        .get(get_gateway_endpoint(
            "/internal/aggregated_feedback/json_success?variant_name=test",
        ))
        .send()
        .await
        .unwrap();

    println!("Response status: {}", response.status());

    assert!(
        response.status().is_success(),
        "Expected success status code"
    );

    let response_json: Value = response.json().await.unwrap();
    println!("Response with variant filter: {response_json:#?}");

    let feedback = response_json.get("feedback").unwrap().as_array().unwrap();

    // Verify all items have the filtered variant_name
    for item in feedback {
        let variant_name = item.get("variant_name").unwrap().as_str().unwrap();
        assert_eq!(
            variant_name, "test",
            "All items should have the filtered variant_name"
        );
    }
}

/// Test the GET /internal/aggregated_feedback/{function_name} endpoint with both filters
#[tokio::test]
async fn e2e_test_get_aggregated_feedback_with_both_filters() {
    let client = Client::new();

    let response = client
        .get(get_gateway_endpoint(
            "/internal/aggregated_feedback/json_success?variant_name=test&metric_name=task_success",
        ))
        .send()
        .await
        .unwrap();

    println!("Response status: {}", response.status());

    assert!(
        response.status().is_success(),
        "Expected success status code"
    );

    let response_json: Value = response.json().await.unwrap();
    println!("Response with both filters: {response_json:#?}");

    let feedback = response_json.get("feedback").unwrap().as_array().unwrap();

    // Verify all items have the filtered variant_name and metric_name
    for item in feedback {
        let variant_name = item.get("variant_name").unwrap().as_str().unwrap();
        let metric_name = item.get("metric_name").unwrap().as_str().unwrap();
        assert_eq!(
            variant_name, "test",
            "All items should have variant_name='test'"
        );
        assert_eq!(
            metric_name, "task_success",
            "All items should have metric_name='task_success'"
        );
    }
}

/// Test the GET /internal/aggregated_feedback/{function_name} endpoint with nonexistent function
#[tokio::test]
async fn e2e_test_get_aggregated_feedback_nonexistent_function() {
    let client = Client::new();

    let response = client
        .get(get_gateway_endpoint(
            "/internal/aggregated_feedback/nonexistent_function_12345",
        ))
        .send()
        .await
        .unwrap();

    println!("Response status: {}", response.status());

    assert!(
        response.status().is_success(),
        "Expected success status code even for nonexistent function"
    );

    let response_json: Value = response.json().await.unwrap();
    println!("Response for nonexistent function: {response_json:#?}");

    let feedback = response_json.get("feedback").unwrap().as_array().unwrap();
    assert!(
        feedback.is_empty(),
        "Should return empty array for nonexistent function"
    );
}
