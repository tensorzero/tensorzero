use googletest::prelude::*;
use reqwest::Client;
use serde_json::{Value, json};
use uuid::Uuid;

use super::common::McpTestClient;
use crate::common::get_gateway_endpoint;

/// Insert an inference and return (inference_id, episode_id, function_name).
async fn insert_inference(function_name: &str) -> (String, String) {
    let client = Client::new();
    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&json!({
            "function_name": function_name,
            "input": {
                "system": {"assistant_name": "TestBot"},
                "messages": [{"role": "user", "content": "Hello"}]
            },
            "stream": false,
        }))
        .send()
        .await
        .unwrap();

    assert!(
        response.status().is_success(),
        "Inference request failed: {:?}",
        response.status()
    );
    let body: Value = response.json().await.unwrap();
    let inference_id = body["inference_id"].as_str().unwrap().to_string();
    let episode_id = body["episode_id"].as_str().unwrap().to_string();
    (inference_id, episode_id)
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_inferences_basic() {
    let (inference_id, _) = insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "list_inferences",
            json!({
                "function_name": "basic_test",
                "limit": 10,
            }),
        )
        .await;

    let inferences = response["inferences"]
        .as_array()
        .expect("Expected `inferences` array");
    expect_that!(inferences.len(), gt(0));

    let found = inferences
        .iter()
        .any(|inf| inf["inference_id"].as_str() == Some(inference_id.as_str()));
    expect_that!(found, eq(true));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_inferences_with_limit() {
    // Insert 3 inferences
    for _ in 0..3 {
        insert_inference("basic_test").await;
    }

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "list_inferences",
            json!({
                "function_name": "basic_test",
                "limit": 2,
            }),
        )
        .await;

    let inferences = response["inferences"]
        .as_array()
        .expect("Expected `inferences` array");
    expect_that!(inferences.len(), le(2));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_inferences_empty() {
    let mcp = McpTestClient::connect().await;
    let nonexistent = format!("nonexistent_function_{}", Uuid::now_v7());
    let response: Value = mcp
        .call_tool(
            "list_inferences",
            json!({
                "function_name": nonexistent,
            }),
        )
        .await;

    let inferences = response["inferences"]
        .as_array()
        .expect("Expected `inferences` array");
    expect_that!(inferences.len(), eq(0));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_inferences_invalid_params() {
    let mcp = McpTestClient::connect().await;
    let result = mcp
        .call_tool_raw(
            "list_inferences",
            json!({
                "limit": "not_a_number",
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
