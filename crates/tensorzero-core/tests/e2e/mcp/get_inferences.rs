use googletest::prelude::*;
use reqwest::Client;
use serde_json::{Value, json};
use uuid::Uuid;

use super::common::McpTestClient;
use crate::common::get_gateway_endpoint;

/// Insert an inference and return the inference_id.
async fn insert_inference(function_name: &str) -> String {
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
    body["inference_id"].as_str().unwrap().to_string()
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_inferences_by_id() {
    let inference_id = insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_inferences",
            json!({
                "ids": [inference_id],
            }),
        )
        .await;

    let inferences = response["inferences"]
        .as_array()
        .expect("Expected `inferences` array");
    expect_that!(inferences.len(), eq(1));
    expect_that!(
        inferences[0]["inference_id"].as_str().unwrap(),
        eq(inference_id.as_str())
    );

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_inferences_multiple_ids() {
    let id1 = insert_inference("basic_test").await;
    let id2 = insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_inferences",
            json!({
                "ids": [id1, id2],
            }),
        )
        .await;

    let inferences = response["inferences"]
        .as_array()
        .expect("Expected `inferences` array");
    expect_that!(inferences.len(), eq(2));

    let returned_ids: Vec<String> = inferences
        .iter()
        .map(|inf| inf["inference_id"].as_str().unwrap().to_string())
        .collect();
    expect_that!(returned_ids, unordered_elements_are![eq(&id1), eq(&id2)]);

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_inferences_unknown_id() {
    let unknown_id = Uuid::now_v7().to_string();

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_inferences",
            json!({
                "ids": [unknown_id],
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
async fn test_mcp_get_inferences_invalid_params() {
    let mcp = McpTestClient::connect().await;
    let result = mcp
        .call_tool_raw(
            "get_inferences",
            json!({
                "ids": "not_an_array",
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
