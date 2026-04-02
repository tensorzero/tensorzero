use googletest::prelude::*;
use serde_json::{Value, json};

use super::common::McpTestClient;

#[gtest]
#[tokio::test]
async fn test_mcp_inference_basic() {
    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "inference",
            json!({
                "function_name": "basic_test",
                "input": {
                    "system": {"assistant_name": "TestBot"},
                    "messages": [{"role": "user", "content": "Hello"}]
                },
            }),
        )
        .await;

    expect_that!(response["inference_id"].as_str(), some(not(eq(""))));
    expect_that!(response["episode_id"].as_str(), some(not(eq(""))));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_inference_with_variant_name() {
    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "inference",
            json!({
                "function_name": "basic_test",
                "variant_name": "test",
                "input": {
                    "system": {"assistant_name": "TestBot"},
                    "messages": [{"role": "user", "content": "Hello"}]
                },
            }),
        )
        .await;

    expect_that!(response["inference_id"].as_str(), some(not(eq(""))));
    expect_that!(response["variant_name"].as_str(), some(eq("test")));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_inference_invalid_function() {
    let mcp = McpTestClient::connect().await;
    let result = mcp
        .call_tool_raw(
            "inference",
            json!({
                "function_name": "nonexistent_function",
                "input": {
                    "messages": [{"role": "user", "content": "Hello"}]
                },
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
