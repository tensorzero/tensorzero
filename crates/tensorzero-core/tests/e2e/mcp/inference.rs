use googletest::prelude::*;
use serde_json::Value;

use super::common::{InferenceToolParams, McpTestClient, chat_input, chat_input_without_system};

#[gtest]
#[tokio::test]
async fn test_mcp_inference_basic() {
    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "inference",
            InferenceToolParams {
                function_name: Some("basic_test".to_string()),
                variant_name: None,
                input: chat_input("Hello"),
            },
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
            InferenceToolParams {
                function_name: Some("basic_test".to_string()),
                variant_name: Some("test".to_string()),
                input: chat_input("Hello"),
            },
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
            InferenceToolParams {
                function_name: Some("nonexistent_function".to_string()),
                variant_name: None,
                input: chat_input_without_system("Hello"),
            },
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
