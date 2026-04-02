use googletest::prelude::*;
use serde_json::{Value, json};

use super::common::McpTestClient;

#[gtest]
#[tokio::test]
async fn test_mcp_create_datapoints_basic() {
    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "create_datapoints",
            json!({
                "dataset_name": "mcp_test_create_datapoints",
                "datapoints": [{
                    "type": "chat",
                    "function_name": "basic_test",
                    "input": {
                        "system": {"assistant_name": "TestBot"},
                        "messages": [{"role": "user", "content": "Hello"}]
                    },
                    "output": [{"type": "text", "text": "Hi there!"}]
                }]
            }),
        )
        .await;

    let ids = response["ids"].as_array().expect("Expected `ids` array");
    expect_that!(ids.len(), eq(1));
    expect_that!(ids[0].as_str(), some(not(eq(""))));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_create_datapoints_multiple() {
    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "create_datapoints",
            json!({
                "dataset_name": "mcp_test_create_datapoints_multi",
                "datapoints": [
                    {
                        "type": "chat",
                        "function_name": "basic_test",
                        "input": {
                            "system": {"assistant_name": "TestBot"},
                            "messages": [{"role": "user", "content": "First"}]
                        },
                        "output": [{"type": "text", "text": "Response 1"}]
                    },
                    {
                        "type": "chat",
                        "function_name": "basic_test",
                        "input": {
                            "system": {"assistant_name": "TestBot"},
                            "messages": [{"role": "user", "content": "Second"}]
                        },
                        "output": [{"type": "text", "text": "Response 2"}]
                    }
                ]
            }),
        )
        .await;

    let ids = response["ids"].as_array().expect("Expected `ids` array");
    expect_that!(ids.len(), eq(2));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_create_datapoints_with_tags() {
    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "create_datapoints",
            json!({
                "dataset_name": "mcp_test_create_datapoints_tags",
                "datapoints": [{
                    "type": "chat",
                    "function_name": "basic_test",
                    "input": {
                        "system": {"assistant_name": "TestBot"},
                        "messages": [{"role": "user", "content": "Hello"}]
                    },
                    "output": [{"type": "text", "text": "Hi!"}],
                    "tags": {"source": "mcp_test"}
                }]
            }),
        )
        .await;

    let ids = response["ids"].as_array().expect("Expected `ids` array");
    expect_that!(ids.len(), eq(1));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_create_datapoints_empty_list() {
    let mcp = McpTestClient::connect().await;
    let result = mcp
        .call_tool_raw(
            "create_datapoints",
            json!({
                "dataset_name": "mcp_test_create_empty",
                "datapoints": []
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
