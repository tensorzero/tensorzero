use googletest::prelude::*;
use serde_json::Value;
use tensorzero::{CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsRequest};

use super::common::{DatasetToolParams, McpTestClient, chat_input, chat_output};

#[gtest]
#[tokio::test]
async fn test_mcp_create_datapoints_basic() {
    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "create_datapoints",
            DatasetToolParams {
                dataset_name: "mcp_test_create_datapoints".to_string(),
                request: CreateDatapointsRequest {
                    datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                        function_name: "basic_test".to_string(),
                        episode_id: None,
                        input: chat_input("Hello"),
                        output: Some(chat_output("Hi there!")),
                        dynamic_tool_params: Default::default(),
                        tags: None,
                        name: None,
                    })],
                },
            },
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
            DatasetToolParams {
                dataset_name: "mcp_test_create_datapoints_multi".to_string(),
                request: CreateDatapointsRequest {
                    datapoints: vec![
                        CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                            function_name: "basic_test".to_string(),
                            episode_id: None,
                            input: chat_input("First"),
                            output: Some(chat_output("Response 1")),
                            dynamic_tool_params: Default::default(),
                            tags: None,
                            name: None,
                        }),
                        CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                            function_name: "basic_test".to_string(),
                            episode_id: None,
                            input: chat_input("Second"),
                            output: Some(chat_output("Response 2")),
                            dynamic_tool_params: Default::default(),
                            tags: None,
                            name: None,
                        }),
                    ],
                },
            },
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
            DatasetToolParams {
                dataset_name: "mcp_test_create_datapoints_tags".to_string(),
                request: CreateDatapointsRequest {
                    datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                        function_name: "basic_test".to_string(),
                        episode_id: None,
                        input: chat_input("Hello"),
                        output: Some(chat_output("Hi!")),
                        dynamic_tool_params: Default::default(),
                        tags: Some([("source".to_string(), "mcp_test".to_string())].into()),
                        name: None,
                    })],
                },
            },
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
            DatasetToolParams {
                dataset_name: "mcp_test_create_empty".to_string(),
                request: CreateDatapointsRequest { datapoints: vec![] },
            },
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
