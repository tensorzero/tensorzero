use googletest::prelude::*;
use serde_json::{Value, json};

use super::common::{McpTestClient, insert_inference};

#[gtest]
#[tokio::test]
async fn test_mcp_create_datapoints_from_inferences_by_ids() {
    let (inference_id, _) = insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "create_datapoints_from_inferences",
            json!({
                "dataset_name": "mcp_test_from_inferences_ids",
                "params": {
                    "type": "inference_ids",
                    "inference_ids": [inference_id],
                },
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
async fn test_mcp_create_datapoints_from_inferences_by_query() {
    // Ensure at least one inference exists
    insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "create_datapoints_from_inferences",
            json!({
                "dataset_name": "mcp_test_from_inferences_query",
                "params": {
                    "type": "inference_query",
                    "function_name": "basic_test",
                    "limit": 1,
                },
            }),
        )
        .await;

    let ids = response["ids"].as_array().expect("Expected `ids` array");
    expect_that!(ids.len(), gt(0));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_create_datapoints_from_inferences_with_output_source() {
    let (inference_id, _) = insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "create_datapoints_from_inferences",
            json!({
                "dataset_name": "mcp_test_from_inferences_output_src",
                "params": {
                    "type": "inference_ids",
                    "inference_ids": [inference_id],
                    "output_source": "none",
                },
            }),
        )
        .await;

    let ids = response["ids"].as_array().expect("Expected `ids` array");
    expect_that!(ids.len(), eq(1));

    mcp.cancel().await;
}
