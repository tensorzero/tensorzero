use googletest::prelude::*;
use serde_json::{Value, json};

use super::common::{McpTestClient, create_test_datapoint};

#[gtest]
#[tokio::test]
async fn test_mcp_update_datapoints_basic() {
    let dataset_name = "mcp_test_update_datapoints";
    let datapoint_id = create_test_datapoint(dataset_name).await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "update_datapoints",
            json!({
                "dataset_name": dataset_name,
                "datapoints": [{
                    "type": "chat",
                    "id": datapoint_id,
                    "output": [{"type": "text", "text": "Updated response"}]
                }]
            }),
        )
        .await;

    let ids = response["ids"].as_array().expect("Expected `ids` array");
    expect_that!(ids.len(), eq(1));
    // Update creates a new version with a new ID
    let new_id = ids[0].as_str().expect("Expected string ID");
    expect_that!(new_id, not(eq(datapoint_id.as_str())));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_update_datapoints_tags() {
    let dataset_name = "mcp_test_update_datapoints_tags";
    let datapoint_id = create_test_datapoint(dataset_name).await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "update_datapoints",
            json!({
                "dataset_name": dataset_name,
                "datapoints": [{
                    "type": "chat",
                    "id": datapoint_id,
                    "tags": {"updated": "true"}
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
async fn test_mcp_update_datapoints_empty_list() {
    let mcp = McpTestClient::connect().await;
    let result = mcp
        .call_tool_raw(
            "update_datapoints",
            json!({
                "dataset_name": "mcp_test_update_empty",
                "datapoints": []
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
