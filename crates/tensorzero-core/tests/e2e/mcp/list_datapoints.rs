use googletest::prelude::*;
use serde_json::{Value, json};

use super::common::{McpTestClient, create_test_datapoint};

#[gtest]
#[tokio::test]
async fn test_mcp_list_datapoints_basic() {
    let dataset_name = "mcp_test_list_datapoints";
    let datapoint_id = create_test_datapoint(dataset_name).await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "list_datapoints",
            json!({
                "dataset_name": dataset_name,
            }),
        )
        .await;

    let datapoints = response["datapoints"]
        .as_array()
        .expect("Expected `datapoints` array");
    expect_that!(datapoints.len(), gt(0));

    let found = datapoints
        .iter()
        .any(|dp| dp["id"].as_str() == Some(datapoint_id.as_str()));
    expect_that!(found, eq(true));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_datapoints_with_limit() {
    let dataset_name = "mcp_test_list_datapoints_limit";
    for _ in 0..3 {
        create_test_datapoint(dataset_name).await;
    }

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "list_datapoints",
            json!({
                "dataset_name": dataset_name,
                "limit": 2,
            }),
        )
        .await;

    let datapoints = response["datapoints"]
        .as_array()
        .expect("Expected `datapoints` array");
    expect_that!(datapoints.len(), le(2));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_datapoints_missing_dataset_name() {
    let mcp = McpTestClient::connect().await;
    let result = mcp.call_tool_raw("list_datapoints", json!({})).await;
    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
