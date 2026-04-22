use googletest::prelude::*;
use serde_json::{Value, json};
use uuid::Uuid;

use super::common::{McpTestClient, create_test_datapoint};

#[gtest]
#[tokio::test]
async fn test_mcp_get_datapoints_by_id() {
    let dataset_name = "mcp_test_get_datapoints";
    let datapoint_id = create_test_datapoint(dataset_name).await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_datapoints",
            json!({
                "dataset_name": dataset_name,
                "ids": [datapoint_id],
            }),
        )
        .await;

    let datapoints = response["datapoints"]
        .as_array()
        .expect("Expected `datapoints` array");
    expect_that!(datapoints.len(), eq(1));
    expect_that!(
        datapoints[0]["id"].as_str().unwrap(),
        eq(datapoint_id.as_str())
    );

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_datapoints_multiple_ids() {
    let dataset_name = "mcp_test_get_datapoints_multi";
    let id1 = create_test_datapoint(dataset_name).await;
    let id2 = create_test_datapoint(dataset_name).await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_datapoints",
            json!({
                "dataset_name": dataset_name,
                "ids": [id1, id2],
            }),
        )
        .await;

    let datapoints = response["datapoints"]
        .as_array()
        .expect("Expected `datapoints` array");
    expect_that!(datapoints.len(), eq(2));

    let returned_ids: Vec<String> = datapoints
        .iter()
        .map(|dp| dp["id"].as_str().unwrap().to_string())
        .collect();
    expect_that!(returned_ids, unordered_elements_are![eq(&id1), eq(&id2)]);

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_datapoints_without_dataset_name() {
    let dataset_name = "mcp_test_get_datapoints_no_ds";
    let datapoint_id = create_test_datapoint(dataset_name).await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_datapoints",
            json!({
                "ids": [datapoint_id],
            }),
        )
        .await;

    let datapoints = response["datapoints"]
        .as_array()
        .expect("Expected `datapoints` array");
    expect_that!(datapoints.len(), eq(1));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_datapoints_unknown_id() {
    let unknown_id = Uuid::now_v7().to_string();

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_datapoints",
            json!({
                "ids": [unknown_id],
            }),
        )
        .await;

    let datapoints = response["datapoints"]
        .as_array()
        .expect("Expected `datapoints` array");
    expect_that!(datapoints.len(), eq(0));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_datapoints_empty_ids() {
    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_datapoints",
            json!({
                "ids": [],
            }),
        )
        .await;

    let datapoints = response["datapoints"]
        .as_array()
        .expect("Expected `datapoints` array");
    expect_that!(datapoints.len(), eq(0));

    mcp.cancel().await;
}
