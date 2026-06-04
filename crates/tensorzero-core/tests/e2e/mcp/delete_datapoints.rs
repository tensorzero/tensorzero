use googletest::prelude::*;
use serde_json::Value;
use tensorzero::DeleteDatapointsRequest;
use uuid::Uuid;

use super::common::{DatasetToolParams, McpTestClient, create_test_datapoint};

#[gtest]
#[tokio::test]
async fn test_mcp_delete_datapoints_basic() {
    let dataset_name = "mcp_test_delete_datapoints";
    let datapoint_id = create_test_datapoint(dataset_name).await;
    let datapoint_uuid = datapoint_id.parse().expect("datapoint ID should be a UUID");

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "delete_datapoints",
            DatasetToolParams {
                dataset_name: dataset_name.to_string(),
                request: DeleteDatapointsRequest {
                    ids: vec![datapoint_uuid],
                },
            },
        )
        .await;

    let num_deleted = response["num_deleted_datapoints"]
        .as_u64()
        .expect("Expected `num_deleted_datapoints`");
    expect_that!(num_deleted, eq(1));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_delete_datapoints_unknown_id() {
    let unknown_id = Uuid::now_v7();

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "delete_datapoints",
            DatasetToolParams {
                dataset_name: "mcp_test_delete_unknown".to_string(),
                request: DeleteDatapointsRequest {
                    ids: vec![unknown_id],
                },
            },
        )
        .await;

    let num_deleted = response["num_deleted_datapoints"]
        .as_u64()
        .expect("Expected `num_deleted_datapoints`");
    expect_that!(num_deleted, eq(0));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_delete_datapoints_empty_ids() {
    let mcp = McpTestClient::connect().await;
    let result = mcp
        .call_tool_raw(
            "delete_datapoints",
            DatasetToolParams {
                dataset_name: "mcp_test_delete_empty".to_string(),
                request: DeleteDatapointsRequest { ids: vec![] },
            },
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
