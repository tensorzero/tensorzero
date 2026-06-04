use googletest::prelude::*;
use serde_json::Value;
use tensorzero::{
    CreateDatapointsFromInferenceRequestParams, InferenceOutputSource, ListInferencesRequest,
};

use super::common::{CreateDatapointsFromInferencesToolParams, McpTestClient, insert_inference};

#[gtest]
#[tokio::test]
async fn test_mcp_create_datapoints_from_inferences_by_ids() {
    let (inference_id, _) = insert_inference("basic_test").await;
    let inference_uuid = inference_id.parse().expect("inference ID should be a UUID");

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "create_datapoints_from_inferences",
            CreateDatapointsFromInferencesToolParams {
                dataset_name: "mcp_test_from_inferences_ids".to_string(),
                params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
                    inference_ids: vec![inference_uuid],
                    output_source: None,
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
async fn test_mcp_create_datapoints_from_inferences_by_query() {
    // Ensure at least one inference exists
    insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "create_datapoints_from_inferences",
            CreateDatapointsFromInferencesToolParams {
                dataset_name: "mcp_test_from_inferences_query".to_string(),
                params: CreateDatapointsFromInferenceRequestParams::InferenceQuery {
                    query: Box::new(ListInferencesRequest {
                        function_name: Some("basic_test".to_string()),
                        limit: Some(1),
                        ..Default::default()
                    }),
                },
            },
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
    let inference_uuid = inference_id.parse().expect("inference ID should be a UUID");

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "create_datapoints_from_inferences",
            CreateDatapointsFromInferencesToolParams {
                dataset_name: "mcp_test_from_inferences_output_src".to_string(),
                params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
                    inference_ids: vec![inference_uuid],
                    output_source: Some(InferenceOutputSource::None),
                },
            },
        )
        .await;

    let ids = response["ids"].as_array().expect("Expected `ids` array");
    expect_that!(ids.len(), eq(1));

    mcp.cancel().await;
}
