use googletest::prelude::*;
use serde_json::{Value, json};

use super::common::{McpTestClient, create_test_datapoint};

#[gtest]
#[tokio::test]
async fn test_mcp_list_datasets_basic() {
    // Ensure at least one dataset exists by creating a datapoint
    create_test_datapoint("mcp_test_list_datasets").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp.call_tool("list_datasets", json!({})).await;

    let datasets = response["datasets"]
        .as_array()
        .expect("Expected `datasets` array");
    expect_that!(datasets.len(), gt(0));

    let found = datasets
        .iter()
        .any(|d| d["dataset_name"].as_str() == Some("mcp_test_list_datasets"));
    expect_that!(found, eq(true));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_datasets_with_function_filter() {
    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "list_datasets",
            json!({
                "function_name": "basic_test",
            }),
        )
        .await;

    let datasets = response["datasets"]
        .as_array()
        .expect("Expected `datasets` array");

    // All returned datasets should be associated with the function
    // (we just verify the call succeeds and returns a valid structure)
    for dataset in datasets {
        expect_that!(dataset["dataset_name"].as_str(), some(not(eq(""))));
    }

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_datasets_with_limit() {
    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "list_datasets",
            json!({
                "limit": 1,
            }),
        )
        .await;

    let datasets = response["datasets"]
        .as_array()
        .expect("Expected `datasets` array");
    expect_that!(datasets.len(), le(1));

    mcp.cancel().await;
}
