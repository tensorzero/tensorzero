use googletest::prelude::*;
use serde_json::json;

use super::common::McpTestClient;

#[gtest]
#[tokio::test]
async fn test_mcp_launch_optimization_missing_data_source() {
    let mcp = McpTestClient::connect().await;

    // Neither output_source nor dataset_name provided
    let result = mcp
        .call_tool_raw(
            "launch_optimization",
            json!({
                "function_name": "basic_test",
                "template_variant_name": "test",
                "optimizer_config": {
                    "type": "dicl",
                    "embedding_model": "text-embedding-3-small",
                    "variant_name": "test_dicl",
                    "function_name": "basic_test",
                },
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_launch_optimization_both_data_sources() {
    let mcp = McpTestClient::connect().await;

    // Both output_source and dataset_name provided — should error
    let result = mcp
        .call_tool_raw(
            "launch_optimization",
            json!({
                "function_name": "basic_test",
                "template_variant_name": "test",
                "output_source": "inference",
                "dataset_name": "some_dataset",
                "optimizer_config": {
                    "type": "dicl",
                    "embedding_model": "text-embedding-3-small",
                    "variant_name": "test_dicl",
                    "function_name": "basic_test",
                },
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
