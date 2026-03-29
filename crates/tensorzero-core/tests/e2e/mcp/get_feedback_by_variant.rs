use googletest::prelude::*;
use serde_json::{Value, json};

use super::common::{McpTestClient, insert_inference, poll_mcp_tool, submit_boolean_feedback};

#[gtest]
#[tokio::test]
async fn test_mcp_get_feedback_by_variant_basic() {
    let (inference_id, _) = insert_inference("basic_test").await;
    submit_boolean_feedback(&inference_id, "task_success", true).await;

    let mcp = McpTestClient::connect().await;
    let params = json!({
        "metric_name": "task_success",
        "function_name": "basic_test",
    });
    let response = poll_mcp_tool(&mcp, "get_feedback_by_variant", params, |r| {
        r["variants"].as_array().is_some_and(|v| !v.is_empty())
    })
    .await;

    let variants = response["variants"]
        .as_array()
        .expect("Expected `variants` array");
    expect_that!(variants.len(), gt(0));

    // Each variant should have the expected fields
    for variant in variants {
        expect_that!(variant["variant_name"].as_str(), some(not(eq(""))));
        expect_that!(variant["count"].as_u64(), some(gt(0)));
    }

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_feedback_by_variant_no_data() {
    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_feedback_by_variant",
            json!({
                "metric_name": "nonexistent_metric",
                "function_name": "basic_test",
            }),
        )
        .await;

    let variants = response["variants"]
        .as_array()
        .expect("Expected `variants` array");
    expect_that!(variants.len(), eq(0));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_feedback_by_variant_with_variant_filter() {
    let (inference_id, _) = insert_inference("basic_test").await;
    submit_boolean_feedback(&inference_id, "task_success", true).await;

    let mcp = McpTestClient::connect().await;
    let params = json!({
        "metric_name": "task_success",
        "function_name": "basic_test",
        "variant_names": ["nonexistent_variant"],
    });
    let response: Value = mcp.call_tool("get_feedback_by_variant", params).await;

    let variants = response["variants"]
        .as_array()
        .expect("Expected `variants` array");
    expect_that!(variants.len(), eq(0));

    mcp.cancel().await;
}
