use googletest::prelude::*;
use serde_json::{Value, json};
use uuid::Uuid;

use super::common::{McpTestClient, insert_inference, poll_mcp_tool, submit_boolean_feedback};

#[gtest]
#[tokio::test]
async fn test_mcp_get_feedback_by_target_id_basic() {
    let (inference_id, _) = insert_inference("basic_test").await;
    submit_boolean_feedback(&inference_id, "task_success", true).await;

    let mcp = McpTestClient::connect().await;
    let params = json!({ "target_id": inference_id });
    let response = poll_mcp_tool(&mcp, "get_feedback_by_target_id", params, |r| {
        r["feedback"].as_array().is_some_and(|f| !f.is_empty())
    })
    .await;

    let feedback = response["feedback"]
        .as_array()
        .expect("Expected `feedback` array");
    expect_that!(feedback.len(), gt(0));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_feedback_by_target_id_with_limit() {
    let (inference_id, _) = insert_inference("basic_test").await;
    submit_boolean_feedback(&inference_id, "task_success", true).await;
    submit_boolean_feedback(&inference_id, "task_success", false).await;

    let mcp = McpTestClient::connect().await;
    // Poll until at least one feedback entry is visible, then test the limit
    let params = json!({ "target_id": inference_id, "limit": 1 });
    let response = poll_mcp_tool(&mcp, "get_feedback_by_target_id", params, |r| {
        r["feedback"].as_array().is_some_and(|f| !f.is_empty())
    })
    .await;

    let feedback = response["feedback"]
        .as_array()
        .expect("Expected `feedback` array");
    expect_that!(feedback.len(), le(1));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_feedback_by_target_id_no_feedback() {
    let (inference_id, _) = insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_feedback_by_target_id",
            json!({
                "target_id": inference_id,
            }),
        )
        .await;

    let feedback = response["feedback"]
        .as_array()
        .expect("Expected `feedback` array");
    expect_that!(feedback.len(), eq(0));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_feedback_by_target_id_unknown_target() {
    let unknown_id = Uuid::now_v7().to_string();

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_feedback_by_target_id",
            json!({
                "target_id": unknown_id,
            }),
        )
        .await;

    let feedback = response["feedback"]
        .as_array()
        .expect("Expected `feedback` array");
    expect_that!(feedback.len(), eq(0));

    mcp.cancel().await;
}
