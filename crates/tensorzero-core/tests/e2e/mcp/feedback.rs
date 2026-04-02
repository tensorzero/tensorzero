use googletest::prelude::*;
use serde_json::{Value, json};

use super::common::{McpTestClient, insert_inference, poll_mcp_tool};

#[gtest]
#[tokio::test]
async fn test_mcp_feedback_comment() {
    let (inference_id, _) = insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "feedback",
            json!({
                "inference_id": inference_id,
                "metric_name": "comment",
                "value": "This was a great response",
            }),
        )
        .await;

    expect_that!(response["feedback_id"].as_str(), some(not(eq(""))));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_feedback_boolean_metric() {
    let (inference_id, _) = insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "feedback",
            json!({
                "inference_id": inference_id,
                "metric_name": "task_success",
                "value": true,
            }),
        )
        .await;

    expect_that!(response["feedback_id"].as_str(), some(not(eq(""))));

    // Verify the feedback shows up via get_feedback_by_target_id
    let feedback_response = poll_mcp_tool(
        &mcp,
        "get_feedback_by_target_id",
        json!({
            "target_id": inference_id,
            "limit": 10,
        }),
        |r| r["feedback"].as_array().is_some_and(|f| !f.is_empty()),
    )
    .await;

    let feedback = feedback_response["feedback"]
        .as_array()
        .expect("Expected feedback array");
    expect_that!(feedback.len(), gt(0));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_feedback_episode_level() {
    let (_, episode_id) = insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "feedback",
            json!({
                "episode_id": episode_id,
                "metric_name": "comment",
                "value": "Episode-level feedback",
            }),
        )
        .await;

    expect_that!(response["feedback_id"].as_str(), some(not(eq(""))));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_feedback_missing_target() {
    let mcp = McpTestClient::connect().await;
    let result = mcp
        .call_tool_raw(
            "feedback",
            json!({
                "metric_name": "comment",
                "value": "No target",
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
