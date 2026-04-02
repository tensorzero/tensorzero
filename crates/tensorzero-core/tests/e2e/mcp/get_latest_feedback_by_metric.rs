use googletest::prelude::*;
use serde_json::{Value, json};
use uuid::Uuid;

use super::common::{McpTestClient, insert_inference, poll_mcp_tool, submit_boolean_feedback};

#[gtest]
#[tokio::test]
async fn test_mcp_get_latest_feedback_by_metric_basic() {
    let (inference_id, _) = insert_inference("basic_test").await;
    let feedback_id = submit_boolean_feedback(&inference_id, "task_success", true).await;

    let mcp = McpTestClient::connect().await;
    let params = json!({ "target_id": inference_id });
    let response = poll_mcp_tool(&mcp, "get_latest_feedback_by_metric", params, |r| {
        r["feedback_id_by_metric"]
            .as_object()
            .is_some_and(|m| !m.is_empty())
    })
    .await;

    let feedback_map = response["feedback_id_by_metric"]
        .as_object()
        .expect("Expected `feedback_id_by_metric` object");
    expect_that!(feedback_map.len(), gt(0));
    expect_that!(
        feedback_map["task_success"].as_str().unwrap(),
        eq(feedback_id.as_str())
    );

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_latest_feedback_by_metric_no_feedback() {
    let (inference_id, _) = insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_latest_feedback_by_metric",
            json!({
                "target_id": inference_id,
            }),
        )
        .await;

    let feedback_map = response["feedback_id_by_metric"]
        .as_object()
        .expect("Expected `feedback_id_by_metric` object");
    expect_that!(feedback_map.len(), eq(0));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_latest_feedback_by_metric_unknown_target() {
    let unknown_id = Uuid::now_v7().to_string();

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "get_latest_feedback_by_metric",
            json!({
                "target_id": unknown_id,
            }),
        )
        .await;

    let feedback_map = response["feedback_id_by_metric"]
        .as_object()
        .expect("Expected `feedback_id_by_metric` object");
    expect_that!(feedback_map.len(), eq(0));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_get_latest_feedback_by_metric_returns_latest() {
    let (inference_id, _) = insert_inference("basic_test").await;

    // Submit two feedback entries for the same metric
    submit_boolean_feedback(&inference_id, "task_success", false).await;
    let latest_feedback_id = submit_boolean_feedback(&inference_id, "task_success", true).await;

    let mcp = McpTestClient::connect().await;
    let params = json!({ "target_id": inference_id });
    let response = poll_mcp_tool(&mcp, "get_latest_feedback_by_metric", params, |r| {
        r["feedback_id_by_metric"].as_object().is_some_and(|m| {
            m.get("task_success")
                .and_then(|v| v.as_str())
                .is_some_and(|id| id == latest_feedback_id)
        })
    })
    .await;

    let feedback_map = response["feedback_id_by_metric"]
        .as_object()
        .expect("Expected `feedback_id_by_metric` object");
    expect_that!(
        feedback_map["task_success"].as_str().unwrap(),
        eq(latest_feedback_id.as_str())
    );

    mcp.cancel().await;
}
