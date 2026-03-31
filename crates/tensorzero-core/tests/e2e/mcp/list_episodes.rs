use googletest::prelude::*;
use serde_json::{Value, json};

use super::common::{McpTestClient, insert_inference, poll_mcp_tool};

#[gtest]
#[tokio::test]
async fn test_mcp_list_episodes_basic() {
    let (_, episode_id) = insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response = poll_mcp_tool(
        &mcp,
        "list_episodes",
        json!({
            "limit": 10,
        }),
        |r| {
            r["episodes"].as_array().is_some_and(|eps| {
                eps.iter()
                    .any(|ep| ep["episode_id"].as_str() == Some(episode_id.as_str()))
            })
        },
    )
    .await;

    let episodes = response["episodes"]
        .as_array()
        .expect("Expected `episodes` array");
    expect_that!(episodes.len(), gt(0));

    let found = episodes
        .iter()
        .any(|ep| ep["episode_id"].as_str() == Some(episode_id.as_str()));
    expect_that!(found, eq(true));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_episodes_with_function_name() {
    insert_inference("basic_test").await;

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "list_episodes",
            json!({
                "limit": 10,
                "function_name": "basic_test",
            }),
        )
        .await;

    let episodes = response["episodes"]
        .as_array()
        .expect("Expected `episodes` array");
    expect_that!(episodes.len(), gt(0));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_episodes_with_limit() {
    for _ in 0..3 {
        insert_inference("basic_test").await;
    }

    let mcp = McpTestClient::connect().await;
    let response: Value = mcp
        .call_tool(
            "list_episodes",
            json!({
                "limit": 2,
            }),
        )
        .await;

    let episodes = response["episodes"]
        .as_array()
        .expect("Expected `episodes` array");
    expect_that!(episodes.len(), le(2));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_episodes_limit_over_100() {
    let mcp = McpTestClient::connect().await;
    let result = mcp
        .call_tool_raw(
            "list_episodes",
            json!({
                "limit": 101,
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_list_episodes_limit_zero() {
    let mcp = McpTestClient::connect().await;
    let result = mcp
        .call_tool_raw(
            "list_episodes",
            json!({
                "limit": 0,
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
