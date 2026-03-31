use googletest::prelude::*;
use serde_json::json;

use super::common::McpTestClient;

#[gtest]
#[tokio::test]
async fn test_mcp_launch_gepa_missing_evaluation_name() {
    let mcp = McpTestClient::connect().await;

    // No evaluation_name — inline evaluators not yet supported
    let result = mcp
        .call_tool_raw(
            "launch_gepa",
            json!({
                "function_name": "basic_test",
                "dataset_name": "some_dataset",
                "evaluators": ["exact_match"],
                "analysis_model": "dummy::good",
                "mutation_model": "dummy::good",
                "max_iterations": 1,
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
