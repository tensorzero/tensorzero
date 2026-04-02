use googletest::prelude::*;
use serde_json::{Value, json};

use super::common::{McpTestClient, create_test_datapoint};

#[gtest]
#[tokio::test]
async fn test_mcp_run_evaluation_named_evaluation() {
    let mcp = McpTestClient::connect().await;

    // Create a datapoint for the evaluation to run on
    let datapoint_id = create_test_datapoint("mcp_eval_test").await;

    let response: Value = mcp
        .call_tool(
            "run_evaluation",
            json!({
                "evaluation_name": "test_evaluation",
                "datapoint_ids": [datapoint_id],
                "variant_name": "test",
            }),
        )
        .await;

    expect_that!(response["evaluation_run_id"].as_str(), some(not(eq(""))));
    expect_that!(response["num_datapoints"].as_u64(), some(eq(1)));
    expect_that!(response["num_successes"].as_u64(), some(eq(1)));
    expect_that!(response["num_errors"].as_u64(), some(eq(0)));

    // Check that stats are returned for all evaluators
    let stats = response["stats"]
        .as_object()
        .expect("stats should be an object");
    expect_that!(stats.len(), gt(0));

    // happy_bool evaluator should return 1.0 (true)
    let happy_bool = &stats["happy_bool"];
    expect_that!(happy_bool["mean"].as_f64(), some(eq(1.0)));
    expect_that!(happy_bool["count"].as_u64(), some(eq(1)));

    // sad_bool evaluator should return 0.0 (false)
    let sad_bool = &stats["sad_bool"];
    expect_that!(sad_bool["mean"].as_f64(), some(eq(0.0)));
    expect_that!(sad_bool["count"].as_u64(), some(eq(1)));

    // zero evaluator should return 0.0
    let zero = &stats["zero"];
    expect_that!(zero["mean"].as_f64(), some(eq(0.0)));

    // one evaluator should return 1.0
    let one = &stats["one"];
    expect_that!(one["mean"].as_f64(), some(eq(1.0)));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_run_evaluation_with_dataset() {
    let mcp = McpTestClient::connect().await;

    // Create datapoints
    let _dp1 = create_test_datapoint("mcp_eval_dataset_test").await;
    let _dp2 = create_test_datapoint("mcp_eval_dataset_test").await;

    let response: Value = mcp
        .call_tool(
            "run_evaluation",
            json!({
                "evaluation_name": "test_evaluation",
                "dataset_name": "mcp_eval_dataset_test",
                "variant_name": "test",
            }),
        )
        .await;

    expect_that!(response["evaluation_run_id"].as_str(), some(not(eq(""))));
    expect_that!(response["num_datapoints"].as_u64(), some(ge(2)));
    expect_that!(response["num_successes"].as_u64(), some(ge(2)));
    expect_that!(response["num_errors"].as_u64(), some(eq(0)));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_run_evaluation_with_max_datapoints() {
    let mcp = McpTestClient::connect().await;

    // Create several datapoints but limit evaluation to 1
    let _dp1 = create_test_datapoint("mcp_eval_max_dp_test").await;
    let _dp2 = create_test_datapoint("mcp_eval_max_dp_test").await;
    let _dp3 = create_test_datapoint("mcp_eval_max_dp_test").await;

    let response: Value = mcp
        .call_tool(
            "run_evaluation",
            json!({
                "evaluation_name": "test_evaluation",
                "dataset_name": "mcp_eval_max_dp_test",
                "variant_name": "test",
                "max_datapoints": 1,
            }),
        )
        .await;

    expect_that!(response["evaluation_run_id"].as_str(), some(not(eq(""))));
    expect_that!(response["num_datapoints"].as_u64(), some(eq(1)));
    expect_that!(response["num_successes"].as_u64(), some(eq(1)));
    expect_that!(response["num_errors"].as_u64(), some(eq(0)));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_run_evaluation_nonexistent_evaluator_on_function() {
    let mcp = McpTestClient::connect().await;

    // basic_test has no evaluators defined at the function level
    let result = mcp
        .call_tool_raw(
            "run_evaluation",
            json!({
                "function_name": "basic_test",
                "evaluator_names": ["nonexistent_evaluator"],
                "dataset_name": "some_dataset",
                "variant_name": "test",
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_run_evaluation_invalid_evaluation_name() {
    let mcp = McpTestClient::connect().await;

    let result = mcp
        .call_tool_raw(
            "run_evaluation",
            json!({
                "evaluation_name": "nonexistent_evaluation",
                "dataset_name": "some_dataset",
                "variant_name": "test",
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}

#[gtest]
#[tokio::test]
async fn test_mcp_run_evaluation_invalid_params() {
    let mcp = McpTestClient::connect().await;

    // Providing both evaluation_name and function_name should error
    let result = mcp
        .call_tool_raw(
            "run_evaluation",
            json!({
                "evaluation_name": "test_evaluation",
                "function_name": "basic_test",
                "evaluator_names": ["exact_match"],
                "dataset_name": "some_dataset",
                "variant_name": "test",
            }),
        )
        .await;

    expect_that!(result.is_error.unwrap_or(false), eq(true));

    mcp.cancel().await;
}
