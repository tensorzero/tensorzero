//! Integration tests for RunEvaluationTool.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use durable::MIGRATOR;
use durable_tools::{CacheEnabledMode, ErasedSimpleTool, RunEvaluationResponse, SimpleToolContext};
use sqlx::PgPool;
use uuid::Uuid;

use autopilot_tools::AutopilotToolSideInfo;
use autopilot_tools::tools::{RunEvaluationTool, RunEvaluationToolParams};
use common::MockTensorZeroClient;

/// Create a mock RunEvaluationResponse for testing.
fn create_mock_run_evaluation_response() -> RunEvaluationResponse {
    let mut stats = HashMap::new();
    stats.insert(
        "accuracy".to_string(),
        durable_tools::EvaluatorStatsResponse {
            mean: 0.85,
            stderr: 0.02,
            count: 100,
        },
    );
    RunEvaluationResponse {
        evaluation_run_id: Uuid::now_v7(),
        num_datapoints: 100,
        num_successes: 95,
        num_errors: 5,
        stats,
    }
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_run_evaluation_tool_with_dataset_name(pool: PgPool) {
    // Create mock response
    let mock_response = create_mock_run_evaluation_response();
    let expected_response = mock_response.clone();

    // Prepare test data
    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = RunEvaluationToolParams {
        evaluation_name: "test_evaluation".to_string(),
        dataset_name: Some("test_dataset".to_string()),
        datapoint_ids: None,
        variant_name: "test_variant".to_string(),
        concurrency: 5,
        max_datapoints: Some(50),
        precision_targets: HashMap::new(),
        inference_cache: CacheEnabledMode::Off,
    };

    let side_info = AutopilotToolSideInfo {
        episode_id,
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    // Create mock client with expectations
    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_run_evaluation()
        .withf(move |params| {
            params.evaluation_name == "test_evaluation"
                && params.dataset_name == Some("test_dataset".to_string())
                && params.datapoint_ids.is_none()
                && params.variant_name == "test_variant"
                && params.concurrency == 5
                && params.max_datapoints == Some(50)
                && params.precision_targets.is_empty()
        })
        .returning(move |_| Ok(mock_response.clone()));

    // Create the tool and context
    let tool = RunEvaluationTool;
    let t0_client: Arc<dyn durable_tools::TensorZeroClient> = Arc::new(mock_client);
    let ctx = SimpleToolContext::new(&pool, &t0_client);

    // Execute the tool
    let result = tool
        .execute_erased(
            serde_json::to_value(&llm_params).expect("Failed to serialize llm_params"),
            serde_json::to_value(&side_info).expect("Failed to serialize side_info"),
            ctx,
            "test-idempotency-key",
        )
        .await
        .expect("RunEvaluationTool execution should succeed");

    // Verify the response
    let response: RunEvaluationResponse =
        serde_json::from_value(result).expect("Failed to deserialize response");
    assert_eq!(response.num_datapoints, expected_response.num_datapoints);
    assert_eq!(response.num_successes, expected_response.num_successes);
    assert_eq!(response.num_errors, expected_response.num_errors);
    assert!(response.stats.contains_key("accuracy"));
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_run_evaluation_tool_with_datapoint_ids(pool: PgPool) {
    // Create mock response
    let mock_response = create_mock_run_evaluation_response();

    // Prepare test data
    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let datapoint_ids = vec![Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7()];
    let expected_datapoint_ids = datapoint_ids.clone();

    let llm_params = RunEvaluationToolParams {
        evaluation_name: "test_evaluation".to_string(),
        dataset_name: None,
        datapoint_ids: Some(datapoint_ids),
        variant_name: "test_variant".to_string(),
        concurrency: 10, // default
        max_datapoints: None,
        precision_targets: HashMap::new(),
        inference_cache: CacheEnabledMode::Off,
    };

    let side_info = AutopilotToolSideInfo {
        episode_id,
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    // Create mock client with expectations
    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_run_evaluation()
        .withf(move |params| {
            params.evaluation_name == "test_evaluation"
                && params.dataset_name.is_none()
                && params.datapoint_ids == Some(expected_datapoint_ids.clone())
                && params.variant_name == "test_variant"
                && params.concurrency == 10
                && params.max_datapoints.is_none()
        })
        .returning(move |_| Ok(mock_response.clone()));

    // Create the tool and context
    let tool = RunEvaluationTool;
    let t0_client: Arc<dyn durable_tools::TensorZeroClient> = Arc::new(mock_client);
    let ctx = SimpleToolContext::new(&pool, &t0_client);

    // Execute the tool
    let result = tool
        .execute_erased(
            serde_json::to_value(&llm_params).expect("Failed to serialize llm_params"),
            serde_json::to_value(&side_info).expect("Failed to serialize side_info"),
            ctx,
            "test-idempotency-key",
        )
        .await
        .expect("RunEvaluationTool execution should succeed");

    // The result should be a JSON object
    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_run_evaluation_tool_error_handling(pool: PgPool) {
    // Prepare test data
    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = RunEvaluationToolParams {
        evaluation_name: "nonexistent_evaluation".to_string(),
        dataset_name: Some("test_dataset".to_string()),
        datapoint_ids: None,
        variant_name: "test_variant".to_string(),
        concurrency: 10,
        max_datapoints: None,
        precision_targets: HashMap::new(),
        inference_cache: CacheEnabledMode::Off,
    };

    let side_info = AutopilotToolSideInfo {
        episode_id,
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    // Create mock client that returns an error
    let mut mock_client = MockTensorZeroClient::new();
    mock_client.expect_run_evaluation().returning(|_| {
        Err(durable_tools::TensorZeroClientError::Evaluation(
            "Evaluation 'nonexistent_evaluation' not found in config".to_string(),
        ))
    });

    // Create the tool and context
    let tool = RunEvaluationTool;
    let t0_client: Arc<dyn durable_tools::TensorZeroClient> = Arc::new(mock_client);
    let ctx = SimpleToolContext::new(&pool, &t0_client);

    // Execute the tool - should fail
    let result = tool
        .execute_erased(
            serde_json::to_value(&llm_params).expect("Failed to serialize llm_params"),
            serde_json::to_value(&side_info).expect("Failed to serialize side_info"),
            ctx,
            "test-idempotency-key",
        )
        .await;

    assert!(
        result.is_err(),
        "Should return an error for invalid evaluation"
    );
}
