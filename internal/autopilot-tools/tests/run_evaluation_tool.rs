//! Integration tests for RunEvaluationTool.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use durable::MIGRATOR;
use durable_tools::{
    ActionInput, ActionResponse, CacheEnabledMode, DatapointResult, ErasedSimpleTool,
    RunEvaluationResponse, SimpleToolContext,
};
use sqlx::PgPool;
use uuid::Uuid;

use autopilot_client::{AutopilotSideInfo, OptimizationWorkflowSideInfo};
use autopilot_tools::tools::{RunEvaluationTool, RunEvaluationToolParams};
use common::MockTensorZeroClient;

/// Create a mock RunEvaluationResponse for testing.
fn create_mock_run_evaluation_response() -> RunEvaluationResponse {
    let mut stats = HashMap::new();
    stats.insert(
        "accuracy".to_string(),
        durable_tools::EvaluatorStats {
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
        datapoint_results: None,
    }
}

/// Test that RunEvaluationTool calls the action endpoint with the correct snapshot hash.
#[sqlx::test(migrator = "MIGRATOR")]
async fn test_run_evaluation_tool_with_snapshot_hash(pool: PgPool) {
    // Create mock response
    let mock_response = create_mock_run_evaluation_response();

    // Prepare test data
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let test_snapshot_hash = "12345678901234567890";

    let llm_params = RunEvaluationToolParams {
        evaluation_name: "test_evaluation".to_string(),
        dataset_name: Some("test_dataset".to_string()),
        datapoint_ids: None,
        variant_name: "test_variant".to_string(),
        concurrency: 5,
        max_datapoints: Some(50),
        precision_targets: HashMap::new(),
        inference_cache: CacheEnabledMode::Off,
        include_datapoint_results: false,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: test_snapshot_hash.to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    // Create mock client with expectations for action()
    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_action()
        .withf(move |snapshot_hash, input| {
            let ActionInput::RunEvaluation(params) = input else {
                return false;
            };
            snapshot_hash.to_string() == test_snapshot_hash
                && params.evaluation_name == "test_evaluation"
                && params.dataset_name == Some("test_dataset".to_string())
                && params.datapoint_ids.is_none()
                && params.variant_name == "test_variant"
                && params.concurrency == 5
                && params.max_datapoints == Some(50)
                && params.precision_targets.is_empty()
        })
        .returning(move |_, _| Ok(ActionResponse::RunEvaluation(mock_response.clone())));

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
async fn test_run_evaluation_tool_with_dataset_name(pool: PgPool) {
    // Create mock response
    let mock_response = create_mock_run_evaluation_response();
    let expected_response = mock_response.clone();

    // Prepare test data
    let session_id = Uuid::now_v7();
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
        include_datapoint_results: false,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: "1234567".to_string(),
        optimization: Default::default(),
    };

    // Create mock client with expectations
    let mut mock_client = MockTensorZeroClient::new();
    let expected_tool_call_event_id = tool_call_event_id;
    let expected_session_id = session_id;
    mock_client
        .expect_action()
        .withf(move |_snapshot_hash, input| {
            let ActionInput::RunEvaluation(params) = input else {
                return false;
            };
            params.evaluation_name == "test_evaluation"
                && params.dataset_name == Some("test_dataset".to_string())
                && params.datapoint_ids.is_none()
                && params.variant_name == "test_variant"
                && params.concurrency == 5
                && params.max_datapoints == Some(50)
                && params.precision_targets.is_empty()
                // Verify tags are being passed correctly
                && params.tags.get("tensorzero::autopilot::tool_call_event_id") == Some(&expected_tool_call_event_id.to_string())
                && params.tags.get("tensorzero::autopilot::session_id") == Some(&expected_session_id.to_string())
                && params.tags.get("tensorzero::autopilot::config_snapshot_hash") == Some(&"1234567".to_string())
                && params.tags.get("tensorzero::autopilot") == Some(&"true".to_string())
        })
        .returning(move |_, _| Ok(ActionResponse::RunEvaluation(mock_response.clone())));

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
    let session_id = Uuid::now_v7();
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
        include_datapoint_results: false,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: "1234567".to_string(),
        optimization: Default::default(),
    };

    // Create mock client with expectations
    let mut mock_client = MockTensorZeroClient::new();
    let expected_tool_call_event_id = tool_call_event_id;
    let expected_session_id = session_id;
    mock_client
        .expect_action()
        .withf(move |_snapshot_hash, input| {
            let ActionInput::RunEvaluation(params) = input else {
                return false;
            };
            params.evaluation_name == "test_evaluation"
                && params.dataset_name.is_none()
                && params.datapoint_ids == Some(expected_datapoint_ids.clone())
                && params.variant_name == "test_variant"
                && params.concurrency == 10
                && params.max_datapoints.is_none()
                // Verify tags are being passed correctly
                && params.tags.get("tensorzero::autopilot::tool_call_event_id") == Some(&expected_tool_call_event_id.to_string())
                && params.tags.get("tensorzero::autopilot::session_id") == Some(&expected_session_id.to_string())
                && params.tags.get("tensorzero::autopilot::config_snapshot_hash") == Some(&"1234567".to_string())
                && params.tags.get("tensorzero::autopilot") == Some(&"true".to_string())
        })
        .returning(move |_, _| Ok(ActionResponse::RunEvaluation(mock_response.clone())));

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
async fn test_run_evaluation_tool_with_precision_targets_and_cache(pool: PgPool) {
    // Create mock response
    let mock_response = create_mock_run_evaluation_response();

    // Prepare test data
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    // Set up non-default precision_targets and inference_cache
    let mut precision_targets = HashMap::new();
    precision_targets.insert("accuracy".to_string(), 0.05);
    precision_targets.insert("f1_score".to_string(), 0.03);
    let expected_precision_targets = precision_targets.clone();

    let llm_params = RunEvaluationToolParams {
        evaluation_name: "test_evaluation".to_string(),
        dataset_name: Some("test_dataset".to_string()),
        datapoint_ids: None,
        variant_name: "test_variant".to_string(),
        concurrency: 10,
        max_datapoints: Some(200),
        precision_targets,
        inference_cache: CacheEnabledMode::ReadOnly,
        include_datapoint_results: false,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: "1234567".to_string(),
        optimization: Default::default(),
    };

    // Create mock client with expectations that verify precision_targets and inference_cache
    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_action()
        .withf(move |_snapshot_hash, input| {
            let ActionInput::RunEvaluation(params) = input else {
                return false;
            };
            params.evaluation_name == "test_evaluation"
                && params.dataset_name == Some("test_dataset".to_string())
                && params.variant_name == "test_variant"
                && params.concurrency == 10
                && params.max_datapoints == Some(200)
                // Verify precision_targets are passed through correctly
                && params.precision_targets == expected_precision_targets
                // Verify inference_cache is passed through correctly
                && params.inference_cache == CacheEnabledMode::ReadOnly
        })
        .returning(move |_, _| Ok(ActionResponse::RunEvaluation(mock_response.clone())));

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
    let session_id = Uuid::now_v7();
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
        include_datapoint_results: false,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: "1234567".to_string(),
        optimization: Default::default(),
    };

    // Create mock client that returns an error
    let mut mock_client = MockTensorZeroClient::new();
    mock_client.expect_action().returning(|_, _| {
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

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_run_evaluation_tool_with_datapoint_results(pool: PgPool) {
    // Create mock response with per-datapoint results
    let datapoint_id_1 = Uuid::now_v7();
    let datapoint_id_2 = Uuid::now_v7();
    let datapoint_id_3 = Uuid::now_v7();

    let mut evaluations_1 = HashMap::new();
    evaluations_1.insert("accuracy".to_string(), Some(0.9));
    evaluations_1.insert("quality".to_string(), Some(0.85));

    // Datapoint 2: inference succeeded, but one evaluator failed (quality)
    // This is the scenario where evaluator_errors is populated
    let mut evaluations_2 = HashMap::new();
    evaluations_2.insert("accuracy".to_string(), Some(0.8));
    // Note: quality evaluator failed, so it's not in evaluations

    let mut evaluator_errors_2 = HashMap::new();
    evaluator_errors_2.insert("quality".to_string(), "Evaluator timeout".to_string());

    // Datapoint 3: inference failed entirely
    // When success=false, evaluator_errors is always empty (per client_ext.rs implementation)

    let datapoint_results = vec![
        DatapointResult {
            datapoint_id: datapoint_id_1,
            success: true,
            evaluations: evaluations_1,
            evaluator_errors: HashMap::new(),
            error: None,
        },
        DatapointResult {
            datapoint_id: datapoint_id_2,
            success: true,
            evaluations: evaluations_2,
            evaluator_errors: evaluator_errors_2,
            error: None,
        },
        DatapointResult {
            datapoint_id: datapoint_id_3,
            success: false,
            evaluations: HashMap::new(),
            evaluator_errors: HashMap::new(), // Always empty when success=false
            error: Some("Inference failed".to_string()),
        },
    ];

    let mut stats = HashMap::new();
    stats.insert(
        "accuracy".to_string(),
        durable_tools::EvaluatorStats {
            mean: 0.85,
            stderr: 0.05,
            count: 2,
        },
    );
    stats.insert(
        "quality".to_string(),
        durable_tools::EvaluatorStats {
            mean: 0.85,
            stderr: 0.0,
            count: 1,
        },
    );

    let mock_response = RunEvaluationResponse {
        evaluation_run_id: Uuid::now_v7(),
        num_datapoints: 3,
        num_successes: 2,
        num_errors: 1,
        stats,
        datapoint_results: Some(datapoint_results),
    };
    let expected_response = mock_response.clone();

    // Prepare test data
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = RunEvaluationToolParams {
        evaluation_name: "test_evaluation".to_string(),
        dataset_name: Some("test_dataset".to_string()),
        datapoint_ids: None,
        variant_name: "test_variant".to_string(),
        concurrency: 10,
        max_datapoints: None,
        precision_targets: HashMap::new(),
        inference_cache: CacheEnabledMode::Off,
        include_datapoint_results: true, // Request per-datapoint results
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: "1234567".to_string(),
        optimization: Default::default(),
    };

    // Create mock client with expectations that verify include_datapoint_results is passed
    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_action()
        .withf(move |_snapshot_hash, input| {
            let ActionInput::RunEvaluation(params) = input else {
                return false;
            };
            params.evaluation_name == "test_evaluation"
                && params.dataset_name == Some("test_dataset".to_string())
                && params.variant_name == "test_variant"
                // Verify include_datapoint_results is passed through correctly
                && params.include_datapoint_results
        })
        .returning(move |_, _| Ok(ActionResponse::RunEvaluation(mock_response.clone())));

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

    assert_eq!(
        response.num_datapoints, expected_response.num_datapoints,
        "num_datapoints should match"
    );
    assert_eq!(
        response.num_successes, expected_response.num_successes,
        "num_successes should match"
    );
    assert_eq!(
        response.num_errors, expected_response.num_errors,
        "num_errors should match"
    );

    // Verify datapoint_results is populated
    let datapoint_results = response
        .datapoint_results
        .expect("datapoint_results should be Some when include_datapoint_results is true");
    assert_eq!(
        datapoint_results.len(),
        3,
        "should have 3 datapoint results"
    );

    // Verify first datapoint (fully successful)
    let dp1 = &datapoint_results[0];
    assert!(dp1.success, "first datapoint should be successful");
    assert_eq!(
        dp1.evaluations.get("accuracy"),
        Some(&Some(0.9)),
        "first datapoint accuracy should be 0.9"
    );
    assert!(
        dp1.evaluator_errors.is_empty(),
        "first datapoint should have no evaluator errors"
    );

    // Verify second datapoint (inference succeeded, but one evaluator failed)
    let dp2 = &datapoint_results[1];
    assert!(
        dp2.success,
        "second datapoint should be successful (inference succeeded)"
    );
    assert_eq!(
        dp2.evaluations.get("accuracy"),
        Some(&Some(0.8)),
        "second datapoint accuracy should be 0.8"
    );
    assert!(
        dp2.evaluator_errors.contains_key("quality"),
        "second datapoint should have evaluator error for quality"
    );

    // Verify third datapoint (inference failed)
    let dp3 = &datapoint_results[2];
    assert!(!dp3.success, "third datapoint should have failed");
    assert!(
        dp3.error.is_some(),
        "third datapoint should have an error message"
    );
    assert!(
        dp3.evaluator_errors.is_empty(),
        "failed datapoint should have empty evaluator_errors (per API semantics)"
    );
}
