//! Integration tests for RunTopKEvaluationTool.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use durable::MIGRATOR;
use durable_tools::{
    CacheEnabledMode, ErasedSimpleTool, GlobalStoppingReason, MeanBettingConfidenceSequence,
    RunTopKEvaluationResponse, ScoringFunctionType, SimpleToolContext, TopKTaskOutput,
    VariantStatus, WealthProcessGridPoints, WealthProcesses,
};
use sqlx::PgPool;
use uuid::Uuid;

use autopilot_client::AutopilotSideInfo;
use autopilot_tools::tools::{RunTopKEvaluationTool, RunTopKEvaluationToolParams};
use common::MockTensorZeroClient;

/// Create a mock MeanBettingConfidenceSequence for testing.
fn create_mock_cs(name: &str, mean: f64) -> MeanBettingConfidenceSequence {
    MeanBettingConfidenceSequence {
        name: name.to_string(),
        mean_regularized: mean,
        variance_regularized: 0.1,
        count: 100,
        mean_est: mean,
        cs_lower: mean - 0.1,
        cs_upper: mean + 0.1,
        alpha: 0.05,
        wealth: WealthProcesses {
            grid: WealthProcessGridPoints::Resolution(101),
            wealth_upper: vec![1.0; 101],
            wealth_lower: vec![1.0; 101],
        },
    }
}

/// Create a mock TopKTaskOutput for testing.
fn create_mock_topk_output() -> TopKTaskOutput {
    let mut variant_status = HashMap::new();
    variant_status.insert("variant_a".to_string(), VariantStatus::Include);
    variant_status.insert("variant_b".to_string(), VariantStatus::Exclude);
    variant_status.insert("variant_c".to_string(), VariantStatus::Exclude);

    let mut variant_performance = HashMap::new();
    variant_performance.insert("variant_a".to_string(), create_mock_cs("variant_a", 0.8));
    variant_performance.insert("variant_b".to_string(), create_mock_cs("variant_b", 0.6));
    variant_performance.insert("variant_c".to_string(), create_mock_cs("variant_c", 0.5));

    let mut variant_failures = HashMap::new();
    variant_failures.insert("variant_a".to_string(), create_mock_cs("variant_a", 0.02));
    variant_failures.insert("variant_b".to_string(), create_mock_cs("variant_b", 0.03));
    variant_failures.insert("variant_c".to_string(), create_mock_cs("variant_c", 0.01));

    TopKTaskOutput {
        evaluation_run_id: Uuid::now_v7(),
        variant_status,
        variant_performance,
        variant_failures,
        evaluator_failures: HashMap::new(),
        stopping_reason: GlobalStoppingReason::TopKFound {
            k: 1,
            top_variants: vec!["variant_a".to_string()],
        },
        num_datapoints_processed: 50,
    }
}

/// Create a mock RunTopKEvaluationResponse for testing.
fn create_mock_run_topk_evaluation_response() -> RunTopKEvaluationResponse {
    RunTopKEvaluationResponse {
        output: create_mock_topk_output(),
    }
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_topk_evaluation_tool_basic(pool: PgPool) {
    // Create mock response
    let mock_response = create_mock_run_topk_evaluation_response();
    let expected_num_datapoints = mock_response.output.num_datapoints_processed;

    // Prepare test data
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = RunTopKEvaluationToolParams {
        evaluation_name: "test_evaluation".to_string(),
        dataset_name: "test_dataset".to_string(),
        variant_names: vec![
            "variant_a".to_string(),
            "variant_b".to_string(),
            "variant_c".to_string(),
        ],
        k_min: 1,
        k_max: 2,
        epsilon: None,
        max_datapoints: Some(100),
        batch_size: Some(20),
        variant_failure_threshold: 0.05,
        evaluator_failure_threshold: 0.05,
        concurrency: 5,
        inference_cache: CacheEnabledMode::Off,
        scoring_function: ScoringFunctionType::AverageEvaluatorScore,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: Default::default(),
    };

    // Create mock client with expectations
    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_run_topk_evaluation()
        .withf(move |params| {
            params.evaluation_name == "test_evaluation"
                && params.dataset_name == "test_dataset"
                && params.variant_names.len() == 3
                && params.k_min == 1
                && params.k_max == 2
                && params.concurrency == 5
                && params.max_datapoints == Some(100)
        })
        .returning(move |_| Ok(mock_response.clone()));

    // Create the tool and context
    let tool = RunTopKEvaluationTool;
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
        .expect("RunTopKEvaluationTool execution should succeed");

    // Verify the response
    let response: RunTopKEvaluationResponse =
        serde_json::from_value(result).expect("Failed to deserialize response");
    assert_eq!(
        response.output.num_datapoints_processed,
        expected_num_datapoints
    );
    assert!(matches!(
        response.output.stopping_reason,
        GlobalStoppingReason::TopKFound { k: 1, .. }
    ));
    assert_eq!(
        response.output.variant_status.get("variant_a"),
        Some(&VariantStatus::Include)
    );
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_topk_evaluation_tool_error_handling(pool: PgPool) {
    // Prepare test data
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = RunTopKEvaluationToolParams {
        evaluation_name: "nonexistent_evaluation".to_string(),
        dataset_name: "test_dataset".to_string(),
        variant_names: vec!["variant_a".to_string()],
        k_min: 1,
        k_max: 1,
        epsilon: None,
        max_datapoints: None,
        batch_size: None,
        variant_failure_threshold: 0.05,
        evaluator_failure_threshold: 0.05,
        concurrency: 5,
        inference_cache: CacheEnabledMode::Off,
        scoring_function: ScoringFunctionType::AverageEvaluatorScore,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: Default::default(),
    };

    // Create mock client that returns an error
    let mut mock_client = MockTensorZeroClient::new();
    mock_client.expect_run_topk_evaluation().returning(|_| {
        Err(durable_tools::TensorZeroClientError::Evaluation(
            "Evaluation 'nonexistent_evaluation' not found in config".to_string(),
        ))
    });

    // Create the tool and context
    let tool = RunTopKEvaluationTool;
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
async fn test_topk_evaluation_tool_dataset_exhausted(pool: PgPool) {
    // Create mock response with DatasetExhausted stopping reason
    let mut mock_output = create_mock_topk_output();
    mock_output.stopping_reason = GlobalStoppingReason::DatasetExhausted;

    let mock_response = RunTopKEvaluationResponse {
        output: mock_output,
    };

    // Prepare test data
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = RunTopKEvaluationToolParams {
        evaluation_name: "test_evaluation".to_string(),
        dataset_name: "small_dataset".to_string(),
        variant_names: vec!["variant_a".to_string(), "variant_b".to_string()],
        k_min: 1,
        k_max: 1,
        epsilon: Some(0.01), // Very tight epsilon
        max_datapoints: None,
        batch_size: None,
        variant_failure_threshold: 0.05,
        evaluator_failure_threshold: 0.05,
        concurrency: 10,
        inference_cache: CacheEnabledMode::On,
        scoring_function: ScoringFunctionType::AverageEvaluatorScore,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: Default::default(),
    };

    // Create mock client
    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_run_topk_evaluation()
        .returning(move |_| Ok(mock_response.clone()));

    // Create the tool and context
    let tool = RunTopKEvaluationTool;
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
        .expect("RunTopKEvaluationTool execution should succeed");

    // Verify the response shows dataset exhausted
    let response: RunTopKEvaluationResponse =
        serde_json::from_value(result).expect("Failed to deserialize response");
    assert!(matches!(
        response.output.stopping_reason,
        GlobalStoppingReason::DatasetExhausted
    ));
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_topk_evaluation_tool_evaluators_failed(pool: PgPool) {
    // Create mock response with EvaluatorsFailed stopping reason
    let mut mock_output = create_mock_topk_output();
    mock_output.stopping_reason = GlobalStoppingReason::EvaluatorsFailed {
        evaluator_names: vec!["llm_judge".to_string(), "exact_match".to_string()],
    };
    // Add evaluator failure tracking
    mock_output
        .evaluator_failures
        .insert("llm_judge".to_string(), create_mock_cs("llm_judge", 0.15));
    mock_output.evaluator_failures.insert(
        "exact_match".to_string(),
        create_mock_cs("exact_match", 0.12),
    );

    let mock_response = RunTopKEvaluationResponse {
        output: mock_output,
    };

    // Prepare test data
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = RunTopKEvaluationToolParams {
        evaluation_name: "test_evaluation".to_string(),
        dataset_name: "test_dataset".to_string(),
        variant_names: vec!["variant_a".to_string(), "variant_b".to_string()],
        k_min: 1,
        k_max: 1,
        epsilon: None,
        max_datapoints: None,
        batch_size: None,
        variant_failure_threshold: 0.05,
        evaluator_failure_threshold: 0.10, // 10% threshold
        concurrency: 5,
        inference_cache: CacheEnabledMode::Off,
        scoring_function: ScoringFunctionType::AverageEvaluatorScore,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: Default::default(),
    };

    // Create mock client
    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_run_topk_evaluation()
        .returning(move |_| Ok(mock_response.clone()));

    // Create the tool and context
    let tool = RunTopKEvaluationTool;
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
        .expect("RunTopKEvaluationTool execution should succeed");

    // Verify the response shows evaluators failed
    let response: RunTopKEvaluationResponse =
        serde_json::from_value(result).expect("Failed to deserialize response");
    match &response.output.stopping_reason {
        GlobalStoppingReason::EvaluatorsFailed { evaluator_names } => {
            assert_eq!(evaluator_names.len(), 2, "Should have 2 failed evaluators");
            assert!(
                evaluator_names.contains(&"llm_judge".to_string()),
                "Should contain llm_judge"
            );
            assert!(
                evaluator_names.contains(&"exact_match".to_string()),
                "Should contain exact_match"
            );
        }
        other => panic!("Expected EvaluatorsFailed stopping reason, got {other:?}"),
    }
    // Verify evaluator failure tracking is present
    assert!(
        response.output.evaluator_failures.contains_key("llm_judge"),
        "Should track llm_judge failures"
    );
    assert!(
        response
            .output
            .evaluator_failures
            .contains_key("exact_match"),
        "Should track exact_match failures"
    );
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_topk_evaluation_tool_too_many_variants_failed(pool: PgPool) {
    // Create mock response with TooManyVariantsFailed stopping reason
    let mut mock_output = create_mock_topk_output();
    mock_output.stopping_reason = GlobalStoppingReason::TooManyVariantsFailed { num_failed: 2 };
    // Update variant statuses to show failures
    mock_output
        .variant_status
        .insert("variant_a".to_string(), VariantStatus::Failed);
    mock_output
        .variant_status
        .insert("variant_b".to_string(), VariantStatus::Failed);
    mock_output
        .variant_status
        .insert("variant_c".to_string(), VariantStatus::Active);
    // Update variant failures to show high failure rates
    mock_output
        .variant_failures
        .insert("variant_a".to_string(), create_mock_cs("variant_a", 0.15));
    mock_output
        .variant_failures
        .insert("variant_b".to_string(), create_mock_cs("variant_b", 0.20));

    let mock_response = RunTopKEvaluationResponse {
        output: mock_output,
    };

    // Prepare test data
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = RunTopKEvaluationToolParams {
        evaluation_name: "test_evaluation".to_string(),
        dataset_name: "test_dataset".to_string(),
        variant_names: vec![
            "variant_a".to_string(),
            "variant_b".to_string(),
            "variant_c".to_string(),
        ],
        k_min: 2, // Need at least 2 variants, but 2 failed
        k_max: 2,
        epsilon: None,
        max_datapoints: None,
        batch_size: None,
        variant_failure_threshold: 0.10, // 10% threshold
        evaluator_failure_threshold: 0.05,
        concurrency: 5,
        inference_cache: CacheEnabledMode::Off,
        scoring_function: ScoringFunctionType::AverageEvaluatorScore,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: Default::default(),
    };

    // Create mock client
    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_run_topk_evaluation()
        .returning(move |_| Ok(mock_response.clone()));

    // Create the tool and context
    let tool = RunTopKEvaluationTool;
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
        .expect("RunTopKEvaluationTool execution should succeed");

    // Verify the response shows too many variants failed
    let response: RunTopKEvaluationResponse =
        serde_json::from_value(result).expect("Failed to deserialize response");
    match &response.output.stopping_reason {
        GlobalStoppingReason::TooManyVariantsFailed { num_failed } => {
            assert_eq!(*num_failed, 2, "Should have 2 failed variants");
        }
        other => panic!("Expected TooManyVariantsFailed stopping reason, got {other:?}"),
    }
    // Verify variant statuses
    assert_eq!(
        response.output.variant_status.get("variant_a"),
        Some(&VariantStatus::Failed),
        "variant_a should be Failed"
    );
    assert_eq!(
        response.output.variant_status.get("variant_b"),
        Some(&VariantStatus::Failed),
        "variant_b should be Failed"
    );
    assert_eq!(
        response.output.variant_status.get("variant_c"),
        Some(&VariantStatus::Active),
        "variant_c should still be Active"
    );
}
