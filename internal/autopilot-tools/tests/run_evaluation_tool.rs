//! Integration tests for RunEvaluationTool (TaskTool).
//!
//! These tests use real Postgres via sqlx::test and MockTensorZeroClient to avoid
//! making actual API calls to external services.

mod common;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use autopilot_client::{AutopilotSideInfo, OptimizationWorkflowSideInfo};
use autopilot_tools::tools::{RunEvaluationTool, RunEvaluationToolParams};
use common::MockTensorZeroClient;
use durable::MIGRATOR;
use durable_tools::{
    ActionInput, ActionResponse, CacheEnabledMode, RunEvaluationResponse, TensorZeroClient,
    ToolExecutor, WorkerOptions,
};
use sqlx::PgPool;
use uuid::Uuid;

fn create_mock_run_evaluation_response(num_datapoints: usize) -> RunEvaluationResponse {
    let mut stats = HashMap::new();
    stats.insert(
        "accuracy".to_string(),
        durable_tools::EvaluatorStats {
            mean: 0.85,
            stderr: 0.02,
            count: num_datapoints,
        },
    );
    RunEvaluationResponse {
        evaluation_run_id: Uuid::now_v7(),
        num_datapoints,
        num_successes: num_datapoints,
        num_errors: 0,
        stats,
        datapoint_results: None,
    }
}

fn create_test_side_info() -> AutopilotSideInfo {
    AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "12345678901234567890".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    }
}

/// Test single-batch evaluation with datapoint_ids (≤ batch_size).
#[sqlx::test(migrator = "MIGRATOR")]
async fn test_run_evaluation_tool_single_batch_with_datapoint_ids(
    pool: PgPool,
) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());

    let datapoint_ids = vec![Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7()];
    let expected_ids = datapoint_ids.clone();
    let mock_response = create_mock_run_evaluation_response(3);

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_action()
        .times(1)
        .withf(move |_snapshot_hash, input| {
            let ActionInput::RunEvaluation(params) = input else {
                return false;
            };
            params.evaluation_name == "test_eval"
                && params.dataset_name.is_none()
                && params.datapoint_ids == Some(expected_ids.clone())
                && params.variant_name == "test_variant"
                && params.precision_targets.is_empty()
        })
        .returning(move |_, _| Ok(ActionResponse::RunEvaluation(mock_response.clone())));

    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client);

    let executor = ToolExecutor::builder(())
        .pool(pool)
        .queue_name(&queue_name)
        .t0_client(t0_client)
        .build()
        .await
        .expect("Failed to build executor");

    executor
        .durable()
        .create_queue(None)
        .await
        .expect("Failed to create queue");

    executor
        .register_task_tool_instance(RunEvaluationTool)
        .await
        .unwrap();

    let llm_params = RunEvaluationToolParams {
        evaluation_name: "test_eval".to_string(),
        dataset_name: None,
        datapoint_ids: Some(datapoint_ids),
        variant_name: "test_variant".to_string(),
        concurrency: 10,
        max_datapoints: None,
        precision_targets: HashMap::new(),
        inference_cache: CacheEnabledMode::Off,
        include_datapoint_results: false,
        batch_size: 25,
    };

    let episode_id = Uuid::now_v7();
    let _spawn_result = executor
        .spawn_tool_by_name(
            "run_evaluation",
            serde_json::to_value(&llm_params).unwrap(),
            serde_json::to_value(create_test_side_info()).unwrap(),
            episode_id,
        )
        .await
        .expect("Failed to spawn task");

    let worker = executor
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(50),
            claim_timeout: Duration::from_secs(30),
            ..Default::default()
        })
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_secs(3)).await;
    worker.shutdown().await;

    Ok(())
}

/// Test multi-batch evaluation with datapoint_ids split across batches.
#[sqlx::test(migrator = "MIGRATOR")]
async fn test_run_evaluation_tool_multi_batch_with_datapoint_ids(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());

    // 5 datapoints with batch_size=2 → 3 batches (2, 2, 1)
    let datapoint_ids: Vec<Uuid> = (0..5).map(|_| Uuid::now_v7()).collect();

    let mut mock_client = MockTensorZeroClient::new();
    // Expect 3 action calls (one per batch)
    mock_client.expect_action().times(3).returning(|_, input| {
        let ActionInput::RunEvaluation(params) = input else {
            panic!("Expected RunEvaluation action");
        };
        let n = params.datapoint_ids.as_ref().unwrap().len();
        Ok(ActionResponse::RunEvaluation(
            create_mock_run_evaluation_response(n),
        ))
    });

    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client);

    let executor = ToolExecutor::builder(())
        .pool(pool)
        .queue_name(&queue_name)
        .t0_client(t0_client)
        .build()
        .await
        .expect("Failed to build executor");

    executor
        .durable()
        .create_queue(None)
        .await
        .expect("Failed to create queue");

    executor
        .register_task_tool_instance(RunEvaluationTool)
        .await
        .unwrap();

    let llm_params = RunEvaluationToolParams {
        evaluation_name: "test_eval".to_string(),
        dataset_name: None,
        datapoint_ids: Some(datapoint_ids),
        variant_name: "test_variant".to_string(),
        concurrency: 10,
        max_datapoints: None,
        precision_targets: HashMap::new(),
        inference_cache: CacheEnabledMode::Off,
        include_datapoint_results: false,
        batch_size: 2,
    };

    let episode_id = Uuid::now_v7();
    let _spawn_result = executor
        .spawn_tool_by_name(
            "run_evaluation",
            serde_json::to_value(&llm_params).unwrap(),
            serde_json::to_value(create_test_side_info()).unwrap(),
            episode_id,
        )
        .await
        .expect("Failed to spawn task");

    let worker = executor
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(50),
            claim_timeout: Duration::from_secs(30),
            ..Default::default()
        })
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_secs(5)).await;
    worker.shutdown().await;

    Ok(())
}

/// Test evaluation with dataset_name (requires list_datapoints call).
#[sqlx::test(migrator = "MIGRATOR")]
async fn test_run_evaluation_tool_with_dataset_name(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());

    let dp_id1 = Uuid::now_v7();
    let dp_id2 = Uuid::now_v7();

    let mut mock_client = MockTensorZeroClient::new();

    // Expect list_datapoints to be called for fetching IDs
    mock_client
        .expect_list_datapoints()
        .times(1)
        .withf(|dataset_name, _request| dataset_name == "test_dataset")
        .returning(move |_, _| {
            Ok(common::create_mock_get_datapoints_response(vec![
                common::create_mock_chat_datapoint(dp_id1, "test_dataset", "test_fn"),
                common::create_mock_chat_datapoint(dp_id2, "test_dataset", "test_fn"),
            ]))
        });

    // Expect 1 action call (both IDs fit in one batch with default batch_size)
    mock_client.expect_action().times(1).returning(|_, input| {
        let ActionInput::RunEvaluation(params) = input else {
            panic!("Expected RunEvaluation action");
        };
        let n = params.datapoint_ids.as_ref().unwrap().len();
        Ok(ActionResponse::RunEvaluation(
            create_mock_run_evaluation_response(n),
        ))
    });

    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client);

    let executor = ToolExecutor::builder(())
        .pool(pool)
        .queue_name(&queue_name)
        .t0_client(t0_client)
        .build()
        .await
        .expect("Failed to build executor");

    executor
        .durable()
        .create_queue(None)
        .await
        .expect("Failed to create queue");

    executor
        .register_task_tool_instance(RunEvaluationTool)
        .await
        .unwrap();

    let llm_params = RunEvaluationToolParams {
        evaluation_name: "test_eval".to_string(),
        dataset_name: Some("test_dataset".to_string()),
        datapoint_ids: None,
        variant_name: "test_variant".to_string(),
        concurrency: 10,
        max_datapoints: None,
        precision_targets: HashMap::new(),
        inference_cache: CacheEnabledMode::Off,
        include_datapoint_results: false,
        batch_size: 25,
    };

    let episode_id = Uuid::now_v7();
    let _spawn_result = executor
        .spawn_tool_by_name(
            "run_evaluation",
            serde_json::to_value(&llm_params).unwrap(),
            serde_json::to_value(create_test_side_info()).unwrap(),
            episode_id,
        )
        .await
        .expect("Failed to spawn task");

    let worker = executor
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(50),
            claim_timeout: Duration::from_secs(30),
            ..Default::default()
        })
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_secs(3)).await;
    worker.shutdown().await;

    Ok(())
}

/// Test that errors from the action endpoint propagate correctly.
#[sqlx::test(migrator = "MIGRATOR")]
async fn test_run_evaluation_tool_error_handling(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());

    let mut mock_client = MockTensorZeroClient::new();
    // The durable framework will retry failed operations, so allow any number of calls
    mock_client.expect_action().returning(|_, _| {
        Err(durable_tools::TensorZeroClientError::Evaluation(
            "Evaluation 'nonexistent_evaluation' not found in config".to_string(),
        ))
    });

    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client);

    let executor = ToolExecutor::builder(())
        .pool(pool)
        .queue_name(&queue_name)
        .t0_client(t0_client)
        .build()
        .await
        .expect("Failed to build executor");

    executor
        .durable()
        .create_queue(None)
        .await
        .expect("Failed to create queue");

    executor
        .register_task_tool_instance(RunEvaluationTool)
        .await
        .unwrap();

    let llm_params = RunEvaluationToolParams {
        evaluation_name: "nonexistent_evaluation".to_string(),
        dataset_name: None,
        datapoint_ids: Some(vec![Uuid::now_v7()]),
        variant_name: "test_variant".to_string(),
        concurrency: 10,
        max_datapoints: None,
        precision_targets: HashMap::new(),
        inference_cache: CacheEnabledMode::Off,
        include_datapoint_results: false,
        batch_size: 25,
    };

    let episode_id = Uuid::now_v7();
    let _spawn_result = executor
        .spawn_tool_by_name(
            "run_evaluation",
            serde_json::to_value(&llm_params).unwrap(),
            serde_json::to_value(create_test_side_info()).unwrap(),
            episode_id,
        )
        .await
        .expect("Failed to spawn task");

    let worker = executor
        .start_worker(WorkerOptions {
            poll_interval: Duration::from_millis(50),
            claim_timeout: Duration::from_secs(30),
            ..Default::default()
        })
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_secs(3)).await;
    worker.shutdown().await;

    // Mock expectations were satisfied (action was called, but kept failing)
    Ok(())
}
