//! Integration tests for LaunchOptimizationWorkflowTool (TaskTool).
//!
//! These tests use real Postgres via sqlx::test and MockTensorZeroClient to avoid
//! making actual API calls to external services.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use autopilot_client::OptimizationWorkflowSideInfo;
use autopilot_tools::tools::{
    LaunchOptimizationWorkflowLlmParams, LaunchOptimizationWorkflowTool, LlmOptimizerConfig,
    LlmOptimizerType, LlmOutputSource,
};
use common::MockTensorZeroClient;
use durable::MIGRATOR;
use durable_tools::{
    SpawnOptions, TensorZeroClient, TensorZeroClientError, ToolExecutor, WorkerOptions,
};
use sqlx::PgPool;
use tensorzero_core::optimization::dicl::DiclOptimizationJobHandle;
use tensorzero_core::optimization::{OptimizationJobHandle, OptimizationJobInfo, OptimizerOutput};
use uuid::Uuid;

// ===== Test Helpers =====

fn create_test_job_handle() -> OptimizationJobHandle {
    OptimizationJobHandle::Dicl(DiclOptimizationJobHandle {
        embedding_model: "test-embedding".to_string(),
        k: 5,
        model: "test-model".to_string(),
    })
}

fn create_test_params() -> LaunchOptimizationWorkflowLlmParams {
    LaunchOptimizationWorkflowLlmParams {
        function_name: "test_function".to_string(),
        template_variant_name: "test_variant".to_string(),
        query_variant_name: None,
        output_source: LlmOutputSource::InferenceOutput,
        limit: Some(100),
        offset: None,
        val_fraction: None,
        optimizer_config: LlmOptimizerConfig {
            optimizer_type: LlmOptimizerType::Dicl,
            embedding_model: Some("text-embedding-3-small".to_string()),
            k: Some(5),
            model: None,
            n_epochs: None,
            learning_rate_multiplier: None,
            suffix: None,
            gepa_function_name: None,
            evaluation_name: None,
            analysis_model: None,
            mutation_model: None,
        },
    }
}

fn create_test_side_info() -> OptimizationWorkflowSideInfo {
    OptimizationWorkflowSideInfo {
        poll_interval_secs: 1, // Short for tests
        max_wait_secs: 30,     // Reasonable timeout for tests
    }
}

fn create_completed_job_info() -> OptimizationJobInfo {
    OptimizationJobInfo::Completed {
        output: OptimizerOutput::Variants(std::collections::HashMap::new()),
    }
}

fn create_pending_job_info() -> OptimizationJobInfo {
    OptimizationJobInfo::Pending {
        message: "Job is running".to_string(),
        estimated_finish: None,
        trained_tokens: None,
        error: None,
    }
}

fn create_failed_job_info() -> OptimizationJobInfo {
    OptimizationJobInfo::Failed {
        message: "Job failed".to_string(),
        error: None,
    }
}

// ===== Tests =====

/// Test that the tool completes successfully when poll returns Completed immediately.
#[sqlx::test(migrator = "MIGRATOR")]
async fn test_launch_optimization_workflow_tool_immediate_completion(
    pool: PgPool,
) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());

    let mut mock_client = MockTensorZeroClient::new();

    // Expect launch to be called once
    mock_client
        .expect_launch_optimization_workflow()
        .times(1)
        .returning(|_| Ok(create_test_job_handle()));

    // Expect poll to be called once and return Completed
    mock_client
        .expect_poll_optimization()
        .times(1)
        .returning(|_| Ok(create_completed_job_info()));

    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client);

    let executor = ToolExecutor::builder()
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
        .register_task_tool::<LaunchOptimizationWorkflowTool>()
        .await
        .unwrap();

    let episode_id = Uuid::now_v7();
    let _spawn_result = executor
        .spawn_tool::<LaunchOptimizationWorkflowTool>(
            create_test_params(),
            create_test_side_info(),
            episode_id,
            SpawnOptions::default(),
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

    // Wait for task to complete
    tokio::time::sleep(Duration::from_secs(3)).await;
    worker.shutdown().await;

    // If we got here without panic, the mock expectations were satisfied
    // (mockall panics if times() expectations are not met)
    Ok(())
}

/// Test that the tool polls multiple times before completion.
#[sqlx::test(migrator = "MIGRATOR")]
async fn test_launch_optimization_workflow_tool_multiple_polls(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());

    // Use atomic counter to track poll calls
    let poll_count = Arc::new(AtomicU32::new(0));
    let poll_count_clone = poll_count.clone();

    let mut mock_client = MockTensorZeroClient::new();

    mock_client
        .expect_launch_optimization_workflow()
        .times(1)
        .returning(|_| Ok(create_test_job_handle()));

    // Return Pending twice, then Completed
    mock_client
        .expect_poll_optimization()
        .times(3)
        .returning(move |_| {
            let count = poll_count_clone.fetch_add(1, Ordering::SeqCst);
            if count < 2 {
                Ok(create_pending_job_info())
            } else {
                Ok(create_completed_job_info())
            }
        });

    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client);

    let executor = ToolExecutor::builder()
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
        .register_task_tool::<LaunchOptimizationWorkflowTool>()
        .await
        .unwrap();

    let episode_id = Uuid::now_v7();
    let _spawn_result = executor
        .spawn_tool::<LaunchOptimizationWorkflowTool>(
            create_test_params(),
            create_test_side_info(),
            episode_id,
            SpawnOptions::default(),
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

    // Wait for task to complete (needs time for polls + sleeps)
    tokio::time::sleep(Duration::from_secs(5)).await;
    worker.shutdown().await;

    // Verify poll was called 3 times
    assert_eq!(poll_count.load(Ordering::SeqCst), 3);

    Ok(())
}

/// Test that the tool returns Failed status when poll returns Failed.
#[sqlx::test(migrator = "MIGRATOR")]
async fn test_launch_optimization_workflow_tool_failed(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());

    let mut mock_client = MockTensorZeroClient::new();

    mock_client
        .expect_launch_optimization_workflow()
        .times(1)
        .returning(|_| Ok(create_test_job_handle()));

    mock_client
        .expect_poll_optimization()
        .times(1)
        .returning(|_| Ok(create_failed_job_info()));

    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client);

    let executor = ToolExecutor::builder()
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
        .register_task_tool::<LaunchOptimizationWorkflowTool>()
        .await
        .unwrap();

    let episode_id = Uuid::now_v7();
    let _spawn_result = executor
        .spawn_tool::<LaunchOptimizationWorkflowTool>(
            create_test_params(),
            create_test_side_info(),
            episode_id,
            SpawnOptions::default(),
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

    // Mock expectations were satisfied
    Ok(())
}

/// Test that launch errors are handled.
#[sqlx::test(migrator = "MIGRATOR")]
async fn test_launch_optimization_workflow_tool_launch_error(pool: PgPool) -> sqlx::Result<()> {
    let queue_name = format!("test_queue_{}", Uuid::now_v7());

    let mut mock_client = MockTensorZeroClient::new();

    // The durable framework will retry failed operations, so allow any number of calls
    mock_client
        .expect_launch_optimization_workflow()
        .returning(|_| Err(TensorZeroClientError::AutopilotUnavailable));

    // poll_optimization should not be called if launch keeps failing
    mock_client.expect_poll_optimization().times(0);

    let t0_client: Arc<dyn TensorZeroClient> = Arc::new(mock_client);

    let executor = ToolExecutor::builder()
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
        .register_task_tool::<LaunchOptimizationWorkflowTool>()
        .await
        .unwrap();

    let episode_id = Uuid::now_v7();
    let _spawn_result = executor
        .spawn_tool::<LaunchOptimizationWorkflowTool>(
            create_test_params(),
            create_test_side_info(),
            episode_id,
            SpawnOptions::default(),
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

    // Mock expectations were satisfied (poll not called)
    Ok(())
}
