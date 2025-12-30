//! Integration tests for LaunchOptimizationWorkflowTool.

mod common;

use std::sync::Arc;

use durable::MIGRATOR;
use durable_tools::{ErasedSimpleTool, SimpleToolContext};
use sqlx::PgPool;
use tensorzero_core::db::inferences::InferenceOutputSource;
use tensorzero_core::optimization::openai_sft::UninitializedOpenAISFTConfig;
use tensorzero_core::optimization::{UninitializedOptimizerConfig, UninitializedOptimizerInfo};
use uuid::Uuid;

use autopilot_tools::AutopilotToolSideInfo;
use autopilot_tools::tools::{
    LaunchOptimizationWorkflowTool, LaunchOptimizationWorkflowToolParams,
};
use common::MockTensorZeroClient;

fn create_test_optimizer_config() -> UninitializedOptimizerInfo {
    UninitializedOptimizerInfo {
        inner: UninitializedOptimizerConfig::OpenAISFT(UninitializedOpenAISFTConfig {
            model: "gpt-4o-mini-2024-07-18".to_string(),
            batch_size: None,
            learning_rate_multiplier: None,
            n_epochs: None,
            seed: None,
            suffix: None,
        }),
    }
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_launch_optimization_workflow_tool_success(pool: PgPool) {
    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let optimizer_config = create_test_optimizer_config();

    let llm_params = LaunchOptimizationWorkflowToolParams {
        function_name: "test_function".to_string(),
        template_variant_name: "test_variant".to_string(),
        query_variant_name: None,
        filters: None,
        output_source: InferenceOutputSource::Inference,
        order_by: None,
        limit: Some(100),
        offset: None,
        val_fraction: Some(0.2),
        optimizer_config,
    };

    let side_info = AutopilotToolSideInfo {
        episode_id,
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    let expected_job_handle = "test_job_handle_base64_encoded".to_string();

    let mut mock_client = MockTensorZeroClient::new();
    let expected_handle_clone = expected_job_handle.clone();
    mock_client
        .expect_launch_optimization_workflow()
        .withf(move |params| {
            params.function_name == "test_function"
                && params.template_variant_name == "test_variant"
                && params.limit == Some(100)
                && params.val_fraction == Some(0.2)
        })
        .returning(move |_| Ok(expected_handle_clone.clone()));

    let tool = LaunchOptimizationWorkflowTool;
    let t0_client: Arc<dyn durable_tools::TensorZeroClient> = Arc::new(mock_client);
    let ctx = SimpleToolContext::new(&pool, &t0_client);

    let result = tool
        .execute_erased(
            serde_json::to_value(&llm_params).expect("Failed to serialize llm_params"),
            serde_json::to_value(&side_info).expect("Failed to serialize side_info"),
            ctx,
            "test-idempotency-key",
        )
        .await
        .expect("Tool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
    assert_eq!(
        result["job_handle"].as_str(),
        Some(expected_job_handle.as_str()),
        "Job handle should match"
    );
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_launch_optimization_workflow_tool_with_query_variant(pool: PgPool) {
    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let optimizer_config = create_test_optimizer_config();

    let llm_params = LaunchOptimizationWorkflowToolParams {
        function_name: "test_function".to_string(),
        template_variant_name: "test_variant".to_string(),
        query_variant_name: Some("specific_variant".to_string()),
        filters: None,
        output_source: InferenceOutputSource::Demonstration,
        order_by: None,
        limit: Some(50),
        offset: Some(10),
        val_fraction: None,
        optimizer_config,
    };

    let side_info = AutopilotToolSideInfo {
        episode_id,
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    let expected_job_handle = "another_job_handle".to_string();

    let mut mock_client = MockTensorZeroClient::new();
    let expected_handle_clone = expected_job_handle.clone();
    mock_client
        .expect_launch_optimization_workflow()
        .withf(move |params| {
            params.function_name == "test_function"
                && params.query_variant_name == Some("specific_variant".to_string())
                && params.limit == Some(50)
                && params.offset == Some(10)
                && params.val_fraction.is_none()
        })
        .returning(move |_| Ok(expected_handle_clone.clone()));

    let tool = LaunchOptimizationWorkflowTool;
    let t0_client: Arc<dyn durable_tools::TensorZeroClient> = Arc::new(mock_client);
    let ctx = SimpleToolContext::new(&pool, &t0_client);

    let result = tool
        .execute_erased(
            serde_json::to_value(&llm_params).expect("Failed to serialize llm_params"),
            serde_json::to_value(&side_info).expect("Failed to serialize side_info"),
            ctx,
            "test-idempotency-key",
        )
        .await
        .expect("Tool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
    assert_eq!(
        result["job_handle"].as_str(),
        Some(expected_job_handle.as_str()),
        "Job handle should match"
    );
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_launch_optimization_workflow_tool_error(pool: PgPool) {
    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let optimizer_config = create_test_optimizer_config();

    let llm_params = LaunchOptimizationWorkflowToolParams {
        function_name: "test_function".to_string(),
        template_variant_name: "test_variant".to_string(),
        query_variant_name: None,
        filters: None,
        output_source: InferenceOutputSource::Inference,
        order_by: None,
        limit: None,
        offset: None,
        val_fraction: None,
        optimizer_config,
    };

    let side_info = AutopilotToolSideInfo {
        episode_id,
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_launch_optimization_workflow()
        .returning(|_| Err(durable_tools::TensorZeroClientError::AutopilotUnavailable));

    let tool = LaunchOptimizationWorkflowTool;
    let t0_client: Arc<dyn durable_tools::TensorZeroClient> = Arc::new(mock_client);
    let ctx = SimpleToolContext::new(&pool, &t0_client);

    let result = tool
        .execute_erased(
            serde_json::to_value(&llm_params).expect("Failed to serialize llm_params"),
            serde_json::to_value(&side_info).expect("Failed to serialize side_info"),
            ctx,
            "test-idempotency-key",
        )
        .await;

    assert!(result.is_err(), "Tool execution should fail");
}
