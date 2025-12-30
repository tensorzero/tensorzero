//! Integration tests for GetLatestFeedbackByMetricTool.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use durable::MIGRATOR;
use durable_tools::{ErasedSimpleTool, SimpleToolContext, TensorZeroClientError};
use sqlx::PgPool;
use tensorzero_core::endpoints::feedback::internal::LatestFeedbackIdByMetricResponse;
use uuid::Uuid;

use autopilot_tools::AutopilotToolSideInfo;
use autopilot_tools::tools::{GetLatestFeedbackByMetricTool, GetLatestFeedbackByMetricToolParams};
use common::MockTensorZeroClient;

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_latest_feedback_by_metric_tool_success(pool: PgPool) {
    let target_id = Uuid::now_v7();
    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let mut expected_map = HashMap::new();
    expected_map.insert("accuracy".to_string(), Uuid::now_v7().to_string());
    expected_map.insert("quality".to_string(), Uuid::now_v7().to_string());

    let llm_params = GetLatestFeedbackByMetricToolParams { target_id };

    let side_info = AutopilotToolSideInfo {
        episode_id,
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    let mut mock_client = MockTensorZeroClient::new();
    let expected_map_clone = expected_map.clone();
    mock_client
        .expect_get_latest_feedback_id_by_metric()
        .withf(move |id| *id == target_id)
        .returning(move |_| {
            Ok(LatestFeedbackIdByMetricResponse {
                feedback_id_by_metric: expected_map_clone.clone(),
            })
        });

    let tool = GetLatestFeedbackByMetricTool;
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
    let feedback_map = result["feedback_id_by_metric"]
        .as_object()
        .expect("Should have feedback_id_by_metric");
    assert_eq!(feedback_map.len(), 2, "Should have 2 metrics");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_latest_feedback_by_metric_tool_empty_result(pool: PgPool) {
    let target_id = Uuid::now_v7();
    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = GetLatestFeedbackByMetricToolParams { target_id };

    let side_info = AutopilotToolSideInfo {
        episode_id,
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_latest_feedback_id_by_metric()
        .withf(move |id| *id == target_id)
        .returning(move |_| {
            Ok(LatestFeedbackIdByMetricResponse {
                feedback_id_by_metric: HashMap::new(),
            })
        });

    let tool = GetLatestFeedbackByMetricTool;
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
    let feedback_map = result["feedback_id_by_metric"]
        .as_object()
        .expect("Should have feedback_id_by_metric");
    assert!(feedback_map.is_empty(), "Should have empty feedback map");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_latest_feedback_by_metric_tool_error(pool: PgPool) {
    let llm_params = GetLatestFeedbackByMetricToolParams {
        target_id: Uuid::now_v7(),
    };

    let side_info = AutopilotToolSideInfo {
        episode_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        tool_call_id: Uuid::now_v7(),
        tool_call_event_id: Uuid::now_v7(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_latest_feedback_id_by_metric()
        .returning(|_| Err(TensorZeroClientError::AutopilotUnavailable));

    let tool = GetLatestFeedbackByMetricTool;
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

    assert!(result.is_err(), "Should return error when client fails");
}
