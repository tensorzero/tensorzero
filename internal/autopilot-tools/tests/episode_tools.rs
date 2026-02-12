//! Integration tests for ListEpisodesTool.

mod common;

use std::sync::Arc;

use autopilot_client::{AutopilotSideInfo, OptimizationWorkflowSideInfo};
use durable::MIGRATOR;
use durable_tools::{ErasedSimpleTool, SimpleToolContext, TensorZeroClientError};
use sqlx::PgPool;
use sqlx::types::chrono::Utc;
use tensorzero_core::db::EpisodeByIdRow;
use uuid::Uuid;

use autopilot_tools::tools::{ListEpisodesTool, ListEpisodesToolParams};
use common::{MockTensorZeroClient, create_mock_list_episodes_response};

// ===== ListEpisodesTool Tests =====

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_list_episodes_tool_basic(pool: PgPool) {
    let episode_id = Uuid::now_v7();
    let inference_id = Uuid::now_v7();
    let now = Utc::now();

    let mock_response = create_mock_list_episodes_response(vec![EpisodeByIdRow {
        episode_id,
        count: 3,
        start_time: now,
        end_time: now,
        last_inference_id: inference_id,
    }]);

    let llm_params = ListEpisodesToolParams {
        limit: 10,
        before: None,
        after: None,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "1234567".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_list_episodes()
        .return_once(move |_| Ok(mock_response));

    let tool = ListEpisodesTool;
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
        .expect("ListEpisodesTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
    let episodes = result.get("episodes").expect("Should have episodes field");
    assert!(episodes.is_array(), "episodes should be an array");
    assert_eq!(
        episodes
            .as_array()
            .expect("episodes should be an array")
            .len(),
        1,
        "Should have 1 episode"
    );
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_list_episodes_tool_with_before_pagination(pool: PgPool) {
    let mock_response = create_mock_list_episodes_response(vec![]);
    let cursor_id = Uuid::now_v7();

    let llm_params = ListEpisodesToolParams {
        limit: 20,
        before: Some(cursor_id),
        after: None,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "1234567".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_list_episodes()
        .withf(move |params| params.limit == 20 && params.before == Some(cursor_id))
        .return_once(move |_| Ok(mock_response));

    let tool = ListEpisodesTool;
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
        .expect("ListEpisodesTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_list_episodes_tool_with_after_pagination(pool: PgPool) {
    let mock_response = create_mock_list_episodes_response(vec![]);
    let cursor_id = Uuid::now_v7();

    let llm_params = ListEpisodesToolParams {
        limit: 15,
        before: None,
        after: Some(cursor_id),
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "1234567".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_list_episodes()
        .withf(move |params| params.limit == 15 && params.after == Some(cursor_id))
        .return_once(move |_| Ok(mock_response));

    let tool = ListEpisodesTool;
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
        .expect("ListEpisodesTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_list_episodes_tool_error(pool: PgPool) {
    let llm_params = ListEpisodesToolParams {
        limit: 10,
        before: None,
        after: None,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "1234567".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_list_episodes()
        .returning(|_| Err(TensorZeroClientError::AutopilotUnavailable));

    let tool = ListEpisodesTool;
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
