//! Integration tests for FeedbackTool.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use durable::MIGRATOR;
use durable_tools::{ErasedSimpleTool, SimpleToolContext, TensorZeroClientError};
use serde_json::json;
use sqlx::PgPool;
use uuid::Uuid;

use autopilot_tools::AutopilotToolSideInfo;
use autopilot_tools::tools::{FeedbackTool, FeedbackToolParams};
use common::{MockTensorZeroClient, create_mock_feedback_response};

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_feedback_tool_comment(pool: PgPool) {
    let feedback_id = Uuid::now_v7();
    let mock_response = create_mock_feedback_response(feedback_id);

    let inference_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = FeedbackToolParams {
        episode_id: None,
        inference_id: Some(inference_id),
        metric_name: "comment".to_string(),
        value: json!("This is a test comment"),
        tags: HashMap::new(),
        dryrun: Some(true),
    };

    let side_info = AutopilotToolSideInfo {
        episode_id: Uuid::now_v7(),
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_feedback()
        .withf(move |params| {
            params.inference_id == Some(inference_id)
                && params.metric_name == "comment"
                && params.internal
                && params.tags.get("autopilot_session_id") == Some(&session_id.to_string())
                && params.tags.get("autopilot_tool_call_id") == Some(&tool_call_id.to_string())
        })
        .return_once(move |_| Ok(mock_response));

    let tool = FeedbackTool;
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
        .expect("FeedbackTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
    assert!(
        result.get("feedback_id").is_some(),
        "Result should contain feedback_id"
    );
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_feedback_tool_float_metric(pool: PgPool) {
    let feedback_id = Uuid::now_v7();
    let mock_response = create_mock_feedback_response(feedback_id);

    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = FeedbackToolParams {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: "user_rating".to_string(),
        value: json!(4.5),
        tags: HashMap::new(),
        dryrun: Some(true),
    };

    let side_info = AutopilotToolSideInfo {
        episode_id: Uuid::now_v7(),
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_feedback()
        .withf(move |params| {
            params.episode_id == Some(episode_id)
                && params.metric_name == "user_rating"
                && params.value == json!(4.5)
                && params.internal
        })
        .return_once(move |_| Ok(mock_response));

    let tool = FeedbackTool;
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
        .expect("FeedbackTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_feedback_tool_boolean_metric(pool: PgPool) {
    let feedback_id = Uuid::now_v7();
    let mock_response = create_mock_feedback_response(feedback_id);

    let inference_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = FeedbackToolParams {
        episode_id: None,
        inference_id: Some(inference_id),
        metric_name: "thumbs_up".to_string(),
        value: json!(true),
        tags: HashMap::new(),
        dryrun: Some(true),
    };

    let side_info = AutopilotToolSideInfo {
        episode_id: Uuid::now_v7(),
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_feedback()
        .withf(move |params| {
            params.inference_id == Some(inference_id)
                && params.metric_name == "thumbs_up"
                && params.value == json!(true)
        })
        .return_once(move |_| Ok(mock_response));

    let tool = FeedbackTool;
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
        .expect("FeedbackTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_feedback_tool_user_tags_take_precedence(pool: PgPool) {
    let feedback_id = Uuid::now_v7();
    let mock_response = create_mock_feedback_response(feedback_id);

    let inference_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    // Create user tags including one that conflicts with autopilot tag
    let mut user_tags = HashMap::new();
    user_tags.insert(
        "autopilot_session_id".to_string(),
        "user_override".to_string(),
    );
    user_tags.insert("custom_tag".to_string(), "custom_value".to_string());

    let llm_params = FeedbackToolParams {
        episode_id: None,
        inference_id: Some(inference_id),
        metric_name: "comment".to_string(),
        value: json!("Test comment"),
        tags: user_tags,
        dryrun: Some(true),
    };

    let side_info = AutopilotToolSideInfo {
        episode_id: Uuid::now_v7(),
        session_id,
        tool_call_id,
        tool_call_event_id,
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_feedback()
        .withf(move |params| {
            // User tag should take precedence
            params.tags.get("autopilot_session_id") == Some(&"user_override".to_string())
                // Custom user tag should be preserved
                && params.tags.get("custom_tag") == Some(&"custom_value".to_string())
                // Other autopilot tags should still be present
                && params.tags.get("autopilot_tool_call_id") == Some(&tool_call_id.to_string())
        })
        .return_once(move |_| Ok(mock_response));

    let tool = FeedbackTool;
    let t0_client: Arc<dyn durable_tools::TensorZeroClient> = Arc::new(mock_client);
    let ctx = SimpleToolContext::new(&pool, &t0_client);

    tool.execute_erased(
        serde_json::to_value(&llm_params).expect("Failed to serialize llm_params"),
        serde_json::to_value(&side_info).expect("Failed to serialize side_info"),
        ctx,
        "test-idempotency-key",
    )
    .await
    .expect("FeedbackTool execution should succeed");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_feedback_tool_error(pool: PgPool) {
    let llm_params = FeedbackToolParams {
        episode_id: None,
        inference_id: Some(Uuid::now_v7()),
        metric_name: "comment".to_string(),
        value: json!("Test comment"),
        tags: HashMap::new(),
        dryrun: Some(true),
    };

    let side_info = AutopilotToolSideInfo {
        episode_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        tool_call_id: Uuid::now_v7(),
        tool_call_event_id: Uuid::now_v7(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_feedback()
        .returning(|_| Err(TensorZeroClientError::AutopilotUnavailable));

    let tool = FeedbackTool;
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
