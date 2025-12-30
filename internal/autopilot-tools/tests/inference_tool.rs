//! Integration tests for InferenceTool.

mod common;

use std::sync::Arc;

use durable::MIGRATOR;
use durable_tools::{ErasedSimpleTool, SimpleToolContext};
use sqlx::PgPool;
use tensorzero::{ActionInput, Input, InputMessage, InputMessageContent, Role};
use tensorzero_core::inference::types::Text;
use uuid::Uuid;

use autopilot_tools::tools::{InferenceTool, InferenceToolParams, InferenceToolSideInfo};
use common::{MockTensorZeroClient, create_mock_chat_response};

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_inference_tool_without_snapshot_hash(pool: PgPool) {
    // Create mock response
    let mock_response = create_mock_chat_response("Hello from mock!");

    // Prepare test data
    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let input = Input {
        system: None,
        messages: vec![InputMessage {
            role: Role::User,
            content: vec![InputMessageContent::Text(Text {
                text: "Hello".to_string(),
            })],
        }],
    };

    let llm_params = InferenceToolParams {
        function_name: Some("test_function".to_string()),
        model_name: None,
        input,
        params: Default::default(),
        variant_name: None,
        dynamic_tool_params: Default::default(),
        output_schema: None,
    };

    let side_info = InferenceToolSideInfo {
        episode_id,
        session_id,
        tool_call_id,
        tool_call_event_id,
        config_snapshot_hash: None,
    };

    // Create mock client with expectations
    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_inference()
        .withf(move |params| {
            params.function_name == Some("test_function".to_string())
                && params.episode_id == Some(episode_id)
                && params.dryrun == Some(false)
                && params.stream == Some(false)
                && params.internal
                && params.tags.get("autopilot_session_id") == Some(&session_id.to_string())
                && params.tags.get("autopilot_tool_call_id") == Some(&tool_call_id.to_string())
                && params.tags.get("autopilot_tool_call_event_id")
                    == Some(&tool_call_event_id.to_string())
        })
        .returning(move |_| Ok(mock_response.clone()));

    // Create the tool and context
    let tool = InferenceTool;
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
        .expect("InferenceTool execution should succeed");

    // The result should be an InferenceResponse (serialized as JSON)
    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_inference_tool_with_snapshot_hash(pool: PgPool) {
    // Create mock response
    let mock_response = create_mock_chat_response("Hello from action!");

    // Prepare test data
    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let input = Input {
        system: None,
        messages: vec![InputMessage {
            role: Role::User,
            content: vec![InputMessageContent::Text(Text {
                text: "Hello via action".to_string(),
            })],
        }],
    };

    // Use a test snapshot hash to trigger the action path
    let test_snapshot_hash = "12345678901234567890";

    let llm_params = InferenceToolParams {
        function_name: Some("test_function".to_string()),
        model_name: None,
        input,
        params: Default::default(),
        variant_name: None,
        dynamic_tool_params: Default::default(),
        output_schema: None,
    };

    let side_info = InferenceToolSideInfo {
        episode_id,
        session_id,
        tool_call_id,
        tool_call_event_id,
        config_snapshot_hash: Some(test_snapshot_hash.to_string()),
    };

    // Create mock client with expectations for action()
    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_action()
        .withf(move |snapshot_hash, input| {
            let ActionInput::Inference(params) = input else {
                return false;
            };
            snapshot_hash.to_string() == test_snapshot_hash
                && params.function_name == Some("test_function".to_string())
                && params.episode_id == Some(episode_id)
                && params.dryrun == Some(false)
                && params.stream == Some(false)
                && params.internal
                && params.tags.get("autopilot_session_id") == Some(&session_id.to_string())
                && params.tags.get("autopilot_tool_call_id") == Some(&tool_call_id.to_string())
                && params.tags.get("autopilot_tool_call_event_id")
                    == Some(&tool_call_event_id.to_string())
        })
        .returning(move |_, _| Ok(mock_response.clone()));

    // Create the tool and context
    let tool = InferenceTool;
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
        .expect("InferenceTool execution should succeed");

    // The result should be an InferenceResponse (serialized as JSON)
    assert!(result.is_object(), "Result should be a JSON object");
}
