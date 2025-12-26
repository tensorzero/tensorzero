//! Integration tests for InferenceTool.

mod common;

use std::sync::Arc;

use durable::MIGRATOR;
use durable_tools::{ErasedSimpleTool, SimpleToolContext};
use sqlx::PgPool;
use tensorzero::{Input, InputMessage, InputMessageContent, Role};
use tensorzero_core::inference::types::Text;
use uuid::Uuid;

use autopilot_tools::tools::{InferenceTool, InferenceToolParams, InferenceToolSideInfo};
use common::{MockInferenceClient, create_mock_chat_response};

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_inference_tool_without_snapshot_hash(pool: PgPool) {
    // Create mock response
    let mock_response = create_mock_chat_response("Hello from mock!");
    let mock_client = Arc::new(MockInferenceClient::new(mock_response));

    // Prepare test data
    let episode_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_id = Uuid::now_v7();

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
        config_snapshot_hash: None, // Testing the non-hash path
    };

    let side_info = InferenceToolSideInfo {
        episode_id,
        session_id,
        tool_call_id,
    };

    // Create the tool and context
    let tool = InferenceTool;
    let inference_client: Arc<dyn durable_tools::InferenceClient> = mock_client.clone();
    let ctx = SimpleToolContext::new(&pool, &inference_client);

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

    // Verify inference() was called (not action())
    let captured = mock_client
        .get_captured_inference_params()
        .await
        .expect("inference should have been called");
    assert!(
        mock_client.get_captured_action_params().await.is_none(),
        "action() should not have been called"
    );

    // Verify params
    assert_eq!(captured.function_name, Some("test_function".to_string()));
    assert_eq!(captured.episode_id, Some(episode_id));
    assert_eq!(captured.dryrun, Some(false));
    assert_eq!(captured.stream, Some(false));
    assert!(captured.internal);
    assert_eq!(
        captured.tags.get("autopilot_session_id"),
        Some(&session_id.to_string())
    );
    assert_eq!(
        captured.tags.get("autopilot_tool_call_id"),
        Some(&tool_call_id.to_string())
    );
}
