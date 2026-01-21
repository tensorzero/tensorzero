//! Integration tests for InferenceTool and ListInferencesTool.

mod common;

use std::sync::Arc;

use autopilot_client::{AutopilotSideInfo, OptimizationWorkflowSideInfo};
use durable::MIGRATOR;
use durable_tools::{ErasedSimpleTool, SimpleToolContext, TensorZeroClientError};
use sqlx::PgPool;
use tensorzero::{
    ActionInput, GetInferencesRequest, GetInferencesResponse, InferenceOutputSource, Input,
    InputMessage, InputMessageContent, ListInferencesRequest, Role,
};
use tensorzero_core::inference::types::Text;
use uuid::Uuid;

use autopilot_tools::tools::{
    GetInferencesTool, GetInferencesToolParams, InferenceTool, InferenceToolParams,
    ListInferencesTool, ListInferencesToolParams,
};
use common::{MockTensorZeroClient, create_mock_chat_response};

use crate::common::create_mock_stored_chat_inference;

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_inference_tool_with_snapshot_hash(pool: PgPool) {
    // Create mock response
    let mock_response = create_mock_chat_response("Hello from action!");

    // Prepare test data
    let session_id = Uuid::now_v7();
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
            let ActionInput::Inference(params) = input else {
                return false;
            };
            snapshot_hash.to_string() == test_snapshot_hash
                && params.function_name == Some("test_function".to_string())
                && params.episode_id.is_none()
                && params.dryrun == Some(false)
                && params.stream == Some(false)
                && params.internal
                && params.tags.get("tensorzero::autopilot::session_id")
                    == Some(&session_id.to_string())
                && params.tags.get("tensorzero::autopilot::tool_call_event_id")
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

// ===== ListInferencesTool Tests =====

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_list_inferences_tool_basic(pool: PgPool) {
    let inference_id = Uuid::now_v7();
    let inference =
        create_mock_stored_chat_inference(inference_id, "test_function", "test_variant");
    let mock_response = GetInferencesResponse {
        inferences: vec![inference],
    };

    let llm_params = ListInferencesToolParams {
        request: ListInferencesRequest::default(),
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_list_inferences()
        .return_once(move |_| Ok(mock_response));

    let tool = ListInferencesTool;
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
        .expect("ListInferencesTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_list_inferences_tool_with_filters(pool: PgPool) {
    let mock_response = GetInferencesResponse { inferences: vec![] };

    let llm_params = ListInferencesToolParams {
        request: ListInferencesRequest {
            function_name: Some("specific_function".to_string()),
            variant_name: Some("specific_variant".to_string()),
            episode_id: Some(Uuid::now_v7()),
            limit: Some(50),
            offset: Some(10),
            ..Default::default()
        },
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_list_inferences()
        .withf(|request| {
            request.function_name == Some("specific_function".to_string())
                && request.variant_name == Some("specific_variant".to_string())
                && request.episode_id.is_some()
                && request.limit == Some(50)
                && request.offset == Some(10)
        })
        .return_once(move |_| Ok(mock_response));

    let tool = ListInferencesTool;
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
        .expect("ListInferencesTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_list_inferences_tool_with_cursor_pagination(pool: PgPool) {
    let mock_response = GetInferencesResponse { inferences: vec![] };
    let cursor_id = Uuid::now_v7();

    let llm_params = ListInferencesToolParams {
        request: ListInferencesRequest {
            before: Some(cursor_id),
            limit: Some(20),
            ..Default::default()
        },
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_list_inferences()
        .withf(move |request| request.before == Some(cursor_id) && request.limit == Some(20))
        .return_once(move |_| Ok(mock_response));

    let tool = ListInferencesTool;
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
        .expect("ListInferencesTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_list_inferences_tool_error(pool: PgPool) {
    let llm_params = ListInferencesToolParams {
        request: ListInferencesRequest::default(),
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_list_inferences()
        .returning(|_| Err(TensorZeroClientError::AutopilotUnavailable));

    let tool = ListInferencesTool;
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

// ===== GetInferencesTool Tests =====

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_inferences_tool_basic(pool: PgPool) {
    let inference_id = Uuid::now_v7();
    let inference =
        create_mock_stored_chat_inference(inference_id, "test_function", "test_variant");
    let mock_response = GetInferencesResponse {
        inferences: vec![inference],
    };

    let llm_params = GetInferencesToolParams {
        request: GetInferencesRequest {
            ids: vec![inference_id],
            function_name: None,
            output_source: InferenceOutputSource::Inference,
        },
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_inferences()
        .withf(move |request| {
            request.ids == vec![inference_id]
                && request.function_name.is_none()
                && request.output_source == InferenceOutputSource::Inference
        })
        .return_once(move |_| Ok(mock_response));

    let tool = GetInferencesTool;
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
        .expect("GetInferencesTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_inferences_tool_with_function_name(pool: PgPool) {
    let inference_id = Uuid::now_v7();
    let inference =
        create_mock_stored_chat_inference(inference_id, "specific_function", "test_variant");
    let mock_response = GetInferencesResponse {
        inferences: vec![inference],
    };

    let llm_params = GetInferencesToolParams {
        request: GetInferencesRequest {
            ids: vec![inference_id],
            function_name: Some("specific_function".to_string()),
            output_source: InferenceOutputSource::Inference,
        },
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_inferences()
        .withf(|request| request.function_name == Some("specific_function".to_string()))
        .return_once(move |_| Ok(mock_response));

    let tool = GetInferencesTool;
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
        .expect("GetInferencesTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_inferences_tool_with_output_source(pool: PgPool) {
    let inference_id = Uuid::now_v7();
    let inference =
        create_mock_stored_chat_inference(inference_id, "test_function", "test_variant");
    let mock_response = GetInferencesResponse {
        inferences: vec![inference],
    };

    let llm_params = GetInferencesToolParams {
        request: GetInferencesRequest {
            ids: vec![inference_id],
            function_name: None,
            output_source: InferenceOutputSource::Demonstration,
        },
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_inferences()
        .withf(|request| request.output_source == InferenceOutputSource::Demonstration)
        .return_once(move |_| Ok(mock_response));

    let tool = GetInferencesTool;
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
        .expect("GetInferencesTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_inferences_tool_error(pool: PgPool) {
    let llm_params = GetInferencesToolParams {
        request: GetInferencesRequest {
            ids: vec![Uuid::now_v7()],
            function_name: None,
            output_source: InferenceOutputSource::Inference,
        },
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_inferences()
        .returning(|_| Err(TensorZeroClientError::AutopilotUnavailable));

    let tool = GetInferencesTool;
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
