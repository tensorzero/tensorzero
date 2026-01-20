//! Integration tests for datapoint tools.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use autopilot_client::{AutopilotSideInfo, OptimizationWorkflowSideInfo};
use durable::MIGRATOR;
use durable_tools::{ErasedSimpleTool, SimpleToolContext, TensorZeroClientError};
use sqlx::PgPool;
use tensorzero::{
    CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsFromInferenceRequestParams,
    ListDatapointsRequest, ListDatapointsResponse, UpdateChatDatapointRequest,
    UpdateDatapointRequest,
};
use uuid::Uuid;

use autopilot_tools::tools::{
    CreateDatapointsFromInferencesTool, CreateDatapointsFromInferencesToolParams,
    CreateDatapointsTool, CreateDatapointsToolParams, DeleteDatapointsTool,
    DeleteDatapointsToolParams, GetDatapointsTool, GetDatapointsToolParams, ListDatapointsTool,
    ListDatapointsToolParams, UpdateDatapointsTool, UpdateDatapointsToolParams,
};
use common::{
    MockTensorZeroClient, create_mock_chat_datapoint, create_mock_create_datapoints_response,
    create_mock_delete_datapoints_response, create_mock_get_datapoints_response,
    create_mock_update_datapoints_response, create_test_input,
};

// ===== CreateDatapointsTool Tests =====

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_create_datapoints_tool_basic(pool: PgPool) {
    let datapoint_id = Uuid::now_v7();
    let mock_response = create_mock_create_datapoints_response(vec![datapoint_id]);

    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = CreateDatapointsToolParams {
        dataset_name: "test_dataset".to_string(),
        datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "test_function".to_string(),
            episode_id: None,
            input: create_test_input("test input"),
            output: None,
            dynamic_tool_params: Default::default(),
            tags: None,
            name: None,
        })],
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_create_datapoints()
        .withf(|dataset_name, datapoints| dataset_name == "test_dataset" && datapoints.len() == 1)
        .return_once(move |_, _| Ok(mock_response));

    let tool = CreateDatapointsTool;
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
        .expect("CreateDatapointsTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_create_datapoints_tool_adds_autopilot_tags(pool: PgPool) {
    let datapoint_id = Uuid::now_v7();
    let mock_response = create_mock_create_datapoints_response(vec![datapoint_id]);

    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    // Create datapoint with no user tags
    let llm_params = CreateDatapointsToolParams {
        dataset_name: "test_dataset".to_string(),
        datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "test_function".to_string(),
            episode_id: None,
            input: create_test_input("test input"),
            output: None,
            dynamic_tool_params: Default::default(),
            tags: None,
            name: None,
        })],
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_create_datapoints()
        .withf(move |_dataset_name, datapoints| {
            // Verify autopilot tags were added
            match &datapoints[0] {
                CreateDatapointRequest::Chat(chat) => {
                    let tags = chat.tags.as_ref().expect("Tags should be set");
                    tags.get("tensorzero::autopilot::session_id") == Some(&session_id.to_string())
                        && tags.get("tensorzero::autopilot::tool_call_event_id")
                            == Some(&tool_call_event_id.to_string())
                }
                CreateDatapointRequest::Json(_) => false,
            }
        })
        .return_once(move |_, _| Ok(mock_response));

    let tool = CreateDatapointsTool;
    let t0_client: Arc<dyn durable_tools::TensorZeroClient> = Arc::new(mock_client);
    let ctx = SimpleToolContext::new(&pool, &t0_client);

    tool.execute_erased(
        serde_json::to_value(&llm_params).expect("Failed to serialize llm_params"),
        serde_json::to_value(&side_info).expect("Failed to serialize side_info"),
        ctx,
        "test-idempotency-key",
    )
    .await
    .expect("CreateDatapointsTool execution should succeed");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_create_datapoints_tool_user_tags_take_precedence(pool: PgPool) {
    let datapoint_id = Uuid::now_v7();
    let mock_response = create_mock_create_datapoints_response(vec![datapoint_id]);

    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    // Create datapoint with user tags including one that conflicts with autopilot tag
    let mut user_tags = HashMap::new();
    user_tags.insert(
        "tensorzero::autopilot::session_id".to_string(),
        "user_override".to_string(),
    );
    user_tags.insert("custom_tag".to_string(), "custom_value".to_string());

    let llm_params = CreateDatapointsToolParams {
        dataset_name: "test_dataset".to_string(),
        datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "test_function".to_string(),
            episode_id: None,
            input: create_test_input("test input"),
            output: None,
            dynamic_tool_params: Default::default(),
            tags: Some(user_tags),
            name: None,
        })],
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_create_datapoints()
        .withf(move |_dataset_name, datapoints| {
            match &datapoints[0] {
                CreateDatapointRequest::Chat(chat) => {
                    let tags = chat.tags.as_ref().expect("Tags should be set");
                    // User tag should take precedence
                    tags.get("tensorzero::autopilot::session_id") == Some(&"user_override".to_string())
                        // Custom user tag should be preserved
                        && tags.get("custom_tag") == Some(&"custom_value".to_string())
                        // Other autopilot tags should still be present
                        && tags.get("tensorzero::autopilot::tool_call_event_id") == Some(&tool_call_event_id.to_string())
                }
                CreateDatapointRequest::Json(_) => false,
            }
        })
        .return_once(move |_, _| Ok(mock_response));

    let tool = CreateDatapointsTool;
    let t0_client: Arc<dyn durable_tools::TensorZeroClient> = Arc::new(mock_client);
    let ctx = SimpleToolContext::new(&pool, &t0_client);

    tool.execute_erased(
        serde_json::to_value(&llm_params).expect("Failed to serialize llm_params"),
        serde_json::to_value(&side_info).expect("Failed to serialize side_info"),
        ctx,
        "test-idempotency-key",
    )
    .await
    .expect("CreateDatapointsTool execution should succeed");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_create_datapoints_tool_error(pool: PgPool) {
    let llm_params = CreateDatapointsToolParams {
        dataset_name: "test_dataset".to_string(),
        datapoints: vec![CreateDatapointRequest::Chat(CreateChatDatapointRequest {
            function_name: "test_function".to_string(),
            episode_id: None,
            input: create_test_input("test input"),
            output: None,
            dynamic_tool_params: Default::default(),
            tags: None,
            name: None,
        })],
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_create_datapoints()
        .returning(|_, _| Err(TensorZeroClientError::AutopilotUnavailable));

    let tool = CreateDatapointsTool;
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

// ===== GetDatapointsTool Tests =====

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_datapoints_tool_with_dataset_name(pool: PgPool) {
    let datapoint_id = Uuid::now_v7();
    let datapoint = create_mock_chat_datapoint(datapoint_id, "test_dataset", "test_function");
    let mock_response = create_mock_get_datapoints_response(vec![datapoint]);

    let llm_params = GetDatapointsToolParams {
        dataset_name: Some("test_dataset".to_string()),
        ids: vec![datapoint_id],
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_datapoints()
        .withf(move |dataset_name, ids| {
            *dataset_name == Some("test_dataset".to_string()) && *ids == vec![datapoint_id]
        })
        .return_once(move |_, _| Ok(mock_response));

    let tool = GetDatapointsTool;
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
        .expect("GetDatapointsTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_datapoints_tool_without_dataset_name(pool: PgPool) {
    let datapoint_id = Uuid::now_v7();
    let datapoint = create_mock_chat_datapoint(datapoint_id, "test_dataset", "test_function");
    let mock_response = create_mock_get_datapoints_response(vec![datapoint]);

    let llm_params = GetDatapointsToolParams {
        dataset_name: None,
        ids: vec![datapoint_id],
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_datapoints()
        .withf(move |dataset_name, ids| dataset_name.is_none() && *ids == vec![datapoint_id])
        .return_once(move |_, _| Ok(mock_response));

    let tool = GetDatapointsTool;
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
        .expect("GetDatapointsTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_datapoints_tool_error(pool: PgPool) {
    let llm_params = GetDatapointsToolParams {
        dataset_name: Some("test_dataset".to_string()),
        ids: vec![Uuid::now_v7()],
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_datapoints()
        .returning(|_, _| Err(TensorZeroClientError::AutopilotUnavailable));

    let tool = GetDatapointsTool;
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

// ===== ListDatapointsTool Tests =====

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_list_datapoints_tool_basic(pool: PgPool) {
    let datapoint_id = Uuid::now_v7();
    let datapoint = create_mock_chat_datapoint(datapoint_id, "test_dataset", "test_function");
    let mock_response = create_mock_get_datapoints_response(vec![datapoint]);

    let llm_params = ListDatapointsToolParams {
        dataset_name: "test_dataset".to_string(),
        request: ListDatapointsRequest::default(),
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_list_datapoints()
        .withf(|dataset_name, _request| dataset_name == "test_dataset")
        .return_once(move |_, _| Ok(ListDatapointsResponse::Datapoints(mock_response)));

    let tool = ListDatapointsTool;
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
        .expect("ListDatapointsTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_list_datapoints_tool_with_filters(pool: PgPool) {
    let mock_response = create_mock_get_datapoints_response(vec![]);

    let llm_params = ListDatapointsToolParams {
        dataset_name: "test_dataset".to_string(),
        request: ListDatapointsRequest {
            function_name: Some("specific_function".to_string()),
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
        .expect_list_datapoints()
        .withf(|dataset_name, request| {
            dataset_name == "test_dataset"
                && request.function_name == Some("specific_function".to_string())
                && request.limit == Some(50)
                && request.offset == Some(10)
        })
        .return_once(move |_, _| Ok(ListDatapointsResponse::Datapoints(mock_response)));

    let tool = ListDatapointsTool;
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
        .expect("ListDatapointsTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_list_datapoints_tool_error(pool: PgPool) {
    let llm_params = ListDatapointsToolParams {
        dataset_name: "test_dataset".to_string(),
        request: ListDatapointsRequest::default(),
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_list_datapoints()
        .returning(|_, _| Err(TensorZeroClientError::AutopilotUnavailable));

    let tool = ListDatapointsTool;
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

// ===== UpdateDatapointsTool Tests =====

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_update_datapoints_tool_basic(pool: PgPool) {
    let original_id = Uuid::now_v7();
    let new_id = Uuid::now_v7();
    let mock_response = create_mock_update_datapoints_response(vec![new_id]);

    let llm_params = UpdateDatapointsToolParams {
        dataset_name: "test_dataset".to_string(),
        datapoints: vec![UpdateDatapointRequest::Chat(UpdateChatDatapointRequest {
            id: original_id,
            input: Some(create_test_input("updated input")),
            output: None,
            tool_params: Default::default(),
            #[expect(deprecated)]
            deprecated_do_not_use_tool_params: None,
            tags: None,
            metadata: Default::default(),
            #[expect(deprecated)]
            deprecated_do_not_use_metadata: None,
        })],
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_update_datapoints()
        .withf(|dataset_name, datapoints| dataset_name == "test_dataset" && datapoints.len() == 1)
        .return_once(move |_, _| Ok(mock_response));

    let tool = UpdateDatapointsTool;
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
        .expect("UpdateDatapointsTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_update_datapoints_tool_error(pool: PgPool) {
    let llm_params = UpdateDatapointsToolParams {
        dataset_name: "test_dataset".to_string(),
        datapoints: vec![UpdateDatapointRequest::Chat(UpdateChatDatapointRequest {
            id: Uuid::now_v7(),
            input: Some(create_test_input("updated input")),
            output: None,
            tool_params: Default::default(),
            #[expect(deprecated)]
            deprecated_do_not_use_tool_params: None,
            tags: None,
            metadata: Default::default(),
            #[expect(deprecated)]
            deprecated_do_not_use_metadata: None,
        })],
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_update_datapoints()
        .returning(|_, _| Err(TensorZeroClientError::AutopilotUnavailable));

    let tool = UpdateDatapointsTool;
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

// ===== DeleteDatapointsTool Tests =====

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_delete_datapoints_tool_basic(pool: PgPool) {
    let datapoint_id = Uuid::now_v7();
    let mock_response = create_mock_delete_datapoints_response(1);

    let llm_params = DeleteDatapointsToolParams {
        dataset_name: "test_dataset".to_string(),
        ids: vec![datapoint_id],
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_delete_datapoints()
        .withf(move |dataset_name, ids| {
            dataset_name == "test_dataset" && *ids == vec![datapoint_id]
        })
        .return_once(move |_, _| Ok(mock_response));

    let tool = DeleteDatapointsTool;
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
        .expect("DeleteDatapointsTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_delete_datapoints_tool_error(pool: PgPool) {
    let llm_params = DeleteDatapointsToolParams {
        dataset_name: "test_dataset".to_string(),
        ids: vec![Uuid::now_v7()],
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "test_hash".to_string(),
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_delete_datapoints()
        .returning(|_, _| Err(TensorZeroClientError::AutopilotUnavailable));

    let tool = DeleteDatapointsTool;
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

// ===== CreateDatapointsFromInferencesTool Tests =====

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_create_datapoints_from_inferences_tool_with_ids(pool: PgPool) {
    let datapoint_id = Uuid::now_v7();
    let inference_id = Uuid::now_v7();
    let mock_response = create_mock_create_datapoints_response(vec![datapoint_id]);

    let llm_params = CreateDatapointsFromInferencesToolParams {
        dataset_name: "test_dataset".to_string(),
        params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
            inference_ids: vec![inference_id],
            output_source: None,
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
        .expect_create_datapoints_from_inferences()
        .withf(move |dataset_name, params| {
            dataset_name == "test_dataset"
                && matches!(
                    params,
                    CreateDatapointsFromInferenceRequestParams::InferenceIds {
                        inference_ids,
                        output_source: _,
                    } if *inference_ids == vec![inference_id]
                )
        })
        .return_once(move |_, _| Ok(mock_response));

    let tool = CreateDatapointsFromInferencesTool;
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
        .expect("CreateDatapointsFromInferencesTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_create_datapoints_from_inferences_tool_error(pool: PgPool) {
    let llm_params = CreateDatapointsFromInferencesToolParams {
        dataset_name: "test_dataset".to_string(),
        params: CreateDatapointsFromInferenceRequestParams::InferenceIds {
            inference_ids: vec![Uuid::now_v7()],
            output_source: None,
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
        .expect_create_datapoints_from_inferences()
        .returning(|_, _| Err(TensorZeroClientError::AutopilotUnavailable));

    let tool = CreateDatapointsFromInferencesTool;
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
