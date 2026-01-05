//! Integration tests for feedback tools.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use autopilot_client::{AutopilotSideInfo, OptimizationWorkflowSideInfo};
use durable::MIGRATOR;
use durable_tools::{ErasedSimpleTool, SimpleToolContext, TensorZeroClientError};
use serde_json::json;
use sqlx::PgPool;
use tensorzero_core::endpoints::feedback::internal::LatestFeedbackIdByMetricResponse;
use uuid::Uuid;

use autopilot_tools::tools::{
    FeedbackTool, FeedbackToolParams, GetFeedbackByVariantTool, GetFeedbackByVariantToolParams,
    GetLatestFeedbackByMetricTool, GetLatestFeedbackByMetricToolParams,
};
use common::{
    MockTensorZeroClient, create_mock_feedback_by_variant, create_mock_feedback_response,
};

// ========== FeedbackTool Tests ==========

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_feedback_tool_comment(pool: PgPool) {
    let feedback_id = Uuid::now_v7();
    let mock_response = create_mock_feedback_response(feedback_id);

    let inference_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = FeedbackToolParams {
        episode_id: None,
        inference_id: Some(inference_id),
        metric_name: "comment".to_string(),
        value: json!("This is a test comment"),
        dryrun: Some(true),
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_feedback()
        .withf(move |params| {
            params.inference_id == Some(inference_id)
                && params.metric_name == "comment"
                && params.internal
                && params.tags.get("tensorzero::autopilot::session_id")
                    == Some(&session_id.to_string())
                && params.tags.get("tensorzero::autopilot::tool_call_event_id")
                    == Some(&tool_call_event_id.to_string())
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
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = FeedbackToolParams {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: "user_rating".to_string(),
        value: json!(4.5),
        dryrun: Some(true),
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: OptimizationWorkflowSideInfo::default(),
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
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = FeedbackToolParams {
        episode_id: None,
        inference_id: Some(inference_id),
        metric_name: "thumbs_up".to_string(),
        value: json!(true),
        dryrun: Some(true),
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: OptimizationWorkflowSideInfo::default(),
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
async fn test_feedback_tool_error(pool: PgPool) {
    let llm_params = FeedbackToolParams {
        episode_id: None,
        inference_id: Some(Uuid::now_v7()),
        metric_name: "comment".to_string(),
        value: json!("Test comment"),
        dryrun: Some(true),
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: None,
        optimization: OptimizationWorkflowSideInfo::default(),
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

// ========== GetLatestFeedbackByMetricTool Tests ==========

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_latest_feedback_by_metric_tool_success(pool: PgPool) {
    let target_id = Uuid::now_v7();
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let mut expected_map = HashMap::new();
    expected_map.insert("accuracy".to_string(), Uuid::now_v7().to_string());
    expected_map.insert("quality".to_string(), Uuid::now_v7().to_string());

    let llm_params = GetLatestFeedbackByMetricToolParams { target_id };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: OptimizationWorkflowSideInfo::default(),
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
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = GetLatestFeedbackByMetricToolParams { target_id };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: OptimizationWorkflowSideInfo::default(),
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

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: None,
        optimization: OptimizationWorkflowSideInfo::default(),
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

// ========== GetFeedbackByVariantTool Tests ==========

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_feedback_by_variant_tool_success(pool: PgPool) {
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let mock_response = vec![
        create_mock_feedback_by_variant("variant_a", 0.85, 100),
        create_mock_feedback_by_variant("variant_b", 0.72, 150),
    ];

    let llm_params = GetFeedbackByVariantToolParams {
        metric_name: "accuracy".to_string(),
        function_name: "test_function".to_string(),
        variant_names: None,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_feedback_by_variant()
        .withf(|metric, function, variants| {
            metric == "accuracy" && function == "test_function" && variants.is_none()
        })
        .return_once(move |_, _, _| Ok(mock_response));

    let tool = GetFeedbackByVariantTool;
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

    assert!(result.is_array(), "Result should be a JSON array");
    let array = result.as_array().expect("Should be array");
    assert_eq!(array.len(), 2, "Should have 2 variants");
    assert_eq!(array[0]["variant_name"], "variant_a");
    assert_eq!(array[1]["variant_name"], "variant_b");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_feedback_by_variant_tool_with_variant_filter(pool: PgPool) {
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let mock_response = vec![create_mock_feedback_by_variant("variant_a", 0.85, 100)];

    let llm_params = GetFeedbackByVariantToolParams {
        metric_name: "accuracy".to_string(),
        function_name: "test_function".to_string(),
        variant_names: Some(vec!["variant_a".to_string()]),
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_feedback_by_variant()
        .withf(|metric, function, variants| {
            metric == "accuracy"
                && function == "test_function"
                && variants.as_ref().map(|v| v.len()) == Some(1)
        })
        .return_once(move |_, _, _| Ok(mock_response));

    let tool = GetFeedbackByVariantTool;
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

    assert!(result.is_array(), "Result should be a JSON array");
    let array = result.as_array().expect("Should be array");
    assert_eq!(array.len(), 1, "Should have 1 variant");
    assert_eq!(array[0]["variant_name"], "variant_a");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_feedback_by_variant_tool_empty_result(pool: PgPool) {
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let llm_params = GetFeedbackByVariantToolParams {
        metric_name: "nonexistent_metric".to_string(),
        function_name: "test_function".to_string(),
        variant_names: None,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id,
        session_id,
        config_snapshot_hash: None,
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_feedback_by_variant()
        .return_once(move |_, _, _| Ok(vec![]));

    let tool = GetFeedbackByVariantTool;
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

    assert!(result.is_array(), "Result should be a JSON array");
    let array = result.as_array().expect("Should be array");
    assert!(array.is_empty(), "Should have no variants");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_feedback_by_variant_tool_error(pool: PgPool) {
    let llm_params = GetFeedbackByVariantToolParams {
        metric_name: "accuracy".to_string(),
        function_name: "test_function".to_string(),
        variant_names: None,
    };

    let side_info = AutopilotSideInfo {
        tool_call_event_id: Uuid::now_v7(),
        session_id: Uuid::now_v7(),
        config_snapshot_hash: None,
        optimization: OptimizationWorkflowSideInfo::default(),
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_feedback_by_variant()
        .returning(|_, _, _| {
            Err(TensorZeroClientError::NotSupported(
                "test error".to_string(),
            ))
        });

    let tool = GetFeedbackByVariantTool;
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
