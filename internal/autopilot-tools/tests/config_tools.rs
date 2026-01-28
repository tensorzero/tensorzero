//! Integration tests for config snapshot tools.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use autopilot_client::{AutopilotSideInfo, OptimizationWorkflowSideInfo};
use durable::MIGRATOR;
use durable_tools::{ErasedSimpleTool, SimpleToolContext};
use sqlx::PgPool;
use tensorzero::{GetConfigResponse, WriteConfigResponse};
use tensorzero_core::config::UninitializedConfig;
use uuid::Uuid;

use autopilot_tools::tools::{
    GetConfigTool, GetConfigToolParams, WriteConfigTool, WriteConfigToolParams,
};
use common::MockTensorZeroClient;

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_get_config_tool_with_hash(pool: PgPool) {
    let config: UninitializedConfig =
        serde_json::from_value(serde_json::json!({})).expect("Config should deserialize");

    let response = GetConfigResponse {
        hash: "1234567".to_string(),
        config,
        extra_templates: HashMap::new(),
        tags: HashMap::new(),
    };

    let llm_params = GetConfigToolParams {};

    let side_info = AutopilotSideInfo {
        session_id: Uuid::now_v7(),
        config_snapshot_hash: "1234567".to_string(),
        tool_call_event_id: Uuid::now_v7(),
        optimization: OptimizationWorkflowSideInfo {
            poll_interval_secs: 10,
            max_wait_secs: 10,
        },
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_get_config_snapshot()
        .withf(|hash| hash.as_deref() == Some("1234567"))
        .return_once(move |_| Ok(response));

    let tool = GetConfigTool;
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
        .expect("GetConfigTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}

#[sqlx::test(migrator = "MIGRATOR")]
async fn test_write_config_tool_sets_autopilot_tags(pool: PgPool) {
    let session_id = Uuid::now_v7();
    let tool_call_event_id = Uuid::now_v7();

    let side_info = AutopilotSideInfo {
        session_id,
        config_snapshot_hash: "1234567".to_string(),
        tool_call_event_id,
        optimization: OptimizationWorkflowSideInfo {
            poll_interval_secs: 10,
            max_wait_secs: 10,
        },
    };

    let mut extra_templates = HashMap::new();
    extra_templates.insert("template_a".to_string(), "content".to_string());

    let llm_params = WriteConfigToolParams {
        config: serde_json::json!({}),
        extra_templates,
    };

    let mut mock_client = MockTensorZeroClient::new();
    mock_client
        .expect_write_config()
        .withf(move |request| {
            request.config.functions.is_empty()
                && request.extra_templates.get("template_a") == Some(&"content".to_string())
                && request.tags.get("tensorzero::autopilot::session_id")
                    == Some(&session_id.to_string())
                && request
                    .tags
                    .get("tensorzero::autopilot::tool_call_event_id")
                    == Some(&tool_call_event_id.to_string())
                && request
                    .tags
                    .get("tensorzero::autopilot::config_snapshot_hash")
                    == Some(&"1234567".to_string())
        })
        .return_once(|_| {
            Ok(WriteConfigResponse {
                hash: "written_hash".to_string(),
            })
        });

    let tool = WriteConfigTool;
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
        .expect("WriteConfigTool execution should succeed");

    assert!(result.is_object(), "Result should be a JSON object");
}
