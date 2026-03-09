//! Functionality for rejecting tool calls for tools that are not available.

use std::sync::Arc;

use durable_tools_spawn::{SpawnClient, SpawnOptions};
use uuid::Uuid;

use crate::error::AutopilotError;
use crate::types::{AutopilotSideInfo, EventPayloadToolCall};

/// Spawn a durable task to reject a tool call for a missing/unavailable tool.
///
/// This enqueues a task that will send a `NotAvailable` authorization event
/// to the autopilot API so the session can continue without hanging on an
/// unknown tool.
///
/// # Duplicate Prevention
///
/// This function checks if a rejection task already exists for the tool call
/// before spawning a new one to prevent duplicate rejections.
///
/// # Errors
///
/// Returns an error if checking for existing tasks or spawning the durable task fails.
pub async fn reject_missing_tool(
    spawn_client: &Arc<SpawnClient>,
    tool_call: &EventPayloadToolCall,
) -> Result<(), AutopilotError> {
    let tool_call_event_id = tool_call.side_info.tool_call_event_id;

    // Check if we've already spawned a rejection for this tool call
    if check_tool_rejection_exists(spawn_client, tool_call_event_id).await? {
        tracing::debug!(
            tool_name = %tool_call.name,
            tool_call_event_id = %tool_call_event_id,
            "Rejection task already exists, skipping"
        );
        return Ok(());
    }

    let side_info = AutopilotSideInfo {
        session_id: tool_call.side_info.session_id,
        tool_call_event_id,
        config_snapshot_hash: tool_call.side_info.config_snapshot_hash.clone(),
        optimization: tool_call.side_info.optimization.clone(),
    };

    let episode_id = Uuid::now_v7();
    spawn_client
        .spawn_tool_by_name(
            "__auto_reject_tool_call__",
            serde_json::json!({}),
            serde_json::to_value(&side_info)?,
            episode_id,
            SpawnOptions::default(),
        )
        .await?;

    tracing::info!(
        tool_name = %tool_call.name,
        session_id = %side_info.session_id,
        tool_call_event_id = %side_info.tool_call_event_id,
        "Rejecting tool call for missing tool"
    );

    Ok(())
}

/// Check if a rejection task already exists for the given tool call event.
///
/// This queries the durable tasks table to see if we've already spawned
/// a `__auto_reject_tool_call__` task for this specific tool call.
pub async fn check_tool_rejection_exists(
    spawn_client: &Arc<SpawnClient>,
    tool_call_event_id: Uuid,
) -> Result<bool, AutopilotError> {
    let queue_name = spawn_client.queue_name();

    // Query for an existing rejection task with matching tool_call_event_id in side_info.
    // The table name is derived from the queue name, which is defaulted or comes from an env var.
    let query = format!(
        r"
        SELECT EXISTS(
            SELECT 1
            FROM durable.t_{queue_name}
            WHERE task_name = '__auto_reject_tool_call__'
              AND params->'side_info'->>'tool_call_event_id' = $1
        )
        "
    );

    let exists: bool = sqlx::query_scalar(sqlx::AssertSqlSafe(query))
        .bind(tool_call_event_id.to_string())
        .fetch_one(spawn_client.pool())
        .await?;

    Ok(exists)
}
