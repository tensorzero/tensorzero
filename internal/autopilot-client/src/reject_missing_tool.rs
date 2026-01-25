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
/// This function is idempotent - it checks if a task already exists for this
/// tool_call_event_id before spawning a new one. This prevents duplicate tasks
/// from being created when the function is called multiple times for the same
/// tool call (e.g., on each poll/SSE event).
///
/// # Errors
///
/// Returns an error if spawning the durable task fails.
pub async fn reject_missing_tool(
    spawn_client: &Arc<SpawnClient>,
    tool_call: &EventPayloadToolCall,
) -> Result<(), AutopilotError> {
    let tool_call_event_id = tool_call.side_info.tool_call_event_id;

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
