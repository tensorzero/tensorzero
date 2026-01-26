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
/// This function does not check for existing tasks. Duplicate spawns are prevented
/// at a higher level:
/// - For `list_events`: once rejected, the tool is removed from `pending_tool_calls`,
///   so subsequent calls won't see it.
/// - For `stream_events`: once a tool call is streamed and rejected, subsequent
///   interactions typically use `list_events` instead, which won't return it.
///
/// In rare cases, duplicate tasks may be spawned (e.g., if streaming and listing
/// happen concurrently). The cost is minimal: the duplicate task will fail when
/// it tries to reject an already-rejected tool call.
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
