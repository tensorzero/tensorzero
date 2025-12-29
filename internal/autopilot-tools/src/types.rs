//! Shared types for TensorZero Autopilot tools.

use durable_tools::SideInfo;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Side information for autopilot tools (hidden from LLM).
///
/// This provides context about the autopilot session that spawned the tool.
/// Tools can use this to tag their outputs for tracking and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopilotToolSideInfo {
    /// Episode ID to use for the tool (links to autopilot session).
    pub episode_id: Uuid,
    /// Session ID for tagging.
    pub session_id: Uuid,
    /// Tool call ID for tagging.
    pub tool_call_id: Uuid,
    /// Tool call event ID for tagging.
    pub tool_call_event_id: Uuid,
}

impl SideInfo for AutopilotToolSideInfo {}
