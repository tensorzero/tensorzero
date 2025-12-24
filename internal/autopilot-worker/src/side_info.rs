//! Side information for autopilot tools.

use durable_tools::SideInfo;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Side information required for autopilot client tools.
///
/// This wraps tool-specific side info with autopilot-specific fields
/// needed to send results back to the autopilot API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopilotSideInfo<S = ()> {
    /// The event ID of the ToolCall event (for correlating ToolResult).
    pub tool_call_event_id: Uuid,

    /// The tool_call_id from the LLM response.
    pub tool_call_id: String,

    /// The session ID for this autopilot session.
    pub session_id: Uuid,

    /// The deployment_id for API calls.
    pub deployment_id: Uuid,

    /// Tool-specific side info (hidden from LLM).
    pub inner: S,
}

impl<S: SideInfo> SideInfo for AutopilotSideInfo<S> {}
