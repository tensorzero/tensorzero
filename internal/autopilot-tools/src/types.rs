//! Shared types for TensorZero Autopilot tools.

use std::collections::HashMap;

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

impl AutopilotToolSideInfo {
    /// Build autopilot tracking tags from this side info.
    pub fn to_tags(&self) -> HashMap<String, String> {
        let mut tags = HashMap::new();
        tags.insert(
            "autopilot_session_id".to_string(),
            self.session_id.to_string(),
        );
        tags.insert(
            "autopilot_tool_call_id".to_string(),
            self.tool_call_id.to_string(),
        );
        tags.insert(
            "autopilot_tool_call_event_id".to_string(),
            self.tool_call_event_id.to_string(),
        );
        tags
    }

    /// Build autopilot tracking tags, merging with existing tags.
    ///
    /// Autopilot tags take precedence over existing tags if there are conflicts.
    pub fn merge_into_tags(&self, existing: HashMap<String, String>) -> HashMap<String, String> {
        let mut tags = existing;
        tags.extend(self.to_tags());
        tags
    }
}
