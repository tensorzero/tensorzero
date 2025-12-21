//! Extension state for autopilot worker.

use std::sync::Arc;

use autopilot_client::AutopilotClient;
use tensorzero_core::utils::gateway::AppStateData;

/// Extension state for autopilot tools.
///
/// This is stored in `ToolAppState::extension` and provides access to
/// autopilot-specific state like the AutopilotClient and gateway state.
#[derive(Clone)]
pub struct AutopilotExtension {
    /// Client for sending events to the autopilot API.
    pub autopilot_client: Arc<AutopilotClient>,
    /// Gateway state for accessing TensorZero functionality.
    pub gateway_state: AppStateData,
}

impl AutopilotExtension {
    /// Create a new autopilot extension.
    pub fn new(autopilot_client: Arc<AutopilotClient>, gateway_state: AppStateData) -> Self {
        Self {
            autopilot_client,
            gateway_state,
        }
    }
}
