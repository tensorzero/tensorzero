use serde::{Deserialize, Serialize};

pub const STORED_AUTOPILOT_CONFIG_SCHEMA_REVISION: i32 = 1;

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredAutopilotConfig {
    pub tool_whitelist: Option<Vec<String>>,
}
