use serde::{Deserialize, Serialize};

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredAutopilotConfig {
    pub tool_whitelist: Option<Vec<String>>,
}
