use serde::{Deserialize, Serialize};

use crate::StoredPromptRef;

pub const STORED_TOOL_CONFIG_SCHEMA_REVISION: i32 = 1;

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredToolConfig {
    pub description: String,
    pub parameters: StoredPromptRef,
    pub name: Option<String>,
    pub strict: bool,
}
