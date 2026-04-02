use serde::{Deserialize, Serialize};

use crate::StoredPromptRef;

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredToolConfig {
    pub description: String,
    pub parameters: StoredPromptRef,
    pub name: Option<String>,
    pub strict: bool,
}
