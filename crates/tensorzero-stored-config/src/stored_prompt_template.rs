use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredPromptRef {
    pub prompt_template_version_id: Uuid,
    pub template_key: String,
}
