use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Reference to a `prompt_template_configs` row.
/// Replaces `ResolvedTomlPathData` in all stored config types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredPromptRef {
    pub prompt_template_version_id: Uuid,
    pub template_key: String,
}

/// A prompt template version stored in the database.
/// This is the stored equivalent of `ResolvedTomlPathData`, which eagerly loads
/// file contents from disk. The stored version keeps the template body inline
/// and tracks its identity via a UUID.
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredPromptTemplate {
    pub id: Uuid,
    pub template_key: String,
    pub source_body: String,
    /// BLAKE3 hash of `source_body`, used to deduplicate identical template content.
    pub content_hash: Vec<u8>,
    pub creation_source: String,
    pub source_autopilot_session_id: Option<Uuid>,
}
