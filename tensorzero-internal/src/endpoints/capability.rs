use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Represents the different endpoint capabilities a model can support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EndpointCapability {
    Chat,
    Embedding,
    Moderation,
    // Future capabilities can be added here:
    // Completions,
    // Images,
    // Audio,
    // FineTuning,
}

impl EndpointCapability {
    /// Returns the human-readable name for error messages
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Embedding => "embedding",
            Self::Moderation => "moderation",
        }
    }
}

/// Default capabilities if none specified (backward compatibility)
pub fn default_capabilities() -> HashSet<EndpointCapability> {
    let mut capabilities = HashSet::new();
    capabilities.insert(EndpointCapability::Chat);
    capabilities
}
