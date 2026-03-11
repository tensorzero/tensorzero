//! Shared parameter types for task spawning.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Wrapper params that include `episode_id`, LLM params, and side info.
///
/// This is what gets serialized as the durable task params.
///
/// **Important**: This type is shared between `durable-tools-spawn` and
/// `durable-tools`. Both crates use identical serialization to ensure
/// tasks spawned by one can be executed by the other.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskToolParams<L, S = ()> {
    /// The LLM-provided parameters.
    pub llm_params: L,
    /// Side information (hidden from LLM).
    pub side_info: S,
    /// The episode ID for this execution.
    pub episode_id: Uuid,
}
