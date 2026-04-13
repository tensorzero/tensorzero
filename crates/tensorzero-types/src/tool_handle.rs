use durable::TaskHandle;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Handle returned by `spawn_tool`, can be joined later with `join_tool`.
///
/// This enum allows a uniform API for both `TaskTool` and `SimpleTool`:
/// - `TaskTool`: spawns as a background subtask, `join_tool` waits for completion
/// - `SimpleTool`: executes immediately (still checkpointed), `join_tool` returns stored result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolHandle {
    /// TaskTool - runs in background, join waits for completion
    Async(TaskHandle<JsonValue>),
    /// SimpleTool - already executed, result stored inline
    Sync(JsonValue),
}
