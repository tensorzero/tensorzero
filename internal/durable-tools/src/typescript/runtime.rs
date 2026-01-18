//! JsRuntime setup with the typescript-tool extension.

use deno_core::{JsRuntime, RuntimeOptions, extension};
use uuid::Uuid;

use super::ops::EpisodeId;
use crate::ToolContext;

// Define the extension using the macro
extension!(
    typescript_tool_ext,
    ops = [
        super::ops::op_task_id,
        super::ops::op_episode_id,
        super::ops::op_call_tool,
        super::ops::op_spawn_tool,
        super::ops::op_join_tool,
        super::ops::op_inference,
        super::ops::op_rand,
        super::ops::op_now,
        super::ops::op_uuid7,
        super::ops::op_sleep_for,
        super::ops::op_await_event,
        super::ops::op_emit_event,
    ],
    esm_entry_point = "ext:typescript_tool_ext/runtime.js",
    esm = [dir "src/typescript", "runtime.js"],
);

/// Create a new JsRuntime configured with the typescript-tool extension.
///
/// The runtime has access to all ToolContext methods via the global `ctx` object.
///
/// Note: This function is intended to be called on a worker thread where the
/// runtime will live for the thread's lifetime. The context, task_id, and
/// episode_id can be updated via `update_runtime_context` before each execution.
pub fn create_runtime() -> JsRuntime {
    JsRuntime::new(RuntimeOptions {
        extensions: vec![typescript_tool_ext::init()],
        ..Default::default()
    })
}

/// Update the ToolContext, task_id, and episode_id in the runtime's OpState.
///
/// Call this before each tool execution to set the context for that execution.
pub fn update_runtime_context(
    runtime: &JsRuntime,
    ctx: ToolContext,
    task_id: Uuid,
    episode_id: Uuid,
) {
    let op_state = runtime.op_state();
    let mut state = op_state.borrow_mut();

    // Remove old values if present and insert new ones
    if state.try_take::<ToolContext>().is_some() {
        // Removed old context
    }
    if state.try_take::<Uuid>().is_some() {
        // Removed old task_id
    }
    if state.try_take::<EpisodeId>().is_some() {
        // Removed old episode_id
    }

    state.put(ctx);
    state.put(task_id);
    state.put(EpisodeId(episode_id));
}
