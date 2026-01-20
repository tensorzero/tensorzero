//! Deno ops that bridge TypeScript to Rust ToolContext methods.

use std::cell::RefCell;
use std::rc::Rc;

use deno_core::OpState;
use deno_core::op2;
use deno_error::JsErrorBox;
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::ToolContext;

/// Wrapper type to distinguish episode_id from task_id in OpState.
#[derive(Clone, Copy)]
pub struct EpisodeId(pub Uuid);

/// Get the task ID.
#[op2]
#[string]
pub fn op_task_id(state: &OpState) -> String {
    state.borrow::<Uuid>().to_string()
}

/// Get the episode ID.
#[op2]
#[string]
pub fn op_episode_id(state: &OpState) -> String {
    state.borrow::<EpisodeId>().0.to_string()
}

/// Call another tool and wait for result.
#[op2(async)]
#[serde]
pub async fn op_call_tool(
    state: Rc<RefCell<OpState>>,
    #[string] name: String,
    #[serde] llm_params: JsonValue,
    #[serde] side_info: JsonValue,
) -> Result<JsonValue, JsErrorBox> {
    let ctx = {
        let state = state.borrow();
        state.borrow::<ToolContext>().clone()
    };

    ctx.call_tool(&name, llm_params, side_info, Default::default())
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))
}

/// Spawn a tool to run in background, returns a handle ID.
#[op2(async)]
#[string]
pub async fn op_spawn_tool(
    state: Rc<RefCell<OpState>>,
    #[string] name: String,
    #[serde] llm_params: JsonValue,
    #[serde] side_info: JsonValue,
) -> Result<String, JsErrorBox> {
    let ctx = {
        let state = state.borrow();
        state.borrow::<ToolContext>().clone()
    };

    let handle = ctx
        .spawn_tool(&name, llm_params, side_info, Default::default())
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    // Serialize the handle to JSON string for storage in JS
    serde_json::to_string(&handle).map_err(|e| JsErrorBox::generic(e.to_string()))
}

/// Join a previously spawned tool and get its result.
#[op2(async)]
#[serde]
pub async fn op_join_tool(
    state: Rc<RefCell<OpState>>,
    #[string] handle_json: String,
) -> Result<JsonValue, JsErrorBox> {
    let ctx = {
        let state = state.borrow();
        state.borrow::<ToolContext>().clone()
    };

    let handle =
        serde_json::from_str(&handle_json).map_err(|e| JsErrorBox::generic(e.to_string()))?;
    ctx.join_tool(handle)
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))
}

/// Make an inference call.
#[op2(async)]
#[serde]
pub async fn op_inference(
    state: Rc<RefCell<OpState>>,
    #[serde] params: JsonValue,
) -> Result<JsonValue, JsErrorBox> {
    let ctx = {
        let state = state.borrow();
        state.borrow::<ToolContext>().clone()
    };

    let params: tensorzero::ClientInferenceParams =
        serde_json::from_value(params).map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let response = ctx
        .inference(params)
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    serde_json::to_value(response).map_err(|e| JsErrorBox::generic(e.to_string()))
}

/// Get a durable random number.
#[op2(async)]
pub async fn op_rand(state: Rc<RefCell<OpState>>) -> Result<f64, JsErrorBox> {
    let ctx = {
        let state = state.borrow();
        state.borrow::<ToolContext>().clone()
    };
    ctx.rand()
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))
}

/// Get the current durable timestamp.
#[op2(async)]
#[string]
pub async fn op_now(state: Rc<RefCell<OpState>>) -> Result<String, JsErrorBox> {
    let ctx = {
        let state = state.borrow();
        state.borrow::<ToolContext>().clone()
    };
    let dt = ctx
        .now()
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    Ok(dt.to_rfc3339())
}

/// Generate a durable UUID v7.
#[op2(async)]
#[string]
pub async fn op_uuid7(state: Rc<RefCell<OpState>>) -> Result<String, JsErrorBox> {
    let ctx = {
        let state = state.borrow();
        state.borrow::<ToolContext>().clone()
    };
    let uuid = ctx
        .uuid7()
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
    Ok(uuid.to_string())
}

/// Sleep for a durable duration.
#[op2(async)]
pub async fn op_sleep_for(
    state: Rc<RefCell<OpState>>,
    #[string] name: String,
    #[bigint] duration_ms: u64,
) -> Result<(), JsErrorBox> {
    let ctx = {
        let state = state.borrow();
        state.borrow::<ToolContext>().clone()
    };
    let duration = std::time::Duration::from_millis(duration_ms);
    ctx.sleep_for(&name, duration)
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))
}

/// Wait for an event.
#[op2(async)]
#[serde]
pub async fn op_await_event(
    state: Rc<RefCell<OpState>>,
    #[string] event_name: String,
    #[serde] timeout_ms: Option<u64>,
) -> Result<JsonValue, JsErrorBox> {
    let ctx = {
        let state = state.borrow();
        state.borrow::<ToolContext>().clone()
    };
    let timeout = timeout_ms.map(std::time::Duration::from_millis);
    ctx.await_event(&event_name, timeout)
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))
}

/// Emit an event.
#[op2(async)]
pub async fn op_emit_event(
    state: Rc<RefCell<OpState>>,
    #[string] event_name: String,
    #[serde] payload: JsonValue,
) -> Result<(), JsErrorBox> {
    let ctx = {
        let state = state.borrow();
        state.borrow::<ToolContext>().clone()
    };
    ctx.emit_event(&event_name, &payload)
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))
}
