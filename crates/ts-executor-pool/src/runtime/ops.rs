// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

use std::{cell::RefCell, rc::Rc, sync::Arc};

use deno_core::{OpState, extension, op2};
use deno_error::JsErrorBox;
use durable::{ControlFlow, SpawnOptions};
use durable_tools::{TensorZeroClient, ToolError, ToolHandle};
use serde::{Deserialize, Serialize};
use tokio_util::task::AbortOnDropHandle;
use tracing::{Instrument, Span};

use crate::{
    ExtraInferenceTags, TsError,
    error::format_non_control_tool_error_for_llm,
    llm_query::llm_query_with_timeout,
    runtime::{
        ConsoleLogBuffer, ExposedTools, FinalAnswer, MAX_CONSOLE_LOG_CHARS, MainRuntimeHandle,
        RlmPermit, RlmRuntimeState, TsCheckerRef, control_flow_to_js_error,
    },
    state::OomSnapshotConfig,
    tool_result::parse_tool_result,
    truncate_to_chars,
};

// ---------------------------------------------------------------------------
// Extension

// ---------------------------------------------------------------------------

extension!(
    rlm_ext,
    // IMPORTANT - any async operations must go through `async_op_helper` to ensure that the future is spawned on the main runtime.
    // When deno_core returns from `run_event_loop`, any futures that it 'owns' (e.g. in-progress async operations from `rlm_ext`)
    // will not get polled until the next `run_event_loop` call. In particular, any mutexes/semaphore permits inside of those futures
    // will remain acquired until the next `run_event_loop` call.
    // This can cause a deadlock if the outer RLM code tries to acquire the same mutex/semaphore - for example, by attempting
    // to call `ctx.heartbeat()`.
    //
    // To avoid this problem, we execute the actual body of all async operations on the main (multi-threaded) runtime,
    // so that they continue to make progress even after `run_event_loop` returns.
    // This is accomplished from within `async_op_helper`, by spawning the future on the main runtime handle and awaiting it.
    // This ensures that the futures 'owned' by the `JsRuntime` just wait for the result of a future executed on another runtime,
    // so suspending the `JsRuntime` futures doesn't prevent any other futures from making progress.
    ops = [op_llm_query, op_llm_query_batched, op_set_final, op_console_log, op_tool_call_dispatch, op_tool_call_join],
    esm_entry_point = "ext:rlm_ext/init.js",
    esm = [dir "src/js", "init.js"],
);

/// This helper functions runs the body of an async op on the main multithreaded runtime,
/// and propagates the cancellation token from the `RlmPermit`
/// See the comment on the `extension!` call for more details
async fn async_op_helper<T: Send + 'static>(
    state: &Rc<RefCell<OpState>>,
    fut: impl Future<Output = Result<Result<T, ControlFlow>, JsErrorBox>> + Send + 'static,
) -> Result<T, JsErrorBox> {
    let main_runtime_handle = state.borrow().borrow::<MainRuntimeHandle>().clone();
    let permit = state.borrow().borrow::<RlmPermit>().clone();
    // If `async_op_helper` is dropped (e.g. from a timeout in the parent code),
    // then cancel the future on the main runtime as well.
    let handle = AbortOnDropHandle::new(main_runtime_handle.0.spawn(
        // Propagate our current span, so that we can trace tool calls all the way back to the original task.
        async move { permit.run_with_cancellation(fut).await }.instrument(Span::current()),
    ));
    let res = handle
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))?
        .map_err(|e: TsError| JsErrorBox::generic(e.to_string()))??;
    res.map_err(|e| control_flow_to_js_error(e, state))
}

/// Store a final answer value from JavaScript.
///
/// Called by `FINAL(value)` in init.js. An empty string is treated as
/// no answer (the loop continues).
#[tracing::instrument(skip_all)]
#[op2(fast)]
fn op_set_final(state: &mut OpState, #[string] value: String) {
    if !value.is_empty() {
        state.put(FinalAnswer(Some(value)));
    }
}

/// Capture a console.log message from JavaScript.
///
/// Writes to `tracing::debug!` and appends to the in-memory log buffer.
/// Messages longer than [`MAX_CONSOLE_LOG_CHARS`] are truncated with a suffix
/// indicating the total length.
#[tracing::instrument(skip_all)]
#[op2(fast)]
fn op_console_log(state: &mut OpState, #[string] message: String) {
    tracing::debug!(target: "rlm_js", "{message}");
    let message_char_count = message.chars().count();
    let truncated = if message_char_count > MAX_CONSOLE_LOG_CHARS {
        let truncated = truncate_to_chars(&message, MAX_CONSOLE_LOG_CHARS);
        format!("{truncated}... (truncated, {message_char_count} chars total)")
    } else {
        message
    };
    state.borrow_mut::<ConsoleLogBuffer>().0.push(truncated);
}

/// Takes a handle produced by `ToolClient` (which internally uses 'op_tool_call_dispatch')
/// and waits for the tool call to complete.
/// This is currently the only 'leaf' source of suspensions - when we get `ToolError::Control(ControlFlow::Suspend)`
/// from waiting on the tool call, we abort the runtime, and set the `SuspendFlag`.
#[tracing::instrument(skip_all)]
#[op2]
#[serde]
async fn op_tool_call_join(
    state: Rc<RefCell<OpState>>,
    #[serde] handle: DenoToolHandle,
) -> Result<serde_json::Value, JsErrorBox> {
    let exposed_tools = {
        let state_ref = state.borrow();
        state_ref
            .try_borrow::<ExposedTools>()
            .cloned()
            .ok_or_else(|| JsErrorBox::generic("No exposed tools configured"))?
    };
    async_op_helper(&state, async move {
        exposed_tools
            .heartbeat()
            .await
            .map_err(|e| JsErrorBox::generic(e.to_string()))?;
        let res = exposed_tools
            .tool_context_helper
            .lock()
            .await
            .join_tool(handle.durable_handle)
            .await;

        match res {
            Ok(val) => match parse_tool_result::<serde_json::Value>(&val) {
                Ok(val) => Ok(Ok(val)),
                Err(e) => Err(JsErrorBox::generic(e.to_string())),
            },
            Err(ToolError::Control(cf)) => Ok(Err(cf)),
            Err(ToolError::NonControl(e)) => Err(JsErrorBox::generic(
                format_non_control_tool_error_for_llm(&e),
            )),
            Err(e) => Err(JsErrorBox::generic(e.to_string())),
        }
    })
    .await
}

// TODO - can we make this completely opaque on the javascript side?
#[derive(Debug, Serialize, Deserialize)]
pub struct DenoToolHandle {
    pub durable_handle: ToolHandle,
}

/// Handles dispatch for the 'ToolClient' interface in the ambient file.
/// This spawns a tool call, and returns an opaque 'Handle' object which can be used to join the tool call.
/// We have javascript glue code which calls into this op with the tool call name and parameters
#[allow(clippy::needless_pass_by_value, clippy::allow_attributes)] // the #[op2] macro requires this
#[tracing::instrument(skip_all)]
#[op2]
#[serde]
async fn op_tool_call_dispatch(
    state: Rc<RefCell<OpState>>,
    #[string] name: String,
    #[serde] params: serde_json::Value,
) -> Result<DenoToolHandle, JsErrorBox> {
    let exposed_tools = {
        let state_ref = state.borrow();
        state_ref
            .try_borrow::<ExposedTools>()
            .cloned()
            .ok_or_else(|| JsErrorBox::generic("No exposed tools configured"))?
    };
    async_op_helper(&state, async move {
        exposed_tools
            .tool_context_helper
            .lock()
            .await
            .heartbeat()
            .await
            .map_err(|e| JsErrorBox::generic(e.to_string()))?;
        let allowed = exposed_tools
            .exposed_tools_data()
            .iter()
            .any(|tool| tool.name == name);
        if !allowed {
            return Err(JsErrorBox::generic(format!(
                "Attempted to invoke unexposed tool {name}"
            )));
        }

        // Construct a new tool call ID for each call invoked from RLM
        let tool_call_id = exposed_tools
            .tool_context_helper
            .lock()
            .await
            .uuid7()
            .await
            .map_err(|e| JsErrorBox::generic(e.to_string()))?
            .to_string();
        let side_info = exposed_tools
            .make_side_info
            .make_side_info(tool_call_id)
            .await
            .map_err(JsErrorBox::generic)?;
        let durable_handle = exposed_tools
            .tool_context_helper
            .lock()
            .await
            .spawn_tool(&name, params, side_info, SpawnOptions::default())
            .await
            .map_err(|e| JsErrorBox::generic(e.to_string()))?;
        Ok(Ok(DenoToolHandle { durable_handle }))
    })
    .await
}

/// Send a prompt to an LLM for analysis.
///
/// At `depth < max_depth`, spawns a child RLM loop (recursive decomposition).
/// At `depth >= max_depth`, makes a single-shot inference call to
/// `rlm_text_analysis`.
#[tracing::instrument(skip_all)]
#[op2(reentrant)]
#[string]
async fn op_llm_query(
    state: Rc<RefCell<OpState>>,
    #[string] prompt: String,
) -> Result<String, JsErrorBox> {
    let (
        t0_client,
        rlm_state,
        extra_inference_tags,
        rlm_permit,
        ts_checker,
        exposed_tools,
        oom_snapshot_config,
    ) = {
        let state_ref = state.borrow();
        let client = state_ref.borrow::<Arc<dyn TensorZeroClient>>().clone();
        let rlm_state = state_ref.borrow::<RlmRuntimeState>().clone();
        let extra_inference_tags = state_ref.borrow::<ExtraInferenceTags>().clone();
        let rlm_permit = state_ref.borrow::<RlmPermit>().clone();
        let ts_checker = state_ref.borrow::<TsCheckerRef>().0.clone();
        let exposed_tools = state_ref.try_borrow::<ExposedTools>().cloned();
        let oom_snapshot_config = state_ref.try_borrow::<OomSnapshotConfig>().cloned();
        (
            client,
            rlm_state,
            extra_inference_tags,
            rlm_permit,
            ts_checker,
            exposed_tools,
            oom_snapshot_config,
        )
    };
    // This propagates a child suspension (e.g. `llm_query_with_timeout` performed a tool call
    // on the child runtime) back to the parent. This ensures that a suspension at any depth
    // will propagate all the way back to the original task, which will then suspend the worker.
    async_op_helper(&state, async move {
        if let Some(ref exposed_tools) = exposed_tools {
            exposed_tools
                .heartbeat()
                .await
                .map_err(|e| JsErrorBox::generic(e.to_string()))?;
        }

        Box::pin(rlm_permit.run_with_cancellation(llm_query_with_timeout(
            t0_client,
            &rlm_state,
            &extra_inference_tags,
            &prompt,
            &rlm_permit,
            ts_checker,
            exposed_tools,
            oom_snapshot_config,
        )))
        .await
        .map_err(|e: TsError| JsErrorBox::generic(e.to_string()))?
        .map_err(|e: TsError| JsErrorBox::generic(e.to_string()))
    })
    .await
}

/// Send multiple prompts concurrently and collect results in order.
///
/// Like `op_llm_query` but processes a batch using `join_all` for
/// concurrent execution.
#[tracing::instrument(skip_all)]
#[op2(reentrant)]
#[serde]
async fn op_llm_query_batched(
    state: Rc<RefCell<OpState>>,
    #[serde] prompts: Vec<String>,
) -> Result<Vec<String>, JsErrorBox> {
    let (
        t0_client,
        rlm_state,
        extra_inference_tags,
        rlm_permit,
        ts_checker,
        exposed_tools,
        oom_snapshot_config,
    ) = {
        let state_ref = state.borrow();
        let client = state_ref.borrow::<Arc<dyn TensorZeroClient>>().clone();
        let rlm_state = state_ref.borrow::<RlmRuntimeState>().clone();
        let extra_inference_tags = state_ref.borrow::<ExtraInferenceTags>().clone();
        let rlm_permit = state_ref.borrow::<RlmPermit>().clone();
        let ts_checker = state_ref.borrow::<TsCheckerRef>().0.clone();
        let exposed_tools = state_ref.try_borrow::<ExposedTools>().cloned();
        let oom_snapshot_config = state_ref.try_borrow::<OomSnapshotConfig>().cloned();
        (
            client,
            rlm_state,
            extra_inference_tags,
            rlm_permit,
            ts_checker,
            exposed_tools,
            oom_snapshot_config,
        )
    };

    async_op_helper(&state, async move {
        if let Some(ref exposed_tools) = exposed_tools {
            exposed_tools
                .heartbeat()
                .await
                .map_err(|e| JsErrorBox::generic(e.to_string()))?;
        }

        let futures: Vec<_> = prompts
            .iter()
            .map(|prompt| {
                llm_query_with_timeout(
                    t0_client.clone(),
                    &rlm_state,
                    &extra_inference_tags,
                    prompt,
                    &rlm_permit,
                    ts_checker.clone(),
                    exposed_tools.clone(),
                    oom_snapshot_config.clone(),
                )
            })
            .collect();

        let results: Vec<Result<Result<String, ControlFlow>, TsError>> = rlm_permit
            .run_with_cancellation(futures::future::join_all(futures))
            .await
            .map_err(|e: TsError| JsErrorBox::generic(e.to_string()))?;

        results
            .into_iter()
            .collect::<Result<Result<Vec<_>, _>, _>>()
            .map_err(|e: TsError| JsErrorBox::generic(e.to_string()))
    })
    .await
}
