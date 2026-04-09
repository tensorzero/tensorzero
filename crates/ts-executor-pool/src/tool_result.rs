//! Tool result parsing utilities.

use durable_tools::{NonControlToolError, ToolFailure, ToolResult};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value as JsonValue;

/// The output type used for all tools (server-side and client-side)
/// that we register with the autopilot worker.
/// The success type `T` is the underlying success output of the tool,
/// which can always be deserialized as a `JsonValue`
///
/// We use this to ensure that we can treat server-side and client-side tools uniformly
/// throughout the autopilot codebase. Currently, only client
/// tools will produce `ToolSuccessOrFail::Failure` (when the tool is done
/// running, but we had a failure on the remote client side).
/// In particular, we unconditionally parse this wrapper when parsing the result
/// of a tool dynamically invoked from UserMessageTask, so that we can show a success
/// vs failure message to the LLM.
///
/// Usage of this type is enforced through `ToolExecutorWrapper`, which only allows
/// registering tools that produce a `ToolSuccessOrFail` output.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "client_side_tool_status", rename_all = "snake_case")]
pub enum ToolSuccessOrFail<T = JsonValue> {
    Success { value: T },
    Failure { error: ToolFailure },
}

/// Parse a tool result from `ctx.call_tool()` or `ctx.join_tool()`.
///
/// Handles the `ClientToolOutput` envelope,
/// extracting the success value or converting failures to errors.
///
/// # Errors
///
/// Returns an error if the client tool failed, or if deserialization fails.
pub fn parse_tool_result<T: DeserializeOwned>(result: &JsonValue) -> ToolResult<T> {
    let client_output = serde_json::from_value::<ToolSuccessOrFail>(result.clone())?;
    match client_output {
        ToolSuccessOrFail::Success { value } => serde_json::from_value::<T>(value).map_err(|e| {
            NonControlToolError::Internal {
                message: format!("Failed to deserialize client tool result: {e}"),
            }
            .into()
        }),
        ToolSuccessOrFail::Failure { error } => Err(NonControlToolError::Internal {
            message: format!("Client tool failed: {error:?}"),
        }
        .into()),
    }
}
