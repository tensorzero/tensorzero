//! Wrapper that adds result publishing to client tools.

use std::borrow::Cow;
use std::marker::PhantomData;

use autopilot_client::{CreateEventRequest, EventPayload, ToolOutcome};
use durable_tools::{
    SideInfo, TaskTool, ToolContext, ToolResult as DurableToolResult, schemars::schema::RootSchema,
};
use serde::{Deserialize, Serialize};
use tensorzero_core::endpoints::status::TENSORZERO_VERSION;
use tensorzero_core::tool::ToolResult;
use uuid::Uuid;

use crate::context::AutopilotToolContext;
use crate::executable::ExecutableClientTool;
use crate::state::AutopilotExtension;

/// Side information required for all autopilot client tools.
///
/// This wraps tool-specific side info with the autopilot-specific fields
/// needed to send results back to the autopilot API.
///
/// Note: The AutopilotClient itself is accessed via `ToolContext::extension()`,
/// not via side info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopilotSideInfo<S = ()>
where
    S: Default,
{
    /// The event ID of the ToolCall event (for sending ToolResult back).
    pub tool_call_event_id: Uuid,

    /// The tool_call_id from the LLM response.
    pub tool_call_id: String,

    /// The session ID for this autopilot session.
    pub session_id: Uuid,

    /// The deployment_id for API calls.
    pub deployment_id: Uuid,

    /// Tool-specific side info.
    #[serde(default, skip_serializing_if = "is_unit")]
    pub inner: S,
}

fn is_unit<T>(_: &T) -> bool {
    std::mem::size_of::<T>() == 0
}

impl<S: SideInfo + Default> SideInfo for AutopilotSideInfo<S> {}

/// Parameters for the publish_result step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PublishResultParams {
    tool_call_event_id: Uuid,
    tool_call_id: String,
    session_id: Uuid,
    deployment_id: Uuid,
    tool_name: String,
    outcome: ToolOutcome,
}

/// Wrapper that adds result publishing to any [`ExecutableClientTool`].
///
/// This wrapper implements [`TaskTool`] and:
/// 1. Executes the underlying tool
/// 2. Sends the result back to the autopilot API via a checkpointed step
///
/// # Example
///
/// ```ignore
/// use autopilot_worker::{ClientToolWrapper, ExecutableClientTool};
///
/// // Register a wrapped tool
/// executor.register_task_tool::<ClientToolWrapper<MyTool>>().await;
/// ```
pub struct ClientToolWrapper<T: ExecutableClientTool> {
    _marker: PhantomData<T>,
}

impl<T: ExecutableClientTool> Default for ClientToolWrapper<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

#[async_trait::async_trait]
impl<T> TaskTool for ClientToolWrapper<T>
where
    T: ExecutableClientTool,
    T::SideInfo: Default,
{
    fn name() -> Cow<'static, str> {
        T::name()
    }

    fn description() -> Cow<'static, str> {
        T::description()
    }

    fn parameters_schema() -> RootSchema {
        // Convert from schemars 1.x Schema to schemars 0.8 RootSchema via JSON.
        // This is needed because autopilot-tools uses schemars 1.x while
        // durable-tools uses schemars 0.8.
        let schema = T::parameters_schema();
        let json = serde_json::to_value(&schema).expect("schemars Schema should serialize to JSON");
        serde_json::from_value(json).expect("JSON Schema should deserialize to RootSchema")
    }

    type LlmParams = T::LlmParams;
    type SideInfo = AutopilotSideInfo<T::SideInfo>;
    type Output = T::Output;

    async fn execute(
        llm_params: Self::LlmParams,
        side_info: Self::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> DurableToolResult<Self::Output> {
        execute_client_tool_impl::<T>(llm_params, side_info, ctx).await
    }
}

/// Internal implementation of client tool execution.
///
/// This is called by the `ClientToolWrapper` implementation.
pub async fn execute_client_tool_impl<T: ExecutableClientTool>(
    llm_params: T::LlmParams,
    side_info: AutopilotSideInfo<T::SideInfo>,
    ctx: &mut ToolContext<'_>,
) -> DurableToolResult<T::Output>
where
    T::SideInfo: Default,
{
    // Get the autopilot extension from the context and clone the gateway state
    // to avoid borrow conflicts (we need both immutable access to gateway_state
    // and mutable access to ctx)
    let gateway_state = ctx
        .extension::<AutopilotExtension>()
        .ok_or_else(|| durable_tools::ToolError::ExecutionFailed(anyhow::anyhow!(
            "AutopilotExtension not found in ToolAppState. Make sure to set it via ToolExecutorBuilder::extension()"
        )))?
        .gateway_state
        .clone();

    // Create the autopilot tool context with gateway state access
    let mut autopilot_ctx = AutopilotToolContext::new(ctx, &gateway_state);

    // Execute the underlying tool
    let result = T::execute(llm_params, side_info.inner, &mut autopilot_ctx).await;

    // Prepare the outcome for the autopilot API
    let tool_name = T::name().to_string();
    let outcome = match &result {
        Ok(output) => {
            let result_json = serde_json::to_string(&output)
                .unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"));
            ToolOutcome::Success(ToolResult {
                name: tool_name.clone(),
                result: result_json,
                id: side_info.tool_call_id.clone(),
            })
        }
        Err(e) => ToolOutcome::Failure {
            message: e.to_string(),
        },
    };

    // Send result back to autopilot API (as checkpointed step)
    let publish_params = PublishResultParams {
        tool_call_event_id: side_info.tool_call_event_id,
        tool_call_id: side_info.tool_call_id,
        session_id: side_info.session_id,
        deployment_id: side_info.deployment_id,
        tool_name,
        outcome,
    };

    // Get fresh mutable reference to ctx after autopilot_ctx is dropped
    let ctx = autopilot_ctx.tool_ctx();
    ctx.step("publish_result", publish_params, publish_result_step)
        .await?;

    // Return the original result, converting error if needed
    result.map_err(|e| durable_tools::ToolError::ExecutionFailed(e.into()))
}

/// Step function to publish the tool result to the autopilot API.
async fn publish_result_step(
    params: PublishResultParams,
    state: durable_tools::ToolAppState,
) -> anyhow::Result<()> {
    // Get the autopilot client from the extension
    let extension = state
        .extension::<AutopilotExtension>()
        .ok_or_else(|| anyhow::anyhow!("AutopilotExtension not found in ToolAppState"))?;

    // Send the tool result event
    extension
        .autopilot_client
        .create_event(
            params.session_id,
            CreateEventRequest {
                deployment_id: params.deployment_id,
                tensorzero_version: TENSORZERO_VERSION.to_string(),
                payload: EventPayload::ToolResult {
                    tool_call_event_id: params.tool_call_event_id,
                    outcome: params.outcome,
                },
                previous_user_message_event_id: None,
            },
        )
        .await?;

    Ok(())
}
