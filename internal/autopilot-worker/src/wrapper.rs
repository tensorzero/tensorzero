//! Wrapper that adds result publishing to client tools.

use std::borrow::Cow;
use std::marker::PhantomData;

use async_trait::async_trait;
use autopilot_client::ToolResult as AutopilotToolResult;
use durable_tools::{
    CreateEventRequest, EventPayload, TaskTool, ToolAppState, ToolContext, ToolMetadata,
    ToolOutcome, ToolResult as DurableToolResult,
};
use schemars::Schema;
use serde::{Deserialize, Serialize};
use tensorzero_core::endpoints::status::TENSORZERO_VERSION;
use uuid::Uuid;

use crate::side_info::AutopilotSideInfo;

/// Parameters for the publish_result step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PublishResultParams {
    session_id: Uuid,
    deployment_id: Uuid,
    tool_call_event_id: Uuid,
    tool_call_id: String,
    tool_name: String,
    outcome: ToolOutcome,
}

/// Wrapper that adds result publishing to any [`TaskTool`].
///
/// This wrapper:
/// 1. Executes the underlying tool
/// 2. Sends the result back to the autopilot API via a checkpointed step
///
/// # Type Parameters
///
/// * `T` - The underlying TaskTool to wrap
///
/// # Example
///
/// ```ignore
/// // Register a wrapped tool
/// executor.register_task_tool::<ClientTaskToolWrapper<MyTool>>().await;
/// ```
pub struct ClientTaskToolWrapper<T: TaskTool> {
    _marker: PhantomData<T>,
}

impl<T: TaskTool> Default for ClientTaskToolWrapper<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: TaskTool> ToolMetadata for ClientTaskToolWrapper<T> {
    fn name() -> Cow<'static, str> {
        T::name()
    }

    fn description() -> Cow<'static, str> {
        T::description()
    }

    fn parameters_schema() -> DurableToolResult<Schema> {
        T::parameters_schema()
    }

    type LlmParams = T::LlmParams;
    type SideInfo = AutopilotSideInfo<T::SideInfo>;
    type Output = T::Output;
}

#[async_trait]
impl<T> TaskTool for ClientTaskToolWrapper<T>
where
    T: TaskTool,
    T::SideInfo: Default + PartialEq,
{
    async fn execute(
        llm_params: Self::LlmParams,
        side_info: Self::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> DurableToolResult<Self::Output> {
        // Execute the underlying tool
        let result = T::execute(llm_params, side_info.inner, ctx).await;

        // Prepare the outcome for the autopilot API
        let tool_name = T::name().to_string();
        let outcome = match &result {
            Ok(output) => {
                let result_json = serde_json::to_string(output)?;
                ToolOutcome::Success(AutopilotToolResult {
                    name: tool_name.clone(),
                    result: result_json,
                    id: side_info.tool_call_id.clone(),
                })
            }
            Err(e) => ToolOutcome::Failure {
                message: e.to_string(),
            },
        };

        // Publish result to autopilot API (checkpointed)
        let publish_params = PublishResultParams {
            session_id: side_info.session_id,
            deployment_id: side_info.deployment_id,
            tool_call_event_id: side_info.tool_call_event_id,
            tool_call_id: side_info.tool_call_id,
            tool_name,
            outcome,
        };

        ctx.step("publish_result", publish_params, publish_result_step)
            .await?;

        result
    }
}

/// Step function to publish the tool result to the autopilot API.
async fn publish_result_step(
    params: PublishResultParams,
    state: ToolAppState,
) -> anyhow::Result<()> {
    let tensorzero_version = TENSORZERO_VERSION.to_string();

    state
        .t0_client()
        .create_autopilot_event(
            params.session_id,
            CreateEventRequest {
                deployment_id: params.deployment_id,
                tensorzero_version,
                payload: EventPayload::ToolResult {
                    tool_call_event_id: params.tool_call_event_id,
                    outcome: params.outcome,
                },
                previous_user_message_event_id: None,
            },
        )
        .await
        .map_err(|e| anyhow::anyhow!("Failed to publish tool result: {e}"))?;

    Ok(())
}
