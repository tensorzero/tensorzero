//! Wrapper that adds result publishing to client tools.

use std::borrow::Cow;
use std::marker::PhantomData;

use async_trait::async_trait;
use autopilot_client::ToolResult as AutopilotToolResult;
use durable_tools::{
    CreateEventRequest, EventPayload, SimpleTool, SimpleToolContext, TaskTool, TensorZeroClient,
    ToolAppState, ToolContext, ToolMetadata, ToolOutcome, ToolResult as DurableToolResult,
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
/// executor.register_task_tool::<ClientToolWrapper<MyTool>>().await;
/// ```
pub struct ClientToolWrapper<T: TaskTool> {
    _marker: PhantomData<T>,
}

impl<T: TaskTool> Default for ClientToolWrapper<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: TaskTool> ToolMetadata for ClientToolWrapper<T> {
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
impl<T: TaskTool> TaskTool for ClientToolWrapper<T> {
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
    publish_result(params, state.t0_client().as_ref()).await
}

/// Publish the tool result to the autopilot API.
async fn publish_result(
    params: PublishResultParams,
    t0_client: &dyn TensorZeroClient,
) -> anyhow::Result<()> {
    let tensorzero_version = TENSORZERO_VERSION.to_string();

    t0_client
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

/// Wrapper that promotes a [`SimpleTool`] to a [`TaskTool`] with autopilot side info
/// and result publishing.
///
/// This wrapper:
/// 1. Wraps the tool's `SideInfo` with [`AutopilotSideInfo`]
/// 2. Executes the underlying `SimpleTool` within a checkpointed step
/// 3. Publishes the result to the autopilot API
///
/// # Type Parameters
///
/// * `T` - The underlying SimpleTool to wrap
pub struct ClientSimpleToolWrapper<T: SimpleTool> {
    _marker: PhantomData<T>,
}

impl<T: SimpleTool> Default for ClientSimpleToolWrapper<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: SimpleTool> ToolMetadata for ClientSimpleToolWrapper<T> {
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

/// Parameters for executing a simple tool within a checkpointed step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimpleToolStepParams<L, S> {
    llm_params: L,
    side_info: S,
    tool_name: String,
    tool_call_event_id: Uuid,
}

#[async_trait]
impl<T: SimpleTool> TaskTool for ClientSimpleToolWrapper<T> {
    async fn execute(
        llm_params: Self::LlmParams,
        side_info: Self::SideInfo,
        ctx: &mut ToolContext<'_>,
    ) -> DurableToolResult<Self::Output> {
        let tool_name = T::name().to_string();

        // Execute the underlying simple tool within a checkpointed step
        let output: Self::Output = ctx
            .step(
                "execute_simple_tool",
                SimpleToolStepParams {
                    llm_params,
                    side_info: side_info.inner,
                    tool_name: tool_name.clone(),
                    tool_call_event_id: side_info.tool_call_event_id,
                },
                execute_simple_tool_step::<T>,
            )
            .await?;

        // Serialize output before the next await point
        let result_json = serde_json::to_string(&output)?;

        // Prepare the outcome for the autopilot API
        let outcome = ToolOutcome::Success(AutopilotToolResult {
            name: tool_name.clone(),
            result: result_json,
            id: side_info.tool_call_id.clone(),
        });

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

        Ok(output)
    }
}

/// Step function to execute a simple tool.
async fn execute_simple_tool_step<T: SimpleTool>(
    params: SimpleToolStepParams<T::LlmParams, T::SideInfo>,
    state: ToolAppState,
) -> anyhow::Result<T::Output> {
    let simple_ctx = SimpleToolContext::new(state.pool(), state.t0_client());
    let idempotency_key = format!(
        "simple_tool:{}:{}",
        params.tool_name, params.tool_call_event_id
    );

    T::execute(
        params.llm_params,
        params.side_info,
        simple_ctx,
        &idempotency_key,
    )
    .await
    .map_err(|e| anyhow::anyhow!("{e}"))
}
