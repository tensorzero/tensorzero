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
    /// The wrapped tool "returns" by writing to the autopilot API
    /// so for our purposes the output of the tool is ()
    type Output = ();
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

        Ok(())
    }
}

/// Step function to publish the tool result to the autopilot API.
/// This is a helper that gives the signature expected by `ToolContext::step`.
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

    type LlmParams = T::LlmParams;
    type SideInfo = AutopilotSideInfo<T::SideInfo>;
    /// The wrapped tool "returns" by writing to the autopilot API
    /// so for our purposes the output of the tool is ()
    type Output = ();
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

        // Execute the underlying simple tool within a checkpointed step.
        // The step returns Ok(Result<output, error_string>) so tool errors are
        // checkpointed, not retried.
        let step_result: Result<T::Output, String> = ctx
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

        // Prepare the outcome for the autopilot API
        let outcome = match &step_result {
            Ok(output) => {
                let result_json = serde_json::to_string(output)?;
                ToolOutcome::Success(AutopilotToolResult {
                    name: tool_name.clone(),
                    result: result_json,
                    id: side_info.tool_call_id.clone(),
                })
            }
            Err(error_message) => ToolOutcome::Failure {
                message: error_message.clone(),
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

        Ok(())
    }
}

/// Step function to execute a simple tool.
///
/// Returns `Ok(Result<output, error_message>)` where the inner result is either
/// success or failure. This ensures tool errors are checkpointed rather than
/// causing step retries. The error is converted to String since ToolError
/// contains non-serializable types.
async fn execute_simple_tool_step<T: SimpleTool>(
    params: SimpleToolStepParams<T::LlmParams, T::SideInfo>,
    state: ToolAppState,
) -> anyhow::Result<Result<T::Output, String>> {
    let simple_ctx = SimpleToolContext::new(state.pool(), state.t0_client());
    let idempotency_key = format!(
        "simple_tool:{}:{}",
        params.tool_name, params.tool_call_event_id
    );

    // Wrap the result in Ok so tool errors are checkpointed, not retried.
    // Convert ToolError to String since it contains non-serializable types.
    Ok(T::execute(
        params.llm_params,
        params.side_info,
        simple_ctx,
        &idempotency_key,
    )
    .await
    .map_err(|e| e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use durable_tools::{CreateEventResponse, TensorZeroClientError};
    use mockall::mock;
    use schemars::JsonSchema;
    use tensorzero::ActionInput;
    use tensorzero::{
        ClientInferenceParams, CreateDatapointRequest, CreateDatapointsFromInferenceRequestParams,
        CreateDatapointsResponse, DeleteDatapointsResponse, FeedbackParams, FeedbackResponse,
        GetConfigResponse, GetDatapointsResponse, InferenceResponse, ListDatapointsRequest,
        UpdateDatapointRequest, UpdateDatapointsResponse, WriteConfigRequest, WriteConfigResponse,
    };
    use tensorzero_core::config::snapshot::SnapshotHash;
    use tensorzero_core::endpoints::feedback::internal::LatestFeedbackIdByMetricResponse;

    // Mock TensorZeroClient using mockall::mock! macro
    // (same pattern as autopilot-tools/tests/common/mod.rs)
    mock! {
        pub TensorZeroClient {}

        #[async_trait]
        impl TensorZeroClient for TensorZeroClient {
            async fn inference(
                &self,
                params: ClientInferenceParams,
            ) -> Result<InferenceResponse, TensorZeroClientError>;

            async fn feedback(
                &self,
                params: FeedbackParams,
            ) -> Result<FeedbackResponse, TensorZeroClientError>;

            async fn create_autopilot_event(
                &self,
                session_id: Uuid,
                request: CreateEventRequest,
            ) -> Result<CreateEventResponse, TensorZeroClientError>;

            async fn list_autopilot_events(
                &self,
                session_id: Uuid,
                params: durable_tools::ListEventsParams,
            ) -> Result<durable_tools::ListEventsResponse, TensorZeroClientError>;

            async fn list_autopilot_sessions(
                &self,
                params: durable_tools::ListSessionsParams,
            ) -> Result<durable_tools::ListSessionsResponse, TensorZeroClientError>;

            async fn action(
                &self,
                snapshot_hash: SnapshotHash,
                params: ActionInput,
            ) -> Result<InferenceResponse, TensorZeroClientError>;

            async fn get_config_snapshot(
                &self,
                hash: Option<String>,
            ) -> Result<GetConfigResponse, TensorZeroClientError>;

            async fn write_config(
                &self,
                request: WriteConfigRequest,
            ) -> Result<WriteConfigResponse, TensorZeroClientError>;

            async fn create_datapoints(
                &self,
                dataset_name: String,
                datapoints: Vec<CreateDatapointRequest>,
            ) -> Result<CreateDatapointsResponse, TensorZeroClientError>;

            async fn create_datapoints_from_inferences(
                &self,
                dataset_name: String,
                params: CreateDatapointsFromInferenceRequestParams,
            ) -> Result<CreateDatapointsResponse, TensorZeroClientError>;

            async fn list_datapoints(
                &self,
                dataset_name: String,
                request: ListDatapointsRequest,
            ) -> Result<GetDatapointsResponse, TensorZeroClientError>;

            async fn get_datapoints(
                &self,
                dataset_name: Option<String>,
                ids: Vec<Uuid>,
            ) -> Result<GetDatapointsResponse, TensorZeroClientError>;

            async fn update_datapoints(
                &self,
                dataset_name: String,
                datapoints: Vec<UpdateDatapointRequest>,
            ) -> Result<UpdateDatapointsResponse, TensorZeroClientError>;

            async fn delete_datapoints(
                &self,
                dataset_name: String,
                ids: Vec<Uuid>,
            ) -> Result<DeleteDatapointsResponse, TensorZeroClientError>;

            async fn get_latest_feedback_id_by_metric(
                &self,
                target_id: Uuid,
            ) -> Result<LatestFeedbackIdByMetricResponse, TensorZeroClientError>;
        }
    }

    // ===== Test TaskTool for wrapper testing =====

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct TestTaskToolParams {
        message: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestTaskToolOutput {
        result: String,
    }

    #[derive(Default)]
    struct TestTaskTool;

    impl ToolMetadata for TestTaskTool {
        fn name() -> Cow<'static, str> {
            Cow::Borrowed("test_task_tool")
        }

        fn description() -> Cow<'static, str> {
            Cow::Borrowed("A test task tool for unit testing")
        }

        type LlmParams = TestTaskToolParams;
        type SideInfo = ();
        type Output = TestTaskToolOutput;
    }

    #[async_trait]
    impl TaskTool for TestTaskTool {
        async fn execute(
            llm_params: Self::LlmParams,
            _side_info: Self::SideInfo,
            _ctx: &mut ToolContext<'_>,
        ) -> DurableToolResult<Self::Output> {
            Ok(TestTaskToolOutput {
                result: format!("Processed: {}", llm_params.message),
            })
        }
    }

    // ===== Test SimpleTool for wrapper testing =====

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    struct TestSimpleToolParams {
        query: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestSimpleToolOutput {
        answer: String,
    }

    #[derive(Default)]
    struct TestSimpleTool;

    impl ToolMetadata for TestSimpleTool {
        fn name() -> Cow<'static, str> {
            Cow::Borrowed("test_simple_tool")
        }

        fn description() -> Cow<'static, str> {
            Cow::Borrowed("A test simple tool for unit testing")
        }

        type LlmParams = TestSimpleToolParams;
        type SideInfo = ();
        type Output = TestSimpleToolOutput;
    }

    #[async_trait]
    impl SimpleTool for TestSimpleTool {
        async fn execute(
            llm_params: Self::LlmParams,
            _side_info: Self::SideInfo,
            _ctx: SimpleToolContext<'_>,
            _idempotency_key: &str,
        ) -> DurableToolResult<Self::Output> {
            Ok(TestSimpleToolOutput {
                answer: format!("Answer to: {}", llm_params.query),
            })
        }
    }

    // ===== Tests for publish_result =====

    #[tokio::test]
    async fn test_publish_result_success_outcome() {
        let session_id = Uuid::now_v7();
        let deployment_id = Uuid::now_v7();
        let tool_call_event_id = Uuid::now_v7();
        let tool_call_id = "call_123".to_string();
        let tool_name = "test_tool".to_string();

        let mut mock_client = MockTensorZeroClient::new();

        // Capture the values we need to verify
        let expected_session_id = session_id;
        let expected_deployment_id = deployment_id;
        let expected_tool_call_event_id = tool_call_event_id;

        mock_client
            .expect_create_autopilot_event()
            .withf(move |sid, request| {
                *sid == expected_session_id
                    && request.deployment_id == expected_deployment_id
                    && matches!(
                        &request.payload,
                        EventPayload::ToolResult {
                            tool_call_event_id: tceid,
                            outcome: ToolOutcome::Success(_),
                        } if *tceid == expected_tool_call_event_id
                    )
            })
            .returning(|sid, _| {
                Ok(CreateEventResponse {
                    event_id: Uuid::now_v7(),
                    session_id: sid,
                })
            });

        let params = PublishResultParams {
            session_id,
            deployment_id,
            tool_call_event_id,
            tool_call_id,
            tool_name,
            outcome: ToolOutcome::Success(AutopilotToolResult {
                name: "test_tool".to_string(),
                result: r#"{"result":"success"}"#.to_string(),
                id: "call_123".to_string(),
            }),
        };

        let result = publish_result(params, &mock_client).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_publish_result_failure_outcome() {
        let session_id = Uuid::now_v7();
        let deployment_id = Uuid::now_v7();
        let tool_call_event_id = Uuid::now_v7();

        let mut mock_client = MockTensorZeroClient::new();

        let expected_tool_call_event_id = tool_call_event_id;

        mock_client
            .expect_create_autopilot_event()
            .withf(move |_sid, request| {
                matches!(
                    &request.payload,
                    EventPayload::ToolResult {
                        tool_call_event_id: tceid,
                        outcome: ToolOutcome::Failure { message },
                    } if *tceid == expected_tool_call_event_id && message == "Tool execution failed"
                )
            })
            .returning(|sid, _| {
                Ok(CreateEventResponse {
                    event_id: Uuid::now_v7(),
                    session_id: sid,
                })
            });

        let params = PublishResultParams {
            session_id,
            deployment_id,
            tool_call_event_id,
            tool_call_id: "call_456".to_string(),
            tool_name: "failing_tool".to_string(),
            outcome: ToolOutcome::Failure {
                message: "Tool execution failed".to_string(),
            },
        };

        let result = publish_result(params, &mock_client).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_publish_result_client_error() {
        let mut mock_client = MockTensorZeroClient::new();

        mock_client
            .expect_create_autopilot_event()
            .returning(|_, _| Err(TensorZeroClientError::AutopilotUnavailable));

        let params = PublishResultParams {
            session_id: Uuid::now_v7(),
            deployment_id: Uuid::now_v7(),
            tool_call_event_id: Uuid::now_v7(),
            tool_call_id: "call_789".to_string(),
            tool_name: "some_tool".to_string(),
            outcome: ToolOutcome::Success(AutopilotToolResult {
                name: "some_tool".to_string(),
                result: "{}".to_string(),
                id: "call_789".to_string(),
            }),
        };

        let result = publish_result(params, &mock_client).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to publish tool result")
        );
    }

    // ===== Tests for metadata delegation =====

    #[test]
    fn test_client_tool_wrapper_metadata_delegation() {
        assert_eq!(
            ClientTaskToolWrapper::<TestTaskTool>::name(),
            "test_task_tool"
        );
        assert_eq!(
            ClientTaskToolWrapper::<TestTaskTool>::description(),
            "A test task tool for unit testing"
        );
    }

    #[test]
    fn test_client_simple_tool_wrapper_metadata_delegation() {
        assert_eq!(
            ClientSimpleToolWrapper::<TestSimpleTool>::name(),
            "test_simple_tool"
        );
        assert_eq!(
            ClientSimpleToolWrapper::<TestSimpleTool>::description(),
            "A test simple tool for unit testing"
        );
    }

    #[test]
    fn test_client_tool_wrapper_side_info_type() {
        // Verify that the wrapper wraps the SideInfo with AutopilotSideInfo
        // This is a compile-time check - if it compiles, the types are correct
        fn assert_side_info_type<T: ToolMetadata<SideInfo = AutopilotSideInfo<()>>>() {}
        assert_side_info_type::<ClientTaskToolWrapper<TestTaskTool>>();
    }

    #[test]
    fn test_client_simple_tool_wrapper_side_info_type() {
        // Verify that the wrapper wraps the SideInfo with AutopilotSideInfo
        fn assert_side_info_type<T: ToolMetadata<SideInfo = AutopilotSideInfo<()>>>() {}
        assert_side_info_type::<ClientSimpleToolWrapper<TestSimpleTool>>();
    }
}
