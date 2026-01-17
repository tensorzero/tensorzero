//! Wrapper that adds result publishing to client tools.

use std::borrow::Cow;
use std::marker::PhantomData;

use async_trait::async_trait;
use autopilot_client::AutopilotToolResult;
use autopilot_tools::AutopilotToolError;
use durable_tools::{
    CreateEventGatewayRequest, EventPayload, NonControlToolError, SimpleTool, SimpleToolContext,
    TaskTool, TensorZeroClient, ToolAppState, ToolContext, ToolError, ToolMetadata, ToolOutcome,
    ToolResult as DurableToolResult, ToolResultExt,
};
use schemars::Schema;
use serde::{Deserialize, Serialize};
use tensorzero_core::error::IMPOSSIBLE_ERROR_MESSAGE;
use uuid::Uuid;

use autopilot_client::AutopilotSideInfo;

/// Parameters for the publish_result step.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PublishResultParams {
    session_id: Uuid,
    tool_call_event_id: Uuid,
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
    type SideInfo = AutopilotSideInfo;
    /// The wrapped tool "returns" by writing to the autopilot API
    /// so for our purposes the output of the tool is ()
    type Output = ();
}

#[async_trait]
impl<T> TaskTool for ClientTaskToolWrapper<T>
where
    T: TaskTool,
    T::SideInfo: TryFrom<AutopilotSideInfo> + Serialize,
    <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error: std::fmt::Display,
{
    async fn execute(
        llm_params: Self::LlmParams,
        side_info: Self::SideInfo,
        ctx: &ToolContext,
    ) -> DurableToolResult<Self::Output> {
        let session_id = side_info.session_id;
        let tool_call_event_id = side_info.tool_call_event_id;
        let side_info: T::SideInfo = side_info.try_into().map_err(
            |e: <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error| {
                AutopilotToolError::validation(format!(
                    "Failed to convert AutopilotSideInfo to tool SideInfo: {e}"
                ))
            },
        )?;
        // Execute the underlying tool
        let result = T::execute(llm_params, side_info, ctx)
            .await
            .propagate_control()?;

        // Prepare the outcome for the autopilot API
        let tool_name = T::name().to_string();
        let outcome = match result {
            Ok(output) => {
                let result_json = serde_json::to_string(&output)?;
                ToolOutcome::Success(AutopilotToolResult {
                    result: result_json,
                })
            }
            Err(e) => ToolOutcome::Failure {
                error: tool_error_to_json(ToolError::NonControl(e)),
            },
        };

        // Publish result to autopilot API (checkpointed)
        let publish_params = PublishResultParams {
            session_id,
            tool_call_event_id,
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
    t0_client
        .create_autopilot_event(
            params.session_id,
            CreateEventGatewayRequest {
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

impl<T: SimpleTool> ToolMetadata for ClientSimpleToolWrapper<T>
where
    T::SideInfo: TryFrom<AutopilotSideInfo> + Serialize,
    <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error: std::fmt::Display,
{
    fn name() -> Cow<'static, str> {
        T::name()
    }

    fn description() -> Cow<'static, str> {
        T::description()
    }

    type LlmParams = T::LlmParams;
    type SideInfo = AutopilotSideInfo;
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
impl<T: SimpleTool> TaskTool for ClientSimpleToolWrapper<T>
where
    T::SideInfo: TryFrom<AutopilotSideInfo> + Serialize,
    <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error: std::fmt::Display,
{
    async fn execute(
        llm_params: Self::LlmParams,
        side_info: Self::SideInfo,
        ctx: &ToolContext,
    ) -> DurableToolResult<Self::Output> {
        let tool_name = T::name().to_string();
        let tool_call_event_id = side_info.tool_call_event_id;
        let session_id = side_info.session_id;
        // Convert AutopilotSideInfo to the underlying tool's SideInfo
        let converted_side_info: T::SideInfo = side_info.try_into().map_err(
            |e: <T::SideInfo as TryFrom<AutopilotSideInfo>>::Error| {
                AutopilotToolError::validation(format!(
                    "Failed to convert AutopilotSideInfo to tool SideInfo for tool {tool_name}: {e}"
                ))
            },
        )?;

        // Execute the underlying simple tool within a checkpointed step.
        // The step returns Ok(Result<output, error_json>) so tool errors are
        // checkpointed, not retried.
        let step_result: Result<T::Output, serde_json::Value> = ctx
            .step(
                "execute_simple_tool",
                SimpleToolStepParams {
                    llm_params,
                    side_info: converted_side_info,
                    tool_name: tool_name.clone(),
                    tool_call_event_id,
                },
                execute_simple_tool_step::<T>,
            )
            .await?;

        // Prepare the outcome for the autopilot API
        let outcome = match step_result {
            Ok(output) => {
                let result_json = serde_json::to_string(&output)?;
                ToolOutcome::Success(AutopilotToolResult {
                    result: result_json,
                })
            }
            Err(error) => ToolOutcome::Failure { error },
        };

        // Publish result to autopilot API (checkpointed)
        let publish_params = PublishResultParams {
            session_id,
            tool_call_event_id,
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
/// Returns `Ok(Result<output, error_json>)` where the inner result is either
/// success or failure. This ensures tool errors are checkpointed rather than
/// causing step retries. The error is converted to structured JSON for
/// programmatic parsing by the autopilot API.
async fn execute_simple_tool_step<T: SimpleTool>(
    params: SimpleToolStepParams<T::LlmParams, T::SideInfo>,
    state: ToolAppState,
) -> anyhow::Result<Result<T::Output, serde_json::Value>> {
    let simple_ctx = SimpleToolContext::new(state.pool(), state.t0_client());
    let idempotency_key = format!(
        "simple_tool:{}:{}",
        params.tool_name, params.tool_call_event_id
    );

    // Wrap the result in Ok so tool errors are checkpointed, not retried.
    // Convert ToolError to structured JSON for programmatic error handling.
    Ok(T::execute(
        params.llm_params,
        params.side_info,
        simple_ctx,
        &idempotency_key,
    )
    .await
    .map_err(tool_error_to_json))
}

/// Convert a `ToolError` to structured JSON for the autopilot API.
///
/// For `ToolError::Error`, serializes the `SerializableToolError` directly.
/// For `ToolError::Control`, returns a control flow error (this should not
/// normally happen as control flow errors are internal).
fn tool_error_to_json(e: ToolError) -> serde_json::Value {
    match e {
        ToolError::Control(cf) => {
            // Control flow errors should not be serialized - this is a bug if it happens
            let failure = ToolFailure::Control {
                message: format!(
                    "Unexpected control flow signal: {cf:?}. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            };
            serde_json::to_value(failure).unwrap_or_else(|e| {
                serde_json::json!({
                    "kind": "serialization",
                    "message": format!("Failed to serialize control flow error: {e}"),
                })
            })
        }
        ToolError::Database(db_error) => {
            let failure = ToolFailure::Database {
                message: db_error.to_string(),
            };
            serde_json::to_value(failure).unwrap_or_else(|e| {
                serde_json::json!({
                    "kind": "serialization",
                    "message": format!("Failed to serialize database error: {e}"),
                })
            })
        }
        ToolError::NonControl(non_control_error) => {
            let failure = ToolFailure::Tool {
                error: non_control_error,
            };
            serde_json::to_value(failure).unwrap_or_else(|e| {
                serde_json::json!({
                    "kind": "serialization",
                    "message": format!("Failed to serialize tool error: {e}"),
                })
            })
        }
        ToolError::Serialization(err) => {
            let failure = ToolFailure::Serialization {
                message: err.to_string(),
            };
            serde_json::to_value(failure).unwrap_or_else(|e| {
                serde_json::json!({
                    "kind": "serialization",
                    "message": format!("Failed to serialize serialization error: {e}. {IMPOSSIBLE_ERROR_MESSAGE}"),
                })
            })
        }
    }
}

/// This is the type that we write in ToolOutcome::Failure for tool errors.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ToolFailure {
    Control { message: String },
    Serialization { message: String },
    Tool { error: NonControlToolError },
    Database { message: String },
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
        GetConfigResponse, GetDatapointsResponse, GetInferencesResponse, InferenceResponse,
        ListDatapointsRequest, ListInferencesRequest, UpdateDatapointRequest,
        UpdateDatapointsResponse, WriteConfigRequest, WriteConfigResponse,
    };
    use tensorzero_core::config::snapshot::SnapshotHash;
    use tensorzero_core::db::feedback::FeedbackByVariant;
    use tensorzero_core::endpoints::feedback::internal::LatestFeedbackIdByMetricResponse;
    use tensorzero_core::optimization::{OptimizationJobHandle, OptimizationJobInfo};
    use tensorzero_optimizers::endpoints::LaunchOptimizationWorkflowParams;

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
                request: CreateEventGatewayRequest,
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

            /// List inferences with filtering and pagination.
            async fn list_inferences(
                &self,
                request: ListInferencesRequest,
            ) -> Result<GetInferencesResponse, TensorZeroClientError>;

            async fn launch_optimization_workflow(
                &self,
                params: LaunchOptimizationWorkflowParams,
            ) -> Result<OptimizationJobHandle, TensorZeroClientError>;

            async fn poll_optimization(
                &self,
                job_handle: &OptimizationJobHandle,
            ) -> Result<OptimizationJobInfo, TensorZeroClientError>;

            async fn get_latest_feedback_id_by_metric(
                &self,
                target_id: Uuid,
            ) -> Result<LatestFeedbackIdByMetricResponse, TensorZeroClientError>;

            async fn get_feedback_by_variant(
                &self,
                metric_name: String,
                function_name: String,
                variant_names: Option<Vec<String>>,
            ) -> Result<Vec<FeedbackByVariant>, TensorZeroClientError>;

            async fn run_evaluation(
                &self,
                params: durable_tools::RunEvaluationParams,
            ) -> Result<durable_tools::RunEvaluationResponse, TensorZeroClientError>;
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
            _ctx: &ToolContext,
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
        let tool_call_event_id = Uuid::now_v7();
        let tool_name = "test_tool".to_string();

        let mut mock_client = MockTensorZeroClient::new();

        // Capture the values we need to verify
        let expected_session_id = session_id;
        let expected_tool_call_event_id = tool_call_event_id;

        mock_client
            .expect_create_autopilot_event()
            .withf(move |sid, request| {
                *sid == expected_session_id
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
            tool_call_event_id,
            tool_name,
            outcome: ToolOutcome::Success(AutopilotToolResult {
                result: r#"{"result":"success"}"#.to_string(),
            }),
        };

        let result = publish_result(params, &mock_client).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_publish_result_failure_outcome() {
        let session_id = Uuid::now_v7();
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
                        outcome: ToolOutcome::Failure { error },
                    } if *tceid == expected_tool_call_event_id && error.get("message") == Some(&serde_json::json!("Tool execution failed"))
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
            tool_call_event_id,
            tool_name: "failing_tool".to_string(),
            outcome: ToolOutcome::Failure {
                error: serde_json::json!({ "kind": "TestError", "message": "Tool execution failed" }),
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
            tool_call_event_id: Uuid::now_v7(),
            tool_name: "some_tool".to_string(),
            outcome: ToolOutcome::Success(AutopilotToolResult {
                result: "{}".to_string(),
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
        fn assert_side_info_type<T: ToolMetadata<SideInfo = AutopilotSideInfo>>() {}
        assert_side_info_type::<ClientTaskToolWrapper<TestTaskTool>>();
    }

    #[test]
    fn test_client_simple_tool_wrapper_side_info_type() {
        // Verify that the wrapper wraps the SideInfo with AutopilotSideInfo
        fn assert_side_info_type<T: ToolMetadata<SideInfo = AutopilotSideInfo>>() {}
        assert_side_info_type::<ClientSimpleToolWrapper<TestSimpleTool>>();
    }
}
