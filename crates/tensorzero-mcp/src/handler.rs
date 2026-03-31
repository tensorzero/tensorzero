use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use autopilot_client::AutopilotSideInfo;
use autopilot_tools::ToolVisitor;
use durable_tools::tensorzero_client::EmbeddedClient;
use durable_tools::{
    Heartbeater, NoopHeartbeater, SimpleTool, SimpleToolContext, TaskTool, TensorZeroClient,
    ToolMetadata, ToolRegistry,
};
use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{
        router::tool::{ToolRoute, ToolRouter},
        tool::ToolCallContext,
    },
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo, Tool},
    tool_handler,
};
use tensorzero_core::utils::gateway::AppStateData;
use uuid::Uuid;

use autopilot_tools::tools::prod::{
    LaunchOptimizationWorkflowToolOutput, LaunchOptimizationWorkflowToolParams,
};
use tensorzero_core::optimization::OptimizationJobInfo;
use tensorzero_core::optimization::gepa::GepaGetResponse;
use tensorzero_optimizers::endpoints::{
    LaunchOptimizationWorkflowParams, OptimizationDataSource, launch_optimization_workflow,
    poll_optimization,
};
use tensorzero_optimizers::gepa::durable::types::{GepaToolOutput, GepaToolParams};

/// TODO - refactor the way we wrap client-side tool so that we can avoid
/// constructing a dummy side info
fn dummy_side_info(config_snapshot_hash: String) -> AutopilotSideInfo {
    AutopilotSideInfo {
        tool_call_event_id: Uuid::nil(),
        session_id: Uuid::nil(),
        config_snapshot_hash,
        optimization: Default::default(),
    }
}

#[derive(Clone)]
pub(crate) struct TensorZeroMcpServer {
    #[expect(dead_code, reason = "retained for future tool implementations")]
    app_state: Arc<AppStateData>,
    tool_router: ToolRouter<Self>,
}

impl TensorZeroMcpServer {
    pub fn new(app_state: Arc<AppStateData>, tool_router: ToolRouter<Self>) -> Self {
        Self {
            app_state,
            tool_router,
        }
    }
}

#[tool_handler]
impl ServerHandler for TensorZeroMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("TensorZero", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "TensorZero MCP Server - query observability data from TensorZero".to_string(),
            )
    }
}

/// Visitor that registers autopilot tool metadata as MCP tool routes.
///
/// Each tool's name, description, and parameter schema are extracted from its
/// `ToolMetadata` implementation and registered on a `ToolRouter`. For `SimpleTool`s,
/// the handler calls `T::execute` directly using an embedded TensorZero client.
/// `TaskTool`s are skipped for now.
struct McpToolVisitor {
    router: ToolRouter<TensorZeroMcpServer>,
    client: Arc<dyn TensorZeroClient>,
    heartbeater: Arc<dyn Heartbeater>,
    registry: Arc<ToolRegistry>,
    config_snapshot_hash: String,
}

impl McpToolVisitor {
    fn new(app_state: &AppStateData) -> Self {
        Self {
            router: ToolRouter::new(),
            client: Arc::new(EmbeddedClient::new(app_state.clone())),
            heartbeater: Arc::new(NoopHeartbeater),
            registry: Arc::new(ToolRegistry::new()),
            config_snapshot_hash: app_state.config.hash.to_string(),
        }
    }

    fn into_router(self) -> ToolRouter<TensorZeroMcpServer> {
        self.router
    }
}

/// Build an rmcp `Tool` definition from a `ToolMetadata` impl.
fn tool_attr_from_metadata(tool: &impl ToolMetadata) -> Result<Tool, String> {
    let name = tool.name();
    let description = tool.description();
    let schema = tool
        .parameters_schema()
        .map_err(|e| format!("Failed to generate schema for tool `{name}`: {e:?}"))?;
    let schema_value = serde_json::to_value(&schema)
        .map_err(|e| format!("Failed to serialize schema for tool `{name}`: {e}"))?;
    let schema_object = match schema_value {
        serde_json::Value::Object(map) => map,
        other => {
            return Err(format!(
                "Schema for tool `{name}` is not a JSON object: {other}"
            ));
        }
    };
    Ok(Tool::new(name, description, Arc::new(schema_object)))
}

#[async_trait]
impl ToolVisitor for McpToolVisitor {
    type Error = String;

    async fn visit_task_tool<T>(&mut self, _tool: T) -> Result<(), String>
    where
        T: TaskTool<SideInfo = AutopilotSideInfo, ExtraState = ()>,
    {
        // TaskTools are skipped for now — they require durable execution context.
        Ok(())
    }

    async fn visit_simple_tool<T>(&mut self) -> Result<(), String>
    where
        T: SimpleTool<SideInfo = AutopilotSideInfo> + Default,
    {
        let tool_attr = tool_attr_from_metadata(&T::default())?;
        let client = self.client.clone();
        let heartbeater = self.heartbeater.clone();
        let registry = self.registry.clone();
        let config_snapshot_hash = self.config_snapshot_hash.clone();
        let route = ToolRoute::new_dyn(
            tool_attr,
            move |ctx: ToolCallContext<'_, TensorZeroMcpServer>| {
                let client = client.clone();
                let heartbeater = heartbeater.clone();
                let registry = registry.clone();
                let config_snapshot_hash = config_snapshot_hash.clone();
                Box::pin(async move {
                    let arguments = ctx.arguments.unwrap_or_default();

                    let params: <T as ToolMetadata>::LlmParams =
                        serde_json::from_value(serde_json::Value::Object(arguments)).map_err(
                            |e| McpError::invalid_params(format!("Invalid parameters: {e}"), None),
                        )?;

                    let simple_ctx =
                        SimpleToolContext::new_without_pool(&client, &heartbeater, &registry);

                    let idempotency_key = Uuid::now_v7().to_string();
                    let result = T::execute(
                        params,
                        dummy_side_info(config_snapshot_hash),
                        simple_ctx,
                        &idempotency_key,
                    )
                    .await;

                    match result {
                        Ok(output) => {
                            let json = serde_json::to_string(&output).map_err(|e| {
                                McpError::internal_error(
                                    format!("Failed to serialize response: {e}"),
                                    None,
                                )
                            })?;
                            Ok(CallToolResult::success(vec![Content::text(json)]))
                        }
                        Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
                    }
                })
            },
        );
        self.router.add_route(route);
        Ok(())
    }

    async fn visit_standalone_task_tool<T>(&mut self, _tool: T) -> Result<(), String>
    where
        T: TaskTool<SideInfo = (), ExtraState = ()>,
    {
        // Standalone TaskTools are skipped for now.
        Ok(())
    }
}

async fn handle_launch_optimization(
    app_state: Arc<AppStateData>,
    params: LaunchOptimizationWorkflowToolParams,
) -> Result<CallToolResult, McpError> {
    let data_source = OptimizationDataSource::from_flat_fields(
        params.output_source,
        params.dataset_name,
        params.query_variant_name,
        params.filters,
        params.order_by,
        params.limit,
        params.offset,
    )
    .map_err(|e| McpError::invalid_params(e, None))?;

    let launch_params = LaunchOptimizationWorkflowParams {
        function_name: params.function_name,
        template_variant_name: params.template_variant_name,
        data_source,
        val_fraction: params.val_fraction,
        optimizer_config: params.optimizer_config,
    };

    let db: Arc<
        dyn tensorzero_core::db::delegating_connection::DelegatingDatabaseQueries + Send + Sync,
    > = Arc::new(app_state.get_delegating_database());

    let job_handle = launch_optimization_workflow(
        &app_state.http_client,
        app_state.config.clone(),
        &db,
        launch_params,
    )
    .await
    .map_err(|e| McpError::internal_error(format!("Failed to launch optimization: {e}"), None))?;

    // Poll until completion with exponential backoff
    let mut poll_interval = Duration::from_secs(2);
    let max_poll_interval = Duration::from_secs(30);

    loop {
        tokio::time::sleep(poll_interval).await;

        let status = poll_optimization(
            &app_state.http_client,
            &job_handle,
            &app_state.config.models.default_credentials,
            &app_state.config.provider_types,
        )
        .await
        .map_err(|e| McpError::internal_error(format!("Failed to poll optimization: {e}"), None))?;

        match &status {
            OptimizationJobInfo::Completed { .. } | OptimizationJobInfo::Failed { .. } => {
                let response = LaunchOptimizationWorkflowToolOutput { result: status };
                let json = serde_json::to_string(&response).map_err(|e| {
                    McpError::internal_error(format!("Failed to serialize response: {e}"), None)
                })?;
                return Ok(CallToolResult::success(vec![Content::text(json)]));
            }
            OptimizationJobInfo::Pending { .. } => {
                poll_interval = (poll_interval * 2).min(max_poll_interval);
            }
        }
    }
}

/// Build a `ToolRoute` for the `launch_optimization` tool.
///
/// This is a TaskTool that cannot be auto-registered via the visitor pattern,
/// so we register it manually.
fn build_launch_optimization_route(
    app_state: &Arc<AppStateData>,
) -> Result<ToolRoute<TensorZeroMcpServer>, String> {
    let schema = schemars::schema_for!(LaunchOptimizationWorkflowToolParams);
    let schema_value = serde_json::to_value(&schema)
        .map_err(|e| format!("Failed to serialize launch_optimization schema: {e}"))?;
    let schema_object = match schema_value {
        serde_json::Value::Object(map) => map,
        other => {
            return Err(format!(
                "Schema for launch_optimization is not a JSON object: {other}"
            ));
        }
    };
    let tool_attr = Tool::new(
        "launch_optimization",
        "Launch an optimization workflow (fine-tuning, prompt optimization, etc.) using stored inferences or a dataset, and poll until completion. Returns the final optimization result.",
        Arc::new(schema_object),
    );

    let app_state = app_state.clone();
    Ok(ToolRoute::new_dyn(
        tool_attr,
        move |ctx: ToolCallContext<'_, TensorZeroMcpServer>| {
            let app_state = app_state.clone();
            Box::pin(async move {
                let arguments = ctx.arguments.unwrap_or_default();
                let params: LaunchOptimizationWorkflowToolParams =
                    serde_json::from_value(serde_json::Value::Object(arguments)).map_err(|e| {
                        McpError::invalid_params(format!("Invalid parameters: {e}"), None)
                    })?;
                handle_launch_optimization(app_state, params).await
            })
        },
    ))
}

async fn handle_launch_gepa(
    app_state: Arc<AppStateData>,
    params: GepaToolParams,
) -> Result<CallToolResult, McpError> {
    let spawn_client = app_state.spawn_client.as_ref().ok_or_else(|| {
        McpError::internal_error(
            "GEPA requires Postgres and a durable task queue to be configured".to_string(),
            None,
        )
    })?;

    // Validate evaluation_name is provided (inline evaluators not yet supported)
    if params.evaluation_name.is_none() {
        return Ok(CallToolResult::error(vec![Content::text(
            "Inline `evaluators` mode is not yet supported; provide `evaluation_name`",
        )]));
    }

    let llm_params = serde_json::to_value(&params).map_err(|e| {
        McpError::internal_error(format!("Failed to serialize GEPA params: {e}"), None)
    })?;

    let episode_id = uuid::Uuid::now_v7();

    let spawn_result = spawn_client
        .spawn_tool_by_name(
            "standalone_gepa",
            llm_params,
            serde_json::json!(null),
            episode_id,
            durable_tools_spawn::SpawnOptions::default(),
        )
        .await
        .map_err(|e| McpError::internal_error(format!("Failed to spawn GEPA task: {e}"), None))?;

    let task_id = spawn_result.task_id;

    // Poll until completion with exponential backoff
    let mut poll_interval = Duration::from_secs(5);
    let max_poll_interval = Duration::from_secs(60);

    loop {
        tokio::time::sleep(poll_interval).await;

        let poll_result = spawn_client.get_task_result(task_id).await.map_err(|e| {
            McpError::internal_error(format!("Failed to poll GEPA task: {e}"), None)
        })?;

        use durable_tools_spawn::TaskStatus;

        match poll_result.status {
            TaskStatus::Completed => {
                let response = if let Some(result_value) = poll_result.result {
                    let output: GepaToolOutput =
                        serde_json::from_value(result_value).map_err(|e| {
                            McpError::internal_error(
                                format!("Failed to deserialize GEPA result: {e}"),
                                None,
                            )
                        })?;
                    GepaGetResponse::Completed {
                        variants: output.variants,
                        statistics: output.statistics,
                    }
                } else {
                    GepaGetResponse::Error {
                        error: "Task completed but no result payload found".to_string(),
                    }
                };

                let json = serde_json::to_string(&response).map_err(|e| {
                    McpError::internal_error(format!("Failed to serialize response: {e}"), None)
                })?;
                return Ok(CallToolResult::success(vec![Content::text(json)]));
            }
            TaskStatus::Failed | TaskStatus::Cancelled => {
                let error = poll_result
                    .error
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "Unknown error".to_string());
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "GEPA task failed: {error}"
                ))]));
            }
            TaskStatus::Pending | TaskStatus::Running | TaskStatus::Sleeping => {
                poll_interval = (poll_interval * 2).min(max_poll_interval);
            }
        }
    }
}

/// Build a `ToolRoute` for the `launch_gepa` tool.
///
/// This is a standalone TaskTool that cannot be auto-registered via the visitor pattern,
/// so we register it manually.
fn build_launch_gepa_route(
    app_state: &Arc<AppStateData>,
) -> Result<ToolRoute<TensorZeroMcpServer>, String> {
    let schema = schemars::schema_for!(GepaToolParams);
    let schema_value = serde_json::to_value(&schema)
        .map_err(|e| format!("Failed to serialize launch_gepa schema: {e}"))?;
    let schema_object = match schema_value {
        serde_json::Value::Object(map) => map,
        other => {
            return Err(format!(
                "Schema for launch_gepa is not a JSON object: {other}"
            ));
        }
    };
    let tool_attr = Tool::new(
        "launch_gepa",
        "Launch GEPA (Genetic Evolution with Pareto Analysis) prompt optimization via durable execution and wait for completion. Returns optimized variants and per-evaluator statistics.",
        Arc::new(schema_object),
    );

    let app_state = app_state.clone();
    Ok(ToolRoute::new_dyn(
        tool_attr,
        move |ctx: ToolCallContext<'_, TensorZeroMcpServer>| {
            let app_state = app_state.clone();
            Box::pin(async move {
                let arguments = ctx.arguments.unwrap_or_default();
                let params: GepaToolParams =
                    serde_json::from_value(serde_json::Value::Object(arguments)).map_err(|e| {
                        McpError::invalid_params(format!("Invalid parameters: {e}"), None)
                    })?;
                handle_launch_gepa(app_state, params).await
            })
        },
    ))
}

/// Build the MCP tool router by visiting all autopilot tools.
pub(crate) async fn build_tool_router(
    app_state: &Arc<AppStateData>,
) -> Result<ToolRouter<TensorZeroMcpServer>, String> {
    let mut visitor = McpToolVisitor::new(app_state);
    autopilot_tools::for_each_tool(&mut visitor).await?;

    // Manually register TaskTool-based tools that the visitor skips.
    let launch_optimization_route = build_launch_optimization_route(app_state)?;
    visitor.router.add_route(launch_optimization_route);

    let launch_gepa_route = build_launch_gepa_route(app_state)?;
    visitor.router.add_route(launch_gepa_route);

    Ok(visitor.into_router())
}
