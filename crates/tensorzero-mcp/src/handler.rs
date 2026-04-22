use std::sync::Arc;

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
#[expect(
    clippy::disallowed_types,
    reason = "MCP server holds SwappableAppStateData to load the latest config for each request"
)]
use tensorzero_core::utils::gateway::SwappableAppStateData;
use uuid::Uuid;

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
#[expect(
    clippy::disallowed_types,
    reason = "MCP server holds SwappableAppStateData to load the latest config for each request"
)]
pub(crate) struct TensorZeroMcpServer {
    #[expect(dead_code, reason = "retained for future tool implementations")]
    app_state: Arc<SwappableAppStateData>,
    tool_router: ToolRouter<Self>,
}

impl TensorZeroMcpServer {
    #[expect(
        clippy::disallowed_types,
        reason = "MCP server holds SwappableAppStateData to load the latest config for each request"
    )]
    pub fn new(app_state: Arc<SwappableAppStateData>, tool_router: ToolRouter<Self>) -> Self {
        Self {
            app_state,
            tool_router,
        }
    }
}

#[tool_handler(router = self.tool_router)]
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
    #[expect(
        clippy::disallowed_types,
        reason = "MCP tool visitor reads SwappableAppStateData once at initialization"
    )]
    fn new(app_state: &SwappableAppStateData) -> Self {
        Self {
            router: ToolRouter::new(),
            client: Arc::new(EmbeddedClient::new(app_state.load_latest())),
            heartbeater: Arc::new(NoopHeartbeater),
            registry: Arc::new(ToolRegistry::new()),
            config_snapshot_hash: app_state.config().load().hash.to_string(),
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

/// Build the MCP tool router by visiting all autopilot tools.
#[expect(
    clippy::disallowed_types,
    reason = "MCP tool router is initialized from SwappableAppStateData at startup"
)]
pub(crate) async fn build_tool_router(
    app_state: &SwappableAppStateData,
) -> Result<ToolRouter<TensorZeroMcpServer>, String> {
    let mut visitor = McpToolVisitor::new(app_state);
    autopilot_tools::for_each_tool(&mut visitor).await?;
    Ok(visitor.into_router())
}
