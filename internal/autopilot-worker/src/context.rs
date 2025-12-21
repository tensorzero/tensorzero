//! Tool execution context for autopilot tools.

use durable_tools::{ClientInferenceParams, InferenceResponse, ToolContext};
use sqlx::PgPool;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::utils::gateway::AppStateData;

use crate::error::{AutopilotToolError, AutopilotToolResult};

/// Context provided to `ExecutableClientTool` execution.
///
/// Wraps durable's `ToolContext` and provides additional access to gateway state
/// for calling TensorZero endpoints.
pub struct AutopilotToolContext<'a, 'b> {
    /// The underlying durable tool context.
    tool_ctx: &'a mut ToolContext<'b>,
    /// Gateway state for accessing TensorZero functionality.
    gateway_state: &'a AppStateData,
}

impl<'a, 'b> AutopilotToolContext<'a, 'b> {
    /// Create a new autopilot tool context.
    pub fn new(tool_ctx: &'a mut ToolContext<'b>, gateway_state: &'a AppStateData) -> Self {
        Self {
            tool_ctx,
            gateway_state,
        }
    }

    /// Get the underlying durable tool context.
    ///
    /// Use this for advanced operations like checkpointing, calling other tools, etc.
    pub fn tool_ctx(&mut self) -> &mut ToolContext<'b> {
        self.tool_ctx
    }

    /// Get a reference to the database pool.
    pub fn pool(&self) -> &PgPool {
        self.tool_ctx.pool()
    }

    /// Get a reference to the ClickHouse connection info.
    pub fn clickhouse(&self) -> &ClickHouseConnectionInfo {
        &self.gateway_state.clickhouse_connection_info
    }

    /// Get a reference to the gateway state.
    pub fn gateway_state(&self) -> &AppStateData {
        self.gateway_state
    }

    /// Call TensorZero inference with full parameter control.
    ///
    /// This is a checkpointed operation - results are cached on restart.
    /// Streaming inference is not supported and will return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the inference call fails.
    pub async fn inference(
        &mut self,
        params: ClientInferenceParams,
    ) -> AutopilotToolResult<InferenceResponse> {
        self.tool_ctx
            .inference(params)
            .await
            .map_err(AutopilotToolError::Tool)
    }
}
