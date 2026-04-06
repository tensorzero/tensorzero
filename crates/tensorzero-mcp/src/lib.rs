use std::sync::Arc;

use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio_util::sync::CancellationToken;

use tensorzero_core::utils::gateway::SwappableAppStateData;

mod handler;

use handler::TensorZeroMcpServer;

/// Build an Axum router that serves the MCP streamable-HTTP service.
///
/// The returned `Router<()>` is intended to be nested on the gateway router
/// (e.g. via `nest_service("/mcp", ...)`) so MCP is served on the same port.
pub async fn build_mcp_router(
    app_state: Arc<SwappableAppStateData>,
    shutdown_token: CancellationToken,
) -> Result<axum::Router, String> {
    let tool_router = handler::build_tool_router(&app_state).await?;

    let service = StreamableHttpService::new(
        move || {
            Ok(TensorZeroMcpServer::new(
                app_state.clone(),
                tool_router.clone(),
            ))
        },
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default().with_cancellation_token(shutdown_token.child_token()),
    );

    Ok(axum::Router::new().fallback_service(service))
}
