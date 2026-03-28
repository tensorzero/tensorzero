use std::sync::Arc;

use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio_util::sync::CancellationToken;

use tensorzero_core::utils::gateway::AppStateData;

mod handler;

use handler::TensorZeroMcpServer;

/// Build an Axum router that serves the MCP streamable-HTTP service.
///
/// The returned `Router<()>` is intended to be nested on the gateway router
/// (e.g. via `nest_service("/mcp", ...)`) so MCP is served on the same port.
pub fn build_mcp_router(
    app_state: Arc<AppStateData>,
    shutdown_token: CancellationToken,
) -> axum::Router {
    let service = StreamableHttpService::new(
        move || Ok(TensorZeroMcpServer::new(app_state.clone())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default().with_cancellation_token(shutdown_token.child_token()),
    );

    axum::Router::new().nest_service("/", service)
}
